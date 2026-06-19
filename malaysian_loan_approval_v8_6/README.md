# Malaysian Loan Approval System

GA-optimised fuzzy inference engine for loan risk, with an AI reasoning agent and a
continuous-learning (auto-retraining) loop. Flask app, deployed on Google Cloud Run.

## What it does

- **Loan Assessment Agent** (user-facing): observe -> reason -> call the fuzzy engine (as a tool) -> explain -> act. Produces APPROVE / REVIEW / REJECT with a risk score (0-100), risk-factor profile, and what-if advice. Both the web app and the Telegram bot use the SAME 3-provider tool-calling chain: Gemini (primary) -> Cerebras gpt-oss-120b -> Groq llama-3.3, returning the first successful reply. Every leg really calls the fuzzy engine as a tool. The decision/score are always computed deterministically by the engine and stay authoritative; the agent only writes the explanation. If every configured provider fails, the app falls back to a local deterministic summary (no model names, no tracebacks) so the result is never lost.
- **Retraining Agent** (system-facing): the only component with autonomous judgement. It decides *whether* to retrain and *whether* to promote a new model.

## Continuous-learning loop

```
labelled data (admin upload)  -->  Gate 1 outlier test --fail--> manual review queue
                                          | pass
                                   candidate pool (GCS)
                                          |
        trigger: Cloud Scheduler hourly  OR  admin "Retrain now"
                                          |  (status lock prevents overlap)
                              Retraining Agent
                                |- check_data_freshness()    >= 30 new rows?
                                |- read_and_validate_data()  Gate 2 quality --fail--> review
                                |- retrain_fuzzy_engine()    GA -> candidate params (async)
                                |- evaluate_and_promote()    new vs old MSE on FIXED
                                                             validation set -> promote only if better
                                          | promote
                  GCS: mf_params_v{n}.json + changelog + model_v{n}.zip (history kept)
```

- **Fair comparison.** New and old models are scored with **MSE on a fixed held-out validation slice** (seeded, never trained on). Promote only if new MSE < old MSE.
- **Versioning.** Every promotion writes a versioned params record, a changelog entry, and a downloadable zip. The admin page supports one-click **rollback** to any version.
- **The fuzzy engine code never changes.** fuzzy_engine.py, the GA logic, MF formulas, fuzzy rules, risk thresholds and loan_data.csv are untouched. Only the data-derived **MF breakpoint parameters** are versioned and swapped.

### Model activation note

Promoted models are stored as versioned **artifacts** and are also applied to the live fuzzy engine. On boot, `app.py` reads `active/mf_params.json` from GCS/local storage and calls `fuzzy_engine.apply_model_params()` so the served single-applicant decision path uses the active version. After a successful promotion or rollback, the current Cloud Run instance applies the selected MF breakpoints immediately. Existing fuzzy rules, risk thresholds, CTOS/default logic and GA math are unchanged; only the GA-derived income/ratio/employment MF breakpoints are swapped.

For multi-instance Cloud Run production, other already-running instances pick up the active artifact on restart/redeploy. For demos or strict single-version consistency, use one active instance/max instance 1, or roll a new revision after promotion.

## Cloud Run notes

- **Persistence.** Cloud Run's filesystem is ephemeral, so all state (pending data, review queue, versions, changelog, status lock) is stored in **Google Cloud Storage**. Without GCS_BUCKET set, the app falls back to a local ./_store directory (dev/demo).
- **Triggering.** Cloud Run cannot run a background watcher (it scales to zero). Use **Cloud Scheduler** to POST /cron/check-retrain hourly; it checks the new-row count and triggers retraining when the threshold is met. A persisted status lock (running/idle) prevents a second run from starting while one is in progress.
- **Long GA runs.** Retraining is dispatched through **Cloud Tasks** when `CLOUD_TASKS_QUEUE`, `CLOUD_TASKS_LOCATION`, and `SERVICE_URL` are set. `/api/admin/retrain` and `/cron/check-retrain` only queue the job and return quickly; `/tasks/retrain` is the worker endpoint that runs the GA. If Cloud Tasks is not configured, the app falls back to a daemon thread for local/demo, and Cloud Run should use CPU always-allocated for that fallback.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| GEMINI_API_KEY | Primary reasoning provider (required for the agent) | - |
| GROQ_API_KEY | Fallback reasoning provider (3rd leg of the chain) | - |
| CEREBRAS_API_KEY | Fallback reasoning provider (2nd leg, OpenAI-compatible) | - |
| GEMINI_MODEL | Gemini model | gemini-3.1-flash-lite |
| CEREBRAS_MODEL | Cerebras model | gpt-oss-120b |
| GROQ_CHAT_MODEL | Groq tool-calling chat model | llama-3.3-70b-versatile |
| GCS_BUCKET | Cloud Storage bucket for all state (prod) | local ./_store |
| ADMIN_TOKEN | Protects /admin, /api/admin/*, /cron/check-retrain and /tasks/retrain (blank = open) | blank |
| RETRAIN_THRESHOLD | New clean rows needed to trigger retraining | 30 |
| RETRAIN_GA_GENS | GA generations for retraining | 40 |
| STORE_DIR | Local store path when not using GCS | ./_store |
| CLOUD_TASKS_QUEUE | Cloud Tasks queue name for production async GA | blank = thread fallback |
| CLOUD_TASKS_LOCATION | Cloud Tasks queue region, e.g. asia-southeast1 | blank = thread fallback |
| SERVICE_URL | Deployed Cloud Run URL used to build /tasks/retrain target | blank = thread fallback |
| CLOUD_TASKS_SERVICE_ACCOUNT | Optional service account for OIDC-authenticated Cloud Tasks calls | blank |

## Admin page (/admin)

Upload labelled CSV, monitor retraining status, view the candidate pool and manual review queue, browse/rollback/download model versions, run a manual retrain, and read the update log.

Required CSV columns: person_income, credit_score, loan_percent_income, person_emp_exp, previous_loan_defaults_on_file, loan_status.

## Cloud Tasks setup (production async GA)

```
gcloud services enable cloudtasks.googleapis.com
gcloud tasks queues create loan-retrain-queue --location=asia-southeast1
```

Set these Cloud Run env vars:

```
CLOUD_TASKS_QUEUE=loan-retrain-queue
CLOUD_TASKS_LOCATION=asia-southeast1
SERVICE_URL=https://<your-cloud-run-url>
```

If these are missing, retraining still works locally/demo via thread fallback.

## Cloud Scheduler setup (example)

If `ADMIN_TOKEN` is blank, the scheduler can call the cron route directly. If `ADMIN_TOKEN` is enabled, include the same token in the request header:

```
gcloud scheduler jobs create http retrain-check \
  --schedule="0 * * * *" \
  --uri="https://<your-cloud-run-url>/cron/check-retrain" \
  --http-method=POST \
  --headers="X-Admin-Token=<your-admin-token>"
```

## Run locally

```
pip install -r requirements.txt
export GEMINI_API_KEY=...        # required for the AI agent
python app.py                    # http://localhost:8080  (GA runs ~15s on boot)
```
