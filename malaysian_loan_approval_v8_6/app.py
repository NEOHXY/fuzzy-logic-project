"""Malaysian Loan Approval System — Flask + Cloud Run

This replaces the Streamlit UI with a normal Flask web app while keeping the
existing fuzzy_engine.py logic unchanged. The app exposes a compact dashboard,
a real sidebar, an integrated AI-agent process animation, GA results, and
membership-function visualisation.
"""

from __future__ import annotations

import json
import os
import hmac
import traceback
import urllib.request
import math
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request
import numpy as np

try:
    import requests
except Exception:
    requests = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
except Exception:
    plt = None
    Wedge = None
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from google import genai
    from google.genai import types
except Exception:  # google-genai is optional; Gemini is the primary LLM provider
    genai = None
    types = None

try:
    from groq import Groq
except Exception:  # Groq is optional; used only if Gemini fails
    Groq = None

import io
import csv
import threading
import collections
import uuid

try:
    from google.cloud import tasks_v2  # type: ignore
except Exception:
    tasks_v2 = None

import storage
import retraining

# Simple admin gate (set ADMIN_TOKEN in Cloud Run; blank = open for local/demo)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

def _admin_ok(req) -> bool:
    if not ADMIN_TOKEN:
        return True
    bearer = req.headers.get("Authorization", "")
    if bearer.lower().startswith("bearer "):
        bearer = bearer.split(" ", 1)[1].strip()
    # Constant-time comparison; only accept the token via headers (never via the
    # query string, which would leak into access logs / Referer).
    candidate = req.headers.get("X-Admin-Token") or bearer or ""
    return hmac.compare_digest(candidate, ADMIN_TOKEN)


def _admin_ok_nav(req) -> bool:
    """Auth for unavoidable browser navigations (file-download links) that
    cannot set request headers. Accepts the header forms OR a ?token= query
    param. The bulk API surface uses the stricter header-only _admin_ok().
    """
    if _admin_ok(req):
        return True
    if not ADMIN_TOKEN:
        return True
    return hmac.compare_digest(req.args.get("token", ""), ADMIN_TOKEN)


def _run_retraining_job(force: bool, job_id: str | None = None, owner_token: str | None = None):
    """Run the Retraining Agent synchronously inside the worker context.

    In the final architecture this function is called by /tasks/retrain after
    Cloud Tasks dispatches the job. If Cloud Tasks is not configured, the same
    function is used by the local/demo thread fallback.

    Phase 4: while running, a side thread pumps the lease heartbeat (so a long but
    live run is never judged stale); the lease is released on exit.
    """
    stop_beat = threading.Event()
    if owner_token:
        def _beat():
            while not stop_beat.wait(storage.HEARTBEAT_INTERVAL_SECONDS):
                if not storage.heartbeat_retrain_lock(owner_token):
                    break  # taken over / released -> stop beating
        threading.Thread(target=_beat, daemon=True).start()
    try:
        status = storage.get_status()
        if job_id and status.get("job_id") and status.get("job_id") != job_id:
            return {"action": "skip", "reason": "stale_job", "job_id": job_id}

        result = retraining.run_retraining_agent(force=force, owner_token=owner_token)

        # If a model was promoted, apply it to the current Cloud Run instance immediately.
        # Other instances will pick it up on restart/redeploy; rollback also applies below.
        if isinstance(result, dict) and result.get("action") == "promote":
            active_params, active_version = retraining.active_params_or_default()
            if active_version and hasattr(fe, "apply_model_params"):
                fe.apply_model_params(active_params, source=f"promoted artifact v{active_version}")
        return result
    finally:
        stop_beat.set()
        if owner_token:
            storage.release_retrain_lock(owner_token)


def _run_retraining_thread(force: bool, job_id: str, owner_token: str | None = None):
    """Demo/local fallback when Cloud Tasks is not configured."""
    def worker():
        try:
            result = _run_retraining_job(force=force, job_id=job_id, owner_token=owner_token)
            follow = maybe_chain_retraining_after_run(result, source="chain_thread")
            if follow.get("triggered"):
                print(f"[retrain] follow-up retrain queued: {follow.get('run_id')}", flush=True)
        except Exception as exc:  # never leak a traceback to the client
            try:
                storage.restore_current_batch(job_id, reason="worker failed")
            except Exception:
                pass
            storage.set_status("idle", "Retraining failed — locked batch restored for retry.", extra={"job_id": job_id})
            if owner_token:
                storage.release_retrain_lock(owner_token)
            storage.add_run({
                "run_id": job_id,
                "time": storage.now_iso(),
                "status": "failed",
                "version": None,
                "old_mse": None,
                "new_mse": None,
                "train_rows": None,
                "validation_rows": None,
                "pending_rows": None,
                "new_rows_used": None,
                "reason": "Retraining worker failed. Pending rows were kept for retry.",
            })
            print(f"[retrain] failed: {exc}", flush=True)
    threading.Thread(target=worker, daemon=True).start()


def _enqueue_retraining_task(force: bool, job_id: str, owner_token: str | None = None) -> dict:
    """Queue the GA retraining worker through Cloud Tasks when configured.

    Required env vars for Cloud Tasks mode:
      CLOUD_TASKS_QUEUE, CLOUD_TASKS_LOCATION, SERVICE_URL
    Optional:
      GOOGLE_CLOUD_PROJECT / GCP_PROJECT / PROJECT_ID, CLOUD_TASKS_SERVICE_ACCOUNT
    """
    queue = os.environ.get("CLOUD_TASKS_QUEUE", "").strip()
    location = os.environ.get("CLOUD_TASKS_LOCATION", os.environ.get("REGION", "")).strip()
    project = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or os.environ.get("PROJECT_ID")
        or ""
    ).strip()
    service_url = os.environ.get("SERVICE_URL", "").strip().rstrip("/")

    if not (tasks_v2 and queue and location and project and service_url):
        return {"queued": False, "mode": "thread", "reason": "cloud_tasks_not_configured"}

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(project, location, queue)
    url = f"{service_url}/tasks/retrain"
    headers = {"Content-Type": "application/json"}
    if ADMIN_TOKEN:
        headers["X-Admin-Token"] = ADMIN_TOKEN
    body = json.dumps({"force": force, "job_id": job_id, "owner_token": owner_token}).encode("utf-8")

    http_request = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": url,
        "headers": headers,
        "body": body,
    }

    task_sa = os.environ.get("CLOUD_TASKS_SERVICE_ACCOUNT", "").strip()
    if task_sa:
        http_request["oidc_token"] = {
            "service_account_email": task_sa,
            "audience": service_url,
        }

    task = {"http_request": http_request}
    response = client.create_task(request={"parent": parent, "task": task})
    return {"queued": True, "mode": "cloud_tasks", "task_name": response.name}


def _queue_retraining(force: bool, message: str, run_id: str | None = None,
                      owner_token: str | None = None) -> dict:
    """Persist the running status (for the UI) then dispatch the worker.

    Phase 4: the atomic lease lock is acquired by the caller
    (maybe_trigger_retraining) BEFORE this runs, so concurrency is already
    guaranteed; this function only records UI status and dispatches. The
    owner_token travels to the worker so it can heartbeat and verify ownership.
    """
    job_id = run_id or uuid.uuid4().hex
    storage.set_status("running", message, extra={"job_id": job_id, "force": force})
    try:
        queued = _enqueue_retraining_task(force=force, job_id=job_id, owner_token=owner_token)
    except Exception as exc:
        print(f"[tasks] enqueue failed, falling back to thread: {exc}", flush=True)
        queued = {"queued": False, "mode": "thread", "reason": "enqueue_failed"}

    if not queued.get("queued"):
        _run_retraining_thread(force=force, job_id=job_id, owner_token=owner_token)
    return {"job_id": job_id, **queued}


def maybe_trigger_retraining(source: str, force: bool = False) -> dict:
    """Phase 3+4 — single entry for ALL retrain triggers (upload / cron / manual).

    One shared volume policy (cron cannot bypass it) + an ATOMIC lease acquire as
    the authoritative concurrency gate. `force` bypasses the volume threshold but
    never the lock.
    """
    # Before the status-based trigger gate runs, heal any old Running state left
    # behind by an interrupted task. Pending rows are kept for retry.
    if hasattr(storage, "recover_stale_retrain_status"):
        storage.recover_stale_retrain_status()
    decision = retraining.trigger_decision(force=force)
    # Volume gate (force already overrides this inside trigger_decision).
    if decision["decision"] == "skipped_not_enough_data":
        return {"ok": True, "triggered": False, "status": "skipped_not_enough_data",
                "source": source, **decision}

    # Authoritative concurrency gate: atomic compare-and-set lease acquire.
    run_id = uuid.uuid4().hex
    owner_token = uuid.uuid4().hex
    lock = storage.acquire_retrain_lock(run_id, owner_token, source)
    if lock is None:
        return {"ok": True, "triggered": False, "status": "skipped_already_running",
                "source": source, **decision}

    n = decision["freshness"]["new_rows"]
    msg = (f"Force retraining queued ({source})…" if force
           else f"Auto-retraining queued ({source}, {n} clean rows ≥ "
                f"{decision['freshness']['threshold']})…")
    try:
        queue_info = _queue_retraining(force=force, message=msg, run_id=run_id,
                                       owner_token=owner_token)
    except Exception:
        storage.release_retrain_lock(owner_token)  # never strand the lease
        raise
    return {"ok": True, "triggered": True, "status": "triggered", "source": source,
            "run_id": run_id, "queue": queue_info, **decision}


def maybe_chain_retraining_after_run(result: dict, source: str = "chain") -> dict:
    """Immediately start the next retrain if rows arrived while the previous
    GA run was locked.

    Batch isolation keeps new uploads in Queued Clean Rows. This follow-up check
    prevents the system from waiting until the next hourly scheduler tick when
    the queued rows already satisfy the threshold.
    """
    if not isinstance(result, dict) or result.get("action") not in ("promote", "skip"):
        return {"ok": True, "triggered": False, "status": "not_applicable"}

    fresh = retraining.check_data_freshness()
    if not fresh.get("ready"):
        return {"ok": True, "triggered": False, "status": "not_enough_queued_rows", "freshness": fresh}

    # The previous job has returned, so its lease should already be released by
    # _run_retraining_job(). Use the same shared trigger path so the lock still
    # prevents duplicate follow-up jobs.
    return maybe_trigger_retraining(source=source, force=False)


import fuzzy_engine as fe

# Activate the serving model at boot.
# Solution A: if no model has been stored yet, build a fair v0 baseline trained
# on the holdout-excluded data (so it never saw the validation set), then serve
# it. On later boots the stored active model (baseline or a promoted artifact)
# is simply re-applied.
def _activate_model_from_artifact_at_boot():
    try:
        created = retraining.build_baseline_if_missing(apply_live=True)
        if created:
            print(f"[model] Built fair v0 baseline on holdout-excluded data "
                  f"(MSE {created.get('new_mse')}) — now serving.", flush=True)
            return
        # Self-heal a stale/old active record that lacks a usable MSE-based GA
        # block (e.g. an older deploy's artifact, or a synthesized v0 rollback).
        # Without this the GA page falls back to the boot full-data engine and
        # shows the misleading 45,000-row / large-error numbers forever.
        try:
            retraining.repair_active_ga(allow_rebuild=True)
        except Exception as _heal_exc:
            print(f"[model] ga self-heal skipped: {_heal_exc}", flush=True)
        active_params, active_version = retraining.active_params_or_default()
        if hasattr(fe, "apply_model_params"):
            fe.apply_model_params(active_params, source=f"active model v{active_version}")
            label = "v0 baseline" if active_version == 0 else f"promoted artifact v{active_version}"
            print(f"[model] Active model {label} is now serving live decisions.", flush=True)
        else:
            print(f"[model] Active model v{active_version} found, but apply_model_params() is missing.", flush=True)
    except Exception as _exc:  # never block startup
        print(f"[model] activation skipped: {_exc}", flush=True)

_activate_model_from_artifact_at_boot()

app = Flask(__name__)

ACCENT = "#137A54"
GREEN = "#15935F"
AMBER = "#C98A1E"
RED = "#D14343"
CARD = "#FFFFFF"
BORDER = "#E3E8EE"
MUTED = "#67767F"
INK = "#16242B"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

# Shown at the bottom of the result page and inside the exported PDF.
HUMAN_OVERSIGHT_DISCLAIMER = (
    "This system supports loan assessment decisions but does not replace "
    "final human officer review."
)


# -----------------------------------------------------------------------------
# Core fuzzy tool wrappers
# -----------------------------------------------------------------------------
def _default_to_value(default: Any) -> int:
    return 1 if str(default).strip().lower() in {"yes", "y", "1", "true", "default"} else 0


def lookup_ctos_category(credit: float) -> Dict[str, Any]:
    c = float(credit)
    category = (
        "Very Poor" if c <= 449 else
        "Poor" if c <= 549 else
        "Fair" if c <= 649 else
        "Good" if c <= 749 else
        "Excellent"
    )
    return {
        "credit_score": round(c, 0),
        "category": category,
        "risk_value": round(float(fe.ctos_risk(c)), 2),
    }


def compute_risk_detailed(
    income: float,
    credit: float,
    ratio: float,
    default: Any,
    emp: float,
) -> Dict[str, Any]:
    """Call the original GA-optimised fuzzy engine for one applicant.

    A fresh ControlSystemSimulation is created for each request to avoid state
    leakage between web requests.
    """
    default_val = _default_to_value(default)

    sim = fe.ctrl.ControlSystemSimulation(fe.risk_ctrl)
    sim.input["income"] = float(income)
    sim.input["credit"] = float(credit)
    sim.input["ratio"] = float(ratio)
    sim.input["default"] = default_val
    sim.input["emp_exp"] = float(emp)
    sim.compute()

    score = float(sim.output["risk"])
    decision = "APPROVE" if score < 40 else ("REVIEW" if score < 70 else "REJECT")
    md = fe.membership_degree

    degrees = {
        "income_high": round(md(float(income), fe.best_mfs["inc_u"], fe.best_mfs["inc"]["high"]), 2),
        "income_medium": round(md(float(income), fe.best_mfs["inc_u"], fe.best_mfs["inc"]["medium"]), 2),
        "income_low": round(md(float(income), fe.best_mfs["inc_u"], fe.best_mfs["inc"]["low"]), 2),
        "ratio_high": round(md(float(ratio), fe.best_mfs["rat_u"], fe.best_mfs["rat"]["high"]), 2),
        "ratio_medium": round(md(float(ratio), fe.best_mfs["rat_u"], fe.best_mfs["rat"]["medium"]), 2),
        "ratio_low": round(md(float(ratio), fe.best_mfs["rat_u"], fe.best_mfs["rat"]["low"]), 2),
        "emp_experienced": round(md(float(emp), fe.best_mfs["emp_u"], fe.best_mfs["emp"]["experienced"]), 2),
        "emp_mid": round(md(float(emp), fe.best_mfs["emp_u"], fe.best_mfs["emp"]["mid"]), 2),
        "emp_junior": round(md(float(emp), fe.best_mfs["emp_u"], fe.best_mfs["emp"]["junior"]), 2),
        "credit_risk": round(float(fe.ctos_risk(float(credit))), 2),
        "defaulted": default_val,
    }

    return {
        "risk_score": round(score, 1),
        "decision": decision,
        "membership_degrees": degrees,
        "rule_weights": {
            "ratio": fe.W_RATIO,
            "credit": fe.W_CREDIT,
            "income": fe.W_INCOME,
            "default": fe.W_DEFAULT,
            "emp": fe.W_EMP,
        },
    }


def reasons_from_degrees(degrees: Dict[str, Any]) -> List[Dict[str, str]]:
    reasons: List[Dict[str, str]] = []

    credit_risk = float(degrees["credit_risk"])
    if credit_risk <= 0.25:
        reasons.append({"tone": "positive", "text": "Credit score is strong, so credit risk is low."})
    elif credit_risk >= 0.55:
        reasons.append({"tone": "negative", "text": "Credit score falls into a weaker band and increases the risk level."})
    else:
        reasons.append({"tone": "neutral", "text": "Credit score is moderate and requires supporting evidence from other factors."})

    if degrees["ratio_high"] >= 0.5:
        reasons.append({"tone": "negative", "text": "Loan-to-income ratio has high membership in the risky range."})
    elif degrees["ratio_low"] >= 0.5:
        reasons.append({"tone": "positive", "text": "Loan-to-income ratio is comfortably low."})
    else:
        reasons.append({"tone": "neutral", "text": "Loan-to-income ratio is in the middle range."})

    if degrees["emp_junior"] >= 0.5:
        reasons.append({"tone": "negative", "text": "Employment experience is still junior, which raises uncertainty."})
    elif degrees["emp_experienced"] >= 0.5:
        reasons.append({"tone": "positive", "text": "Employment experience is strong and improves repayment confidence."})
    else:
        reasons.append({"tone": "neutral", "text": "Employment experience is mid-level."})

    if int(degrees["defaulted"]) == 1:
        reasons.append({"tone": "negative", "text": "Previous loan default is present, so the risk score increases."})
    else:
        reasons.append({"tone": "positive", "text": "No previous loan default is recorded."})

    return reasons


def build_what_if(income: float, credit: float, ratio: float, default: Any, emp: float, current: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run what-if tool calls and keep only actions that reduce risk.

    The dashboard should not show confusing options where the risk score rises.
    """
    if current["decision"] == "APPROVE":
        return []

    default_text = "Yes" if _default_to_value(default) else "No"
    current_score = float(current["risk_score"])
    candidates = []

    # Increase annual income (~RM500/month). Keeps the same requested loan amount,
    # so the debt-to-income ratio falls as income rises.
    income_step = 6000.0
    new_income = float(income) + income_step
    loan_amount = float(income) * float(ratio)
    new_ratio_income = round(min(1.0, loan_amount / new_income), 4) if new_income > 0 else float(ratio)
    candidates.append((
        "Increase monthly income",
        {"income": new_income, "credit": credit, "ratio": new_ratio_income, "default": default_text, "emp": emp},
        f"Increasing annual income by about RM{income_step:,.0f} (~RM500/month)",
    ))

    # Lower loan ratio / requested amount
    new_ratio = max(0.05, round(float(ratio) * 0.80, 3))
    new_amount = float(income) * new_ratio
    candidates.append((
        "Reduce requested loan amount",
        {"income": income, "credit": credit, "ratio": new_ratio, "default": default_text, "emp": emp},
        f"Reducing the requested amount to about RM{new_amount:,.0f} (DTI {float(ratio):.2f} → {new_ratio:.2f})",
    ))

    # Improve CTOS score
    if float(credit) < 800:
        new_credit = min(850, int(float(credit) + 50))
        candidates.append((
            "Improve CTOS score",
            {"income": income, "credit": new_credit, "ratio": ratio, "default": default_text, "emp": emp},
            f"Improving CTOS from {int(float(credit))} to around {new_credit} before reapplying",
        ))

    # Build employment stability
    if float(emp) < 8:
        new_emp = round(float(emp) + 1.5, 1)
        candidates.append((
            "Build employment stability",
            {"income": income, "credit": credit, "ratio": ratio, "default": default_text, "emp": new_emp},
            f"Increasing employment experience from {float(emp):.1f} to {new_emp:.1f} years",
        ))

    results = []
    for label, payload, advice in candidates:
        try:
            # Exact recalculation through the live fuzzy/risk engine — the new
            # score below is computed, not estimated.
            r = compute_risk_detailed(**payload)
            new_score = float(r["risk_score"])
            improvement = round(current_score - new_score, 1)
            if improvement > 0:
                results.append({
                    "label": label,
                    "advice": f"{advice} may reduce the risk score from {current_score:.0f} to {new_score:.0f}.",
                    "new_score": r["risk_score"],
                    "new_decision": r["decision"],
                    "delta": improvement,
                    "from_score": round(current_score, 1),
                    "to_score": round(new_score, 1),
                    "estimated": False,
                    "delta_text": f"Risk score {current_score:.0f} → {new_score:.0f}",
                })
        except Exception:
            continue

    return sorted(results, key=lambda item: item["delta"], reverse=True)[:3]


def build_why_factors(applicant: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Top contributing factors behind the decision, in plain language.

    Drives the result page 'Why this decision?' section. Derived from the same
    fuzzy membership degrees used for scoring, so it never invents reasons.
    """
    d = result.get("membership_degrees", {}) or {}
    factors: List[Dict[str, str]] = []

    credit_risk = float(d.get("credit_risk", 0))
    if credit_risk >= 0.55:
        factors.append({"tone": "negative", "text": "High CTOS credit risk"})
    elif credit_risk <= 0.25:
        factors.append({"tone": "positive", "text": "Strong CTOS credit score"})

    if float(d.get("ratio_high", 0)) >= 0.5:
        factors.append({"tone": "negative", "text": "High loan amount relative to income"})
    elif float(d.get("ratio_low", 0)) >= 0.5:
        factors.append({"tone": "positive", "text": "Healthy loan-to-income ratio"})

    if float(d.get("income_low", 0)) >= 0.5:
        factors.append({"tone": "negative", "text": "Low income-to-loan ratio (weak repayment capacity)"})
    elif float(d.get("income_high", 0)) >= 0.5:
        factors.append({"tone": "positive", "text": "Strong income capacity"})

    if float(d.get("emp_junior", 0)) >= 0.5:
        factors.append({"tone": "negative", "text": "Unstable employment status (junior experience)"})
    elif float(d.get("emp_experienced", 0)) >= 0.5:
        factors.append({"tone": "positive", "text": "Stable employment history"})

    if int(d.get("defaulted", 0)) == 1:
        factors.append({"tone": "negative", "text": "Previous loan default on record"})

    if not factors:
        factors.append({"tone": "neutral", "text": "Balanced profile across all five factors"})
    return factors[:5]

def local_agent_summary(applicant: Dict[str, Any], ctos: Dict[str, Any], result: Dict[str, Any], what_if: List[Dict[str, Any]]) -> str:
    decision = result["decision"]
    score = result["risk_score"]
    d = result["membership_degrees"]
    text = (
        f"The workflow verified all five required variables and collected CTOS evidence: "
        f"{int(float(applicant['credit']))} is classified as {ctos['category']}. "
        f"The fuzzy engine was then called as the scoring tool. The final decision is {decision} "
        f"with a risk score of {score}/100. The explanation is based on membership degrees: "
        f"income_high={d['income_high']}, ratio_high={d['ratio_high']}, "
        f"emp_junior={d['emp_junior']}, and credit_risk={d['credit_risk']}."
    )
    if what_if:
        best = what_if[0]
        text += f" Recommended action: {best['advice']} This may reduce the score by about {best['delta']} points."
    elif decision == "APPROVE":
        text += " The applicant profile is acceptable. Maintain repayment discipline and credit quality."
    return text


def build_agent_trace(applicant: Dict[str, Any], ctos: Dict[str, Any], result: Dict[str, Any], what_if: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "step": "Observe",
            "detail": "Extracted income, CTOS score, loan ratio, previous default, and employment experience.",
        },
        {
            "step": "Collect",
            "detail": f"Mapped CTOS {int(float(applicant['credit']))} to {ctos['category']} credit evidence.",
        },
        {
            "step": "Reason",
            "detail": "All required variables are complete, so the agent can call the fuzzy risk tool.",
        },
        {
            "step": "Call Engine",
            "detail": f"calculate_loan_risk returned {result['risk_score']}/100 and {result['decision']}.",
        },
        {
            "step": "Explain",
            "detail": "Explanation uses real fuzzy membership degrees instead of invented reasons.",
        },
        {
            "step": "Act",
            "detail": "Returned final decision and improvement actions." if what_if else "Returned final decision and approval guidance.",
        },
    ]


# -----------------------------------------------------------------------------
# Optional Gemini tool-calling agent
# -----------------------------------------------------------------------------
def _gemini_tools():
    if types is None:
        return None
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="calculate_loan_risk",
            description="Run the GA-optimised fuzzy logic engine. Returns risk score, decision and membership degrees.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={
                "income": types.Schema(type=types.Type.NUMBER, description="Annual income in RM"),
                "credit": types.Schema(type=types.Type.NUMBER, description="CTOS score from 300 to 850"),
                "ratio": types.Schema(type=types.Type.NUMBER, description="Loan fraction of income from 0 to 1"),
                "default": types.Schema(type=types.Type.STRING, description="Yes or No"),
                "emp": types.Schema(type=types.Type.NUMBER, description="Employment experience in years"),
            }, required=["income", "credit", "ratio", "default", "emp"]),
        ),
        types.FunctionDeclaration(
            name="lookup_ctos_category",
            description="Look up the CTOS credit category for a credit score.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={
                "credit": types.Schema(type=types.Type.NUMBER, description="CTOS credit score")},
                required=["credit"]),
        ),
    ])


def execute_agent_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "calculate_loan_risk":
        return compute_risk_detailed(
            args["income"], args["credit"], args["ratio"], args["default"], args["emp"]
        )
    if name == "lookup_ctos_category":
        return lookup_ctos_category(args["credit"])
    return {"error": f"Unknown tool: {name}"}


def _gemini_thinking_config():
    """Minimise Gemini 'thinking' to cut latency. Gemini 3.x uses
    thinking_level ('minimal'|'low'|'medium'|'high'); older 2.x SDKs use
    thinking_budget. Try the new field first, fall back to the budget form,
    and return None if neither is supported (simply omitted)."""
    if types is None:
        return None
    try:
        return types.ThinkingConfig(thinking_level="minimal")
    except Exception:
        pass
    try:
        return types.ThinkingConfig(thinking_budget=0)
    except Exception:
        return None


SYSTEM_PROMPT = """You are a Malaysian loan officer AI agent. Follow this loop:
Observe the five variables, collect CTOS evidence, reason whether data is complete,
call the fuzzy engine tool, then explain the tool output.

The final explanation must be concise and non-repetitive:
- Do not repeat a full assessment summary; the UI already shows decision, score, and evidence.
- Write as an AI officer explanation in 90-130 words.
- Mention the main trade-off between positive factors and risk factors.
- Use clear English, no raw JSON, no HTML, and avoid excessive membership-degree numbers.
- Do not invent model reasons; only use tool outputs.
"""


# -----------------------------------------------------------------------------
# Plotly helpers
# -----------------------------------------------------------------------------
def _base_layout(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"family": FONT, "color": MUTED, "size": 12},
        margin={"l": 12, "r": 12, "t": 42, "b": 18},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": MUTED, "size": 11}},
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": ACCENT,
                    "font": {"color": INK, "size": 13, "family": FONT}},
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="#EEF1F4", zerolinecolor="#E3E8EE", linecolor="#E3E8EE")
    fig.update_yaxes(gridcolor="#EEF1F4", zerolinecolor="#E3E8EE", linecolor="#E3E8EE")
    return fig


def convergence_figure(fit=None, metric="training fitness") -> go.Figure:
    if fit is None:
        fit = fe.ga_result["fit_history"]
    x = list(range(len(fit)))
    # MSE values are small (~0.18); engine fitness is large (~10000). Use an
    # adaptive hover precision so both read cleanly.
    yfmt = ".5f" if (fit and max(fit) < 10) else ".1f"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=fit,
        mode="lines",
        name="Best fitness",
        line={"color": ACCENT, "width": 3},
        fill="tozeroy",
        fillcolor="rgba(19,122,84,.08)",
        hovertemplate=f"<b>Gen %{{x}}</b><br>{metric} %{{y:{yfmt}}}<extra></extra>",
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": ACCENT,
                    "font": {"color": INK, "size": 13, "family": FONT}},
    ))
    best_idx = int(min(range(len(fit)), key=lambda i: fit[i]))
    fig.add_vline(x=best_idx, line_dash="dot", line_color=ACCENT, opacity=0.65)
    fig.add_annotation(
        x=best_idx,
        y=fit[best_idx],
        text="<b>Model converged here</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor=ACCENT,
        ax=35,
        ay=-42,
        font={"color": INK, "size": 13, "family": FONT},
        bgcolor="#FFFFFF",
        bordercolor=ACCENT,
        borderwidth=1.2,
        borderpad=6,
    )
    fig.update_layout(title={"text": f"GA convergence — lower {metric} is better", "font": {"color": INK, "size": 15}})
    return _base_layout(fig, 330)


def membership_figure(view: str = "all") -> go.Figure:
    """Build membership-function charts.

    view="all" shows the complete fuzzy system.
    view="ga" shows only GA-tuned input variables.
    view="fixed" shows only standard/fixed variables.
    This makes the Membership page cards act like real filters instead of static labels.
    """
    all_specs: List[Tuple[str, Any, Dict[str, Any], str, List[str]]] = [
        (
            "ga",
            fe.inc_u,
            {"Low": fe.income_fuzzy["low"].mf, "Medium": fe.income_fuzzy["medium"].mf, "High": fe.income_fuzzy["high"].mf},
            "Annual Income · GA-tuned",
            ["#C65146", "#B7791F", "#0F766E"],
        ),
        (
            "fixed",
            fe.credit_fuzzy.universe,
            {
                "Very Poor": fe.credit_fuzzy["very_poor"].mf,
                "Poor": fe.credit_fuzzy["poor"].mf,
                "Fair": fe.credit_fuzzy["fair"].mf,
                "Good": fe.credit_fuzzy["good"].mf,
                "Excellent": fe.credit_fuzzy["excellent"].mf,
            },
            "CTOS Credit Score · fixed standard",
            ["#B91C1C", "#EA580C", "#CA8A04", "#15803D", "#0E7490"],
        ),
        (
            "ga",
            fe.rat_u,
            {"Low": fe.ratio_fuzzy["low"].mf, "Medium": fe.ratio_fuzzy["medium"].mf, "High": fe.ratio_fuzzy["high"].mf},
            "Loan-to-Income Ratio · GA-tuned",
            ["#0F766E", "#B7791F", "#C65146"],
        ),
        (
            "fixed",
            fe.default_fuzzy.universe,
            {"No Default": fe.default_fuzzy["no"].mf, "Defaulted": fe.default_fuzzy["yes"].mf},
            "Previous Loan Default · fixed binary",
            ["#0F766E", "#C65146"],
        ),
        (
            "ga",
            fe.emp_u,
            {"Junior": fe.emp_fuzzy["junior"].mf, "Mid-Level": fe.emp_fuzzy["mid"].mf, "Experienced": fe.emp_fuzzy["experienced"].mf},
            "Employment Experience · GA-tuned",
            ["#C65146", "#B7791F", "#0F766E"],
        ),
        (
            "output",
            fe.risk.universe,
            {"Low Risk": fe.risk["low"].mf, "Medium Risk": fe.risk["medium"].mf, "High Risk": fe.risk["high"].mf},
            "Output · Risk Score",
            ["#0F766E", "#B7791F", "#C65146"],
        ),
    ]

    view = (view or "all").lower()
    if view == "ga":
        specs = [item for item in all_specs if item[0] == "ga"]
        title = "GA-tuned membership functions"
    elif view == "fixed":
        # "Fixed + Output" tab: CTOS, Previous Loan Default AND the Output Risk Score.
        specs = [item for item in all_specs if item[0] in ("fixed", "output")]
        title = "Fixed standard membership functions"
    else:
        specs = all_specs
        title = "All membership functions"

    cols = 2
    rows = int((len(specs) + cols - 1) / cols)
    # Keep EVERY subplot the same physical size across all three views by using a
    # fixed per-row height and converting the gaps from px to Plotly's fractional
    # spacing (which is relative to total height).
    PER_ROW_PX = 300
    TOP_PAD_PX = 70
    V_GAP_PX = 58
    height = rows * PER_ROW_PX + TOP_PAD_PX
    vertical_spacing = min(0.45, V_GAP_PX / height)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[item[3] for item in specs],
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.08,
    )
    for idx, (_kind, universe, mfs, _title, colors) in enumerate(specs):
        row = idx // cols + 1
        col = idx % cols + 1

        # Binary "Previous Loan Default" has only a 2-point universe {0,1}; plotting
        # trimf[0,0,0] / trimf[1,1,1] across it draws two misleading diagonal lines
        # (implying fractional membership at x=0.5). Render it as clean binary steps
        # instead — display only, the engine math is unchanged.
        if str(_title).startswith("Previous Loan Default"):
            xx = np.linspace(0.0, 1.0, 240)
            mfs = {
                "No Default": np.where(xx < 0.5, 1.0, 0.0),
                "Defaulted": np.where(xx >= 0.5, 1.0, 0.0),
            }
            universe = xx

        for (label, values), color in zip(mfs.items(), colors):
            def _rgba(hexc, a):
                hexc = hexc.lstrip("#")
                r, g, b = (int(hexc[i:i+2], 16) for i in (0, 2, 4))
                return f"rgba({r},{g},{b},{a})"
            fig.add_trace(go.Scatter(
                x=list(universe),
                y=list(values),
                mode="lines",
                name=label,
                line={"color": color, "width": 2.2},
                fill="tozeroy",
                fillcolor=_rgba(color, 0.12),
                legendgroup=f"g{idx}",
                hovertemplate=f"<b>{label}</b><br>x %{{x:.2f}}<br>μ %{{y:.2f}}<extra></extra>",
                hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": color,
                            "font": {"color": INK, "size": 12, "family": FONT}},
            ), row=row, col=col)

        # GA-tuned variables span their full raw universe (income.max()≈270k,
        # emp.max()≈26yrs), so when the optimiser clusters the breakpoints near
        # the low end the whole transition collapses into a thin sliver. Frame
        # each GA-tuned subplot tightly around its own breakpoints (+margin) so
        # the sets read as proper shapes regardless of where they landed.
        try:
            if str(_title).startswith("Annual Income"):
                xmax = min(float(fe.INC_MAX), max(float(fe.inc_c) * 2.2, 60000.0))
                fig.update_xaxes(range=[0, xmax], row=row, col=col)
            elif str(_title).startswith("Loan-to-Income Ratio"):
                xmax = min(float(fe.RAT_MAX), max(float(fe.rat_c) * 2.5, 0.5))
                fig.update_xaxes(range=[0, xmax], row=row, col=col)
            elif str(_title).startswith("Employment Experience"):
                xmax = min(float(fe.EMP_MAX), max(float(fe.emp_c) * 3.0, 1.5))
                fig.update_xaxes(range=[0, xmax], row=row, col=col)
        except Exception:
            pass

    # height is computed above from the row count so subplots stay uniform.
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        font={"family": FONT, "color": "#5D6975", "size": 11},
        margin={"l": 10, "r": 10, "t": 48, "b": 28},
        title={"text": title, "font": {"size": 14, "color": "#0F172A"}, "x": 0.01, "xanchor": "left"},
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#EDF1F5", linecolor="#D9E1E8", zerolinecolor="#D9E1E8")
    fig.update_yaxes(gridcolor="#EDF1F5", linecolor="#D9E1E8", zerolinecolor="#D9E1E8", range=[-0.05, 1.08])
    for annotation in fig.layout.annotations:
        annotation.font = {"color": "#0F172A", "size": 13, "family": FONT}
    return fig



def membership_card_figures(view: str = "all") -> List[Dict[str, Any]]:
    """Return one independent Plotly figure per membership variable.

    This is used by the Admin UI card grid. It avoids the responsive sizing and
    clipping problems caused by packing six subplots into one large figure.
    """
    specs: List[Tuple[str, Any, Dict[str, Any], str, List[str], str]] = [
        ("ga", fe.inc_u, {"Low": fe.income_fuzzy["low"].mf, "Medium": fe.income_fuzzy["medium"].mf, "High": fe.income_fuzzy["high"].mf}, "Annual Income", ["#C65146", "#B7791F", "#0F766E"], "GA-Tuned"),
        ("fixed", fe.credit_fuzzy.universe, {
            "Very Poor": fe.credit_fuzzy["very_poor"].mf,
            "Poor": fe.credit_fuzzy["poor"].mf,
            "Fair": fe.credit_fuzzy["fair"].mf,
            "Good": fe.credit_fuzzy["good"].mf,
            "Excellent": fe.credit_fuzzy["excellent"].mf,
        }, "CTOS Credit Score", ["#B91C1C", "#EA580C", "#CA8A04", "#15803D", "#0E7490"], "Fixed"),
        ("ga", fe.rat_u, {"Low": fe.ratio_fuzzy["low"].mf, "Medium": fe.ratio_fuzzy["medium"].mf, "High": fe.ratio_fuzzy["high"].mf}, "Loan-to-Income Ratio", ["#0F766E", "#B7791F", "#C65146"], "GA-Tuned"),
        ("fixed", fe.default_fuzzy.universe, {"No Default": fe.default_fuzzy["no"].mf, "Defaulted": fe.default_fuzzy["yes"].mf}, "Previous Loan Default", ["#0F766E", "#C65146"], "Fixed"),
        ("ga", fe.emp_u, {"Junior": fe.emp_fuzzy["junior"].mf, "Mid-Level": fe.emp_fuzzy["mid"].mf, "Experienced": fe.emp_fuzzy["experienced"].mf}, "Employment Experience", ["#C65146", "#B7791F", "#0F766E"], "GA-Tuned"),
        ("output", fe.risk.universe, {"Low Risk": fe.risk["low"].mf, "Medium Risk": fe.risk["medium"].mf, "High Risk": fe.risk["high"].mf}, "Output Risk Score", ["#0F766E", "#B7791F", "#C65146"], "Output"),
    ]
    view = (view or "all").lower()
    if view == "ga":
        specs = [item for item in specs if item[0] == "ga"]
    elif view == "fixed":
        specs = [item for item in specs if item[0] in ("fixed", "output")]

    def _rgba(hexc: str, a: float) -> str:
        hexc = hexc.lstrip("#")
        r, g, b = (int(hexc[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{a})"

    cards: List[Dict[str, Any]] = []
    for kind, universe, mfs, title, colors, tag in specs:
        # Render binary default as a clear step chart rather than misleading
        # diagonal trimf lines between x=0 and x=1.
        if title == "Previous Loan Default":
            xx = np.linspace(0.0, 1.0, 240)
            universe = xx
            mfs = {
                "No Default": np.where(xx < 0.5, 1.0, 0.0),
                "Defaulted": np.where(xx >= 0.5, 1.0, 0.0),
            }

        fig = go.Figure()
        for (label, values), color in zip(mfs.items(), colors):
            fig.add_trace(go.Scatter(
                x=list(universe),
                y=list(values),
                mode="lines",
                name=label,
                line={"color": color, "width": 2.25},
                fill="tozeroy",
                fillcolor=_rgba(color, 0.12),
                hovertemplate=f"<b>{label}</b><br>x %{{x:.2f}}<br>μ %{{y:.2f}}<extra></extra>",
                hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": color,
                            "font": {"color": INK, "size": 12, "family": FONT}},
            ))

        # Tight, readable x-ranges for GA-tuned variables.
        try:
            if title == "Annual Income":
                xmax = min(float(fe.INC_MAX), max(float(fe.inc_c) * 2.2, 60000.0))
                fig.update_xaxes(range=[0, xmax])
            elif title == "Loan-to-Income Ratio":
                xmax = min(float(fe.RAT_MAX), max(float(fe.rat_c) * 2.5, 0.5))
                fig.update_xaxes(range=[0, xmax])
            elif title == "Employment Experience":
                xmax = min(float(fe.EMP_MAX), max(float(fe.emp_c) * 3.0, 1.5))
                fig.update_xaxes(range=[0, xmax])
        except Exception:
            pass

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=285,
            margin={"l": 44, "r": 16, "t": 10, "b": 38},
            showlegend=True,
            legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 10, "color": "#64748b"}},
            font={"family": FONT, "color": "#5D6975", "size": 11},
        )
        fig.update_xaxes(gridcolor="#EDF1F5", linecolor="#D9E1E8", zerolinecolor="#D9E1E8")
        fig.update_yaxes(gridcolor="#EDF1F5", linecolor="#D9E1E8", zerolinecolor="#D9E1E8", range=[-0.05, 1.08])
        cards.append({
            "kind": kind,
            "tag": tag,
            "title": title,
            "subtitle": "GA-tuned membership breakpoints" if tag == "GA-Tuned" else ("Fixed standard mapping" if tag == "Fixed" else "Output risk score mapping"),
            "figure": json.loads(fig.to_json()),
        })
    return cards

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    # AI is considered available if AT LEAST ONE of the three configured
    # providers has a key: Gemini, Cerebras, or Groq.
    gemini_ready = bool(os.environ.get("GEMINI_API_KEY"))
    cerebras_ready = bool(os.environ.get("CEREBRAS_API_KEY"))
    groq_ready = bool(os.environ.get("GROQ_API_KEY"))
    ai_ready = gemini_ready or cerebras_ready or groq_ready
    llm_status = "Powered by AI" if ai_ready else "AI key required"
    version_info = _active_version_payload()
    return render_template(
        "index.html",
        gemini_configured=gemini_ready,
        cerebras_configured=cerebras_ready,
        groq_configured=groq_ready,
        ai_configured=ai_ready,
        llm_status=llm_status,
        **version_info,
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def _active_version_payload() -> Dict[str, Any]:
    try:
        _active_params, active_version = retraining.active_params_or_default()
        active_version = int(active_version or 0)
    except Exception:
        active_version = 0
    return {"active_version": active_version, "active_version_label": f"v{active_version}"}


@app.get("/api/model-status")
def api_model_status():
    return jsonify({"ok": True, **_active_version_payload()})


@app.post("/api/evaluate")
def api_evaluate():
    """Agentic web evaluation path.

    The AI agent is the reasoning brain. It must call the fuzzy risk engine as a
    tool. Without a configured AI provider, the assessment is not allowed to run.
    """
    data = request.get_json(force=True) or {}
    try:
        income_value = float(data.get("income", 65000))
        loan_amount_value = data.get("loan_amount", None)
        if loan_amount_value is not None and str(loan_amount_value) != "":
            loan_amount_value = float(loan_amount_value)
            if income_value <= 0:
                return jsonify({"ok": False, "code": "VALIDATION", "error": "Annual income must be greater than 0 to calculate Debt-to-Income Ratio."}), 400
            ratio_value = loan_amount_value / income_value
        else:
            ratio_value = float(data.get("ratio", 0.18))
            loan_amount_value = income_value * ratio_value

        if ratio_value < 0 or ratio_value > 1:
            return jsonify({"ok": False, "code": "VALIDATION", "error": "Debt-to-Income Ratio must be between 0.00 and 1.00. Please reduce the requested loan amount or check the income value."}), 400

        applicant_input = {
            "income": income_value,
            "loan_amount": round(float(loan_amount_value), 2),
            "credit": float(data.get("credit", 720)),
            "ratio": round(float(ratio_value), 4),
            "default": str(data.get("default", "No")),
            "emp": float(data.get("emp", 4)),
        }

        ai_configured = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("CEREBRAS_API_KEY") or os.environ.get("GROQ_API_KEY"))
        if not ai_configured:
            return jsonify({
                "ok": False,
                "code": "AI_REQUIRED",
                "error": "AI Reasoning Key Required. This assessment uses an AI agent to call the fuzzy risk engine as a tool. Please configure GEMINI_API_KEY, CEREBRAS_API_KEY, or GROQ_API_KEY before evaluating.",
            }), 503

        agent_message = (
            "Assess this applicant using the agentic tool-calling workflow. "
            "You must call calculate_loan_risk before giving the final verdict. "
            "After the tool call, give one concise AI officer explanation only; do not repeat the UI summary. "
            f"income={applicant_input['income']}, credit={applicant_input['credit']}, "
            f"requested_loan_amount=RM{applicant_input.get('loan_amount', 0)}, "
            f"debt_to_income_ratio={applicant_input['ratio']}, previous_default={applicant_input['default']}, "
            f"employment_years={applicant_input['emp']}"
        )
        agent_reply = run_agent_chain(
            [{"role": "user", "text": agent_message}],
            system_prompt=SYSTEM_PROMPT,
        )

        if not agent_reply or agent_reply.get("error"):
            if agent_reply and agent_reply.get("error"):
                print(f"AI_PROVIDER_UNAVAILABLE: {agent_reply.get('error')}", flush=True)
            return jsonify({
                "ok": False,
                "code": "AI_UNAVAILABLE",
                "error": "AI service unavailable. Please configure a valid Gemini, Cerebras, or Groq key, or try again later.",
            }), 503

        risk_event = _tg_latest_risk_event(agent_reply)
        if not risk_event:
            return jsonify({
                "ok": False,
                "code": "TOOL_CALL_REQUIRED",
                "error": "AI agent did not call the fuzzy risk engine tool. Please try again.",
            }), 502

        tool_args = risk_event.get("args") or {}
        result = risk_event.get("result") or {}
        applicant = {
            "income": float(tool_args.get("income", applicant_input["income"])),
            "loan_amount": applicant_input["loan_amount"],
            "credit": float(tool_args.get("credit", applicant_input["credit"])),
            "ratio": round(float(tool_args.get("ratio", applicant_input["ratio"])), 4),
            "default": str(tool_args.get("default", applicant_input["default"])),
            "emp": float(tool_args.get("emp", applicant_input["emp"])),
        }
        ctos = lookup_ctos_category(applicant["credit"])
        reasons = reasons_from_degrees(result.get("membership_degrees", {}))
        why_factors = build_why_factors(applicant, result)
        what_if = build_what_if(
            applicant["income"],
            applicant["credit"],
            applicant["ratio"],
            applicant["default"],
            applicant["emp"],
            current=result,
        )
        trace = build_agent_trace(applicant, ctos, result, what_if)

        return jsonify({
            "ok": True,
            "applicant": applicant,
            "ctos": ctos,
            "result": result,
            "reasons": reasons,
            "why_factors": why_factors,
            "what_if": what_if,
            "agent_trace": trace,
            "agent_summary": agent_reply.get("text"),
            "ai_provider": agent_reply.get("provider"),
            "ai_error": None,
            "agent_events": agent_reply.get("events") or [],
            "disclaimer": HUMAN_OVERSIGHT_DISCLAIMER,
        })
    except Exception:
        print("API_EVALUATE_ERROR:", traceback.format_exc(), flush=True)
        return jsonify({"ok": False, "error": "Analysis temporarily unavailable. Please try again."}), 400


def _generate_assessment_pdf(data: Dict[str, Any]) -> Optional[bytes]:
    """Build a more professional assessment PDF.

    The web UI already groups the result into decision, evidence, drivers,
    what-if options, and AI explanation. This report mirrors that structure
    instead of dumping duplicated web sections.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.lib.utils import simpleSplit
    except Exception:
        return None

    applicant = data.get("applicant", {}) or {}
    ctos = data.get("ctos", {}) or {}
    result = data.get("result", {}) or {}
    reasons = data.get("reasons", []) or []
    why_factors = data.get("why_factors", []) or []
    what_if = data.get("what_if", []) or []
    ai_provider = data.get("ai_provider") or "Unavailable"
    ai_summary = str(data.get("agent_summary") or "").strip()
    disclaimer = data.get("disclaimer") or HUMAN_OVERSIGHT_DISCLAIMER

    decision = str(result.get("decision", "—"))
    score = float(result.get("risk_score", 0) or 0)
    risk_band = "Low" if score < 40 else ("Medium" if score < 70 else "High")
    recommendation = (
        "Approve loan" if decision == "APPROVE"
        else "Manual review required" if decision == "REVIEW"
        else "Reject / restructure"
    )

    def strip_md(value: str) -> str:
        return (
            str(value or "")
            .replace("**", "")
            .replace("* ", "")
            .replace("`", "")
            .strip()
        )

    def split_drivers():
        seen = set()
        pos, risk = [], []
        for item in list(why_factors) + list(reasons):
            text = strip_md(item.get("text", "") if isinstance(item, dict) else str(item))
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            tone = (item.get("tone", "neutral") if isinstance(item, dict) else "neutral")
            if tone == "positive":
                pos.append(text)
            else:
                risk.append(text)
        return pos[:4], risk[:5]

    positive_drivers, risk_drivers = split_drivers()

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    left, right = 42, width - 42
    top = height - 42
    state = {"y": top}

    ink = colors.HexColor("#0f172a")
    muted = colors.HexColor("#64748b")
    line = colors.HexColor("#dbe4f0")
    soft = colors.HexColor("#f8fafc")
    accent = colors.HexColor("#0f766e")
    green = colors.HexColor("#0f8a5f")
    amber = colors.HexColor("#b7791f")
    red = colors.HexColor("#c2413d")
    tone_col = green if decision == "APPROVE" else amber if decision == "REVIEW" else red

    def ensure(space=80):
        if state["y"] < space:
            footer()
            c.showPage()
            state["y"] = top

    def footer():
        c.setStrokeColor(line)
        c.line(left, 34, right, 34)
        c.setFillColor(muted)
        c.setFont("Helvetica", 8)
        c.drawString(left, 22, "Human officer review remains required for final approval decisions.")
        c.drawRightString(right, 22, "Malaysian Loan Approval System")

    def text_block(text, x, y, w, font="Helvetica", size=9, leading=12, color=ink, max_lines=None):
        c.setFont(font, size)
        c.setFillColor(color)
        lines = []
        for para in strip_md(text).splitlines():
            para = para.strip()
            if not para:
                continue
            lines.extend(simpleSplit(para, font, size, w))
        if max_lines:
            lines = lines[:max_lines]
        for ln in lines:
            c.drawString(x, y, ln)
            y -= leading
        return y

    def section(title):
        ensure(70)
        state["y"] -= 10
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(accent)
        c.drawString(left, state["y"], title)
        state["y"] -= 7
        c.setStrokeColor(line)
        c.line(left, state["y"], right, state["y"])
        state["y"] -= 15

    def kv_grid(items, x, y, w, col_w=None):
        col_w = col_w or (w / 2)
        row_h = 34
        for idx, (label, value) in enumerate(items):
            cx = x + (idx % 2) * col_w
            cy = y - (idx // 2) * row_h
            c.setFillColor(colors.white)
            c.setStrokeColor(line)
            c.roundRect(cx, cy - 25, col_w - 8, 27, 7, stroke=1, fill=1)
            c.setFont("Helvetica-Bold", 6.8)
            c.setFillColor(muted)
            c.drawString(cx + 8, cy - 8, str(label).upper())
            c.setFont("Helvetica-Bold", 9.5)
            c.setFillColor(ink)
            c.drawString(cx + 8, cy - 20, str(value))
        return y - (((len(items) + 1) // 2) * row_h)

    # Header
    report_id = f"ML-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:5].upper()}"
    c.setFillColor(ink)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(left, state["y"], "Malaysian Loan Approval Assessment Report")
    c.setFont("Helvetica", 9)
    c.setFillColor(muted)
    state["y"] -= 17
    c.drawString(left, state["y"], f"Report ID: {report_id}")
    c.drawRightString(right, state["y"], f"AI Provider: {ai_provider}")
    state["y"] -= 14
    c.drawString(left, state["y"], f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    c.drawRightString(right, state["y"], f"Model Version: {storage.get_active_version_label() if hasattr(storage, 'get_active_version_label') else 'active'}")
    state["y"] -= 22

    # Decision summary card
    ensure(130)
    box_h = 86
    c.setFillColor(colors.HexColor("#fffaf0") if decision == "REVIEW" else colors.HexColor("#f0fbf7") if decision == "APPROVE" else colors.HexColor("#fff1f0"))
    c.setStrokeColor(colors.HexColor("#f1d6a8") if decision == "REVIEW" else colors.HexColor("#bfead8") if decision == "APPROVE" else colors.HexColor("#ffd0cc"))
    c.roundRect(left, state["y"] - box_h, right - left, box_h, 12, stroke=1, fill=1)
    c.setFillColor(tone_col)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(left + 18, state["y"] - 32, decision)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left + 18, state["y"] - 55, recommendation)
    c.setFillColor(ink)
    c.setFont("Helvetica-Bold", 22)
    c.drawRightString(right - 18, state["y"] - 32, f"{score:.1f} / 100")
    c.setFillColor(muted)
    c.setFont("Helvetica", 10)
    c.drawRightString(right - 18, state["y"] - 52, f"{risk_band} Risk · Lower score is safer")
    state["y"] -= box_h + 10

    # Score visual bar
    bar_x, bar_y, bar_w, bar_h = left, state["y"] - 20, right - left, 10
    c.setFillColor(green); c.rect(bar_x, bar_y, bar_w * 0.40, bar_h, stroke=0, fill=1)
    c.setFillColor(amber); c.rect(bar_x + bar_w * 0.40, bar_y, bar_w * 0.30, bar_h, stroke=0, fill=1)
    c.setFillColor(red); c.rect(bar_x + bar_w * 0.70, bar_y, bar_w * 0.30, bar_h, stroke=0, fill=1)
    c.setFillColor(ink)
    marker_x = bar_x + bar_w * max(0, min(100, score)) / 100
    c.circle(marker_x, bar_y + bar_h / 2, 4, stroke=0, fill=1)
    c.setFont("Helvetica", 8)
    c.setFillColor(muted)
    c.drawString(bar_x, bar_y - 11, "0-39 Approve")
    c.drawCentredString(bar_x + bar_w * 0.55, bar_y - 11, "40-69 Review")
    c.drawRightString(right, bar_y - 11, "70-100 Reject")
    state["y"] -= 44

    section("Applicant Snapshot")
    loan_amount = float(applicant.get("loan_amount", 0) or 0)
    snapshot = [
        ("Annual Income", f"RM{float(applicant.get('income', 0)):,.0f}"),
        ("Requested Loan", f"RM{loan_amount:,.0f}"),
        ("Debt-to-Income", f"{float(applicant.get('ratio', 0)):.2f}"),
        ("CTOS Evidence", f"{int(float(applicant.get('credit', 0)))} ({ctos.get('category', '—')})"),
        ("Employment", f"{float(applicant.get('emp', 0)):.1f} years"),
        ("Previous Default", str(applicant.get("default", "No"))),
    ]
    state["y"] = kv_grid(snapshot, left, state["y"], right - left) - 4

    section("Decision Drivers")
    col_gap = 14
    col_w = (right - left - col_gap) / 2
    y0 = state["y"]
    c.setFillColor(colors.HexColor("#f5fbf8"))
    c.setStrokeColor(colors.HexColor("#bfead8"))
    c.roundRect(left, y0 - 98, col_w, 98, 9, stroke=1, fill=1)
    c.setFillColor(colors.HexColor("#fffaf0"))
    c.setStrokeColor(colors.HexColor("#f1d6a8"))
    c.roundRect(left + col_w + col_gap, y0 - 98, col_w, 98, 9, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(green)
    c.drawString(left + 10, y0 - 17, "Positive Factors")
    c.setFillColor(amber)
    c.drawString(left + col_w + col_gap + 10, y0 - 17, "Risk / Review Factors")
    yy = y0 - 34
    for item in (positive_drivers or ["No strong positive factor detected."])[:4]:
        c.setFillColor(green); c.circle(left + 14, yy + 3, 2.5, stroke=0, fill=1)
        yy = text_block(item, left + 23, yy, col_w - 34, size=8.6, leading=10.5, max_lines=2)
    yy = y0 - 34
    for item in (risk_drivers or ["No major risk factor detected."])[:4]:
        c.setFillColor(red if "default" in item.lower() or "high" in item.lower() else amber)
        c.circle(left + col_w + col_gap + 14, yy + 3, 2.5, stroke=0, fill=1)
        yy = text_block(item, left + col_w + col_gap + 23, yy, col_w - 34, size=8.6, leading=10.5, max_lines=2)
    state["y"] = y0 - 112

    section("What-if Improvement Options")
    if what_if:
        for item in what_if[:2]:
            label = strip_md(item.get("label", "Improvement option"))
            advice = strip_md(item.get("advice", ""))
            new_score = item.get("new_score", "—")
            ensure(38)
            c.setFillColor(colors.HexColor("#f6fbf9"))
            c.setStrokeColor(colors.HexColor("#cfe4df"))
            c.roundRect(left, state["y"] - 30, right - left, 30, 8, stroke=1, fill=1)
            c.setFillColor(ink)
            c.setFont("Helvetica-Bold", 9.5)
            c.drawString(left + 9, state["y"] - 12, label)
            c.setFillColor(muted)
            c.setFont("Helvetica", 8.5)
            c.drawString(left + 9, state["y"] - 23, advice[:100])
            c.setFillColor(accent)
            c.setFont("Helvetica-Bold", 9.5)
            c.drawRightString(right - 9, state["y"] - 17, f"New risk: {new_score}/100")
            state["y"] -= 38
    else:
        state["y"] = text_block("No major adjustment is required for this profile." if decision == "APPROVE" else "This profile needs major restructuring before approval can be considered.", left, state["y"], right-left) - 4

    if ai_summary:
        section("AI Officer Explanation")
        state["y"] = text_block(ai_summary, left, state["y"], right - left, size=9.2, leading=12.5, max_lines=12) - 4

    section("Human Oversight Note")
    state["y"] = text_block(disclaimer, left, state["y"], right - left, font="Helvetica-Oblique", size=8.8, leading=12, color=muted, max_lines=4)

    footer()
    c.showPage()
    c.save()
    return buf.getvalue()


@app.post("/api/assessment/pdf")
def api_assessment_pdf():
    """Export the current assessment result as a professional PDF.

    The frontend posts the result it already rendered (applicant, result,
    reasons, what-if, provider) so the PDF matches exactly what the user sees.
    """
    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data.get("result"), dict):
        return jsonify({"ok": False, "error": "Missing assessment data."}), 400
    try:
        pdf_bytes = _generate_assessment_pdf(data)
    except Exception:
        print("ASSESSMENT_PDF_ERROR:", traceback.format_exc(), flush=True)
        pdf_bytes = None
    if not pdf_bytes:
        return jsonify({"ok": False, "error": "PDF generation unavailable on this deployment."}), 500
    from flask import Response
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=loan_assessment.pdf"},
    )


@app.get("/api/ga")
def api_ga():
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    # Reflect the ACTIVE served model (v0 baseline or a promoted version),
    # not the original boot engine. The active record stores its own GA
    # convergence (MSE-based) and the data it was actually trained on.
    # Safety net: heal records missing a usable ga block so we never fall back
    # to the boot full-data engine (the misleading 45,000-row / error numbers).
    retraining.repair_active_ga(allow_rebuild=False)
    active = storage.get_active_params()
    active_params, active_version = retraining.active_params_or_default()
    ga = (active or {}).get("ga") if active else None

    if ga and ga.get("fit_history"):
        params = active_params
        breakpoints = [
            {"variable": "Annual Income (RM)", "a": round(float(params["inc"][0]), 2), "b": round(float(params["inc"][1]), 2), "c": round(float(params["inc"][2]), 2)},
            {"variable": "Loan Ratio (0–1)", "a": round(float(params["rat"][0]), 4), "b": round(float(params["rat"][1]), 4), "c": round(float(params["rat"][2]), 4)},
            {"variable": "Employment Exp (yrs)", "a": round(float(params["emp"][0]), 2), "b": round(float(params["emp"][1]), 2), "c": round(float(params["emp"][2]), 2)},
        ]
        return jsonify({
            "ok": True,
            "model_version": active_version,
            "stats": {
                "initial_fitness": ga["init_fitness"],
                "final_fitness": ga["final_fitness"],
                "improvement_pct": ga["improvement_pct"],
                "rows": ga.get("train_rows"),
                "approved": ga.get("approved"),
                "rejected": ga.get("rejected"),
                "generations": ga.get("generations"),
                "population": ga.get("population"),
                "metric": ga.get("metric", "MSE"),
            },
            "breakpoints": breakpoints,
            "figure": json.loads(convergence_figure(ga["fit_history"], metric="training fitness").to_json()),
        })

    # Fallback (should not happen once a baseline exists): original engine GA.
    ga = fe.ga_result
    return jsonify({
        "ok": True,
        "model_version": active_version,
        "stats": {
            "initial_fitness": round(float(ga["init_fitness"]), 2),
            "final_fitness": round(float(ga["best_fitness"]), 2),
            "improvement_pct": round(float(ga["improvement_pct"]), 1),
            "rows": int(len(fe.REF_STATUS)),
            "approved": int(fe.n_approved),
            "rejected": int(fe.n_rejected),
            "generations": int(fe.GA_GENS),
            "population": int(fe.GA_POP),
        },
        "breakpoints": [
            {"variable": "Annual Income (RM)", "a": round(float(fe.inc_a), 2), "b": round(float(fe.inc_b), 2), "c": round(float(fe.inc_c), 2)},
            {"variable": "Loan Ratio (0–1)", "a": round(float(fe.rat_a), 4), "b": round(float(fe.rat_b), 4), "c": round(float(fe.rat_c), 4)},
            {"variable": "Employment Exp (yrs)", "a": round(float(fe.emp_a), 2), "b": round(float(fe.emp_b), 2), "c": round(float(fe.emp_c), 2)},
        ],
        "figure": json.loads(convergence_figure().to_json()),
    })


@app.get("/api/membership")
def api_membership():
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    view = request.args.get("view", "all")
    return jsonify({"ok": True, "view": view, "charts": membership_card_figures(view)})


# --- Engine-aware demo samples -------------------------------------------------
# The user page's "Approve / Review / Reject sample" buttons must produce their
# labelled decision under the CURRENTLY served model. Hardcoded profiles drift:
# after a retrain the MF breakpoints move, so a fixed "review" profile can score
# into APPROVE. Instead we evaluate a small grid with the LIVE engine and pick a
# representative profile for each decision band. Cached per active model version.
_SAMPLE_CACHE: Dict[str, Any] = {"key": None, "data": None}


def _build_engine_aware_samples() -> Dict[str, Any]:
    incomes = [25000, 55000, 110000]
    credits = [500, 620, 740]
    ratios = [0.12, 0.30, 0.50, 0.72]
    emps = [1, 6]
    defaults = ["No", "Yes"]

    rows = []
    for inc in incomes:
        for cr in credits:
            for rt in ratios:
                for em in emps:
                    for df in defaults:
                        res = compute_risk_detailed(inc, cr, rt, df, em)
                        rows.append({
                            "income": inc, "credit": cr, "emp": em, "default": df,
                            "loan_amount": int(round(inc * rt)),
                            "score": float(res["risk_score"]), "decision": res["decision"],
                        })

    def fields(r):
        return {"income": r["income"], "credit": r["credit"],
                "loan_amount": r["loan_amount"], "emp": r["emp"], "default": r["default"]}

    approve_pool = [r for r in rows if r["decision"] == "APPROVE"]
    review_pool = [r for r in rows if r["decision"] == "REVIEW"]
    reject_pool = [r for r in rows if r["decision"] == "REJECT"]

    approve = min(approve_pool or rows, key=lambda r: r["score"])
    reject = max(reject_pool or rows, key=lambda r: r["score"])
    review = min(review_pool or rows, key=lambda r: abs(r["score"] - 55.0))

    return {
        "approve": fields(approve),
        "review": fields(review),
        "reject": fields(reject),
        "scores": {"approve": round(approve["score"], 1),
                   "review": round(review["score"], 1),
                   "reject": round(reject["score"], 1)},
    }


@app.get("/api/samples")
def api_samples():
    """Return Approve/Review/Reject demo profiles that land in their band under
    the model currently being served (recomputed whenever the active version
    changes), so the user-page sample buttons stay consistent after retraining."""
    try:
        _params, version = retraining.active_params_or_default()
        key = str(version)
        if _SAMPLE_CACHE.get("key") != key or not _SAMPLE_CACHE.get("data"):
            _SAMPLE_CACHE["data"] = _build_engine_aware_samples()
            _SAMPLE_CACHE["key"] = key
        return jsonify({"ok": True, "version": version, "samples": _SAMPLE_CACHE["data"]})
    except Exception:
        print("API_SAMPLES_ERROR:", traceback.format_exc(), flush=True)
        return jsonify({"ok": False, "error": "Could not build samples"}), 500


# ===============================================================
# Continuous-learning / admin routes
# ===============================================================
@app.get("/admin")
def admin_page():
    # Protect the admin console: it must not be freely visible.
    # Production note: prefer an `Authorization: Bearer <token>` header (handled
    # by _admin_ok). A browser navigation cannot set request headers, so for the
    # demo _admin_ok_nav() also accepts a ?token= query param. After a successful
    # unlock the token is kept in localStorage (see admin_locked.html / admin.js)
    # so the operator does not have to re-paste it on every visit.
    if ADMIN_TOKEN and not _admin_ok_nav(request):
        # If a token WAS supplied via the query string but did not match, flag it
        # so the gate shows an error and clears any stale saved token instead of
        # auto-redirecting in a loop.
        bad_token = bool(request.args.get("token"))
        return render_template("admin_locked.html", bad_token=bad_token), 401
    return render_template(
        "admin.html",
        using_gcs=storage.using_gcs(),
        threshold=retraining.RETRAIN_THRESHOLD,
        admin_open=(ADMIN_TOKEN == ""),
    )


def _parse_csv(text: str):
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        # Excel/Windows CSVs sometimes include a UTF-8 BOM or extra spaces in
        # headers. Normalise keys here so valid files are not incorrectly sent
        # to the manual review queue. Values are trimmed only when they are text.
        rows.append({
            str(k).strip().lstrip("\ufeff"): (v.strip() if isinstance(v, str) else v)
            for k, v in (r or {}).items()
            if k is not None
        })
    return rows


@app.post("/api/admin/upload")
def api_admin_upload():
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    try:
        rows = []
        if request.files.get("file"):
            text = request.files["file"].read().decode("utf-8-sig", errors="ignore")
            rows = _parse_csv(text)
        elif request.is_json:
            payload = request.get_json(silent=True) or {}
            rows = payload.get("rows", [])
        if not rows:
            return jsonify({"ok": False, "error": "No rows found. Upload a CSV with the required columns."}), 400
        result = retraining.intake_rows(rows)
        # Phase 3: check the volume trigger ONCE, after the whole batch is in
        # pending (never per-row). Auto-starts a retrain if clean pending >= 5%.
        trigger = maybe_trigger_retraining(source="upload")
        return jsonify({"ok": True, **result, "auto_retrain": trigger})
    except Exception:
        return jsonify({"ok": False, "error": "Upload could not be processed. Check the CSV columns."}), 400


@app.get("/api/admin/state")
def api_admin_state():
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    # Heal stale Running status on refresh so the admin page does not stay
    # permanently stuck after a completed/interrupted worker.
    if hasattr(storage, "recover_stale_retrain_status"):
        storage.recover_stale_retrain_status()
    fresh = retraining.check_data_freshness()
    active_params, active_version = retraining.active_params_or_default()
    pending = storage.get_pending()
    current_batch = storage.get_current_batch() if hasattr(storage, "get_current_batch") else {}
    review = storage.get_review()
    pending_preview_limit = 200
    review_preview_limit = 200
    return jsonify({
        "ok": True,
        "freshness": fresh,
        "status": storage.get_status(),
        "active_version": active_version,
        # `pending` / `review` are previews only, so large uploads do not make
        # the admin payload huge. The metric cards use the totals below.
        "pending": pending[:pending_preview_limit],
        "pending_total": len(pending),
        "current_batch": {
            "run_id": current_batch.get("run_id"),
            "row_count": current_batch.get("row_count", len(current_batch.get("rows") or [])) if isinstance(current_batch, dict) else 0,
            "time": current_batch.get("time") if isinstance(current_batch, dict) else None,
        },
        "pending_preview_limit": pending_preview_limit,
        "review": review[:review_preview_limit],
        "review_total": len(review),
        "review_preview_limit": review_preview_limit,
        "versions": storage.list_versions(),
        "runs": storage.get_runs()[:100],
        "attempted_batches": storage.get_attempted_batches()[:50],
        "changelog": storage.read_changelog().splitlines()[-80:],
        "using_gcs": storage.using_gcs(),
    })


@app.post("/api/admin/retrain")
def api_admin_retrain():
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force", False))

    # Route through the one shared trigger. Force bypasses the volume threshold
    # but not the lock; normal manual retrain still respects the 5% threshold.
    trigger = maybe_trigger_retraining(source="manual", force=force)
    if not trigger["triggered"]:
        if trigger["status"] == "skipped_already_running":
            return jsonify({"ok": False, "error": "Retraining already in progress."}), 409
        fresh = trigger["freshness"]
        return jsonify({
            "ok": False,
            "code": "THRESHOLD_NOT_MET",
            "error": f"Retrain requires at least {fresh.get('threshold')} clean rows. Use Force retrain to override.",
            "freshness": fresh,
        }), 400

    return jsonify({"ok": True, "message": "Retraining queued.", "queue": trigger.get("queue"),
                    "status": storage.get_status(), "freshness": trigger["freshness"]})


@app.post("/api/admin/review/<action>")
def api_admin_review(action):
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    idx = payload.get("index")
    review = storage.get_review()
    if idx is None or idx < 0 or idx >= len(review):
        return jsonify({"ok": False, "error": "Invalid item."}), 400
    item = review.pop(idx)
    if action == "approve":
        # Admin approval is an explicit override of Gate 1.
        # Do not send the row back through intake_rows(), or it may be rejected again.
        try:
            storage.add_pending([retraining.normalise_row(item["row"])])
        except Exception:
            item.setdefault("reasons", []).append("approve failed: row could not be normalised")
            review.insert(idx, item)
            storage.set_review(review)
            return jsonify({"ok": False, "error": "Row could not be normalised."}), 400
    elif action != "discard":
        review.insert(idx, item)
        storage.set_review(review)
        return jsonify({"ok": False, "error": "Invalid review action."}), 400
    storage.set_review(review)
    return jsonify({"ok": True, "remaining": len(review), "pending_total": len(storage.get_pending())})


@app.post("/api/admin/rollback")
def api_admin_rollback():
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    version = (request.get_json(silent=True) or {}).get("version")
    rec = retraining.rollback(int(version)) if version is not None else None
    if not rec:
        return jsonify({"ok": False, "error": "Version not found."}), 404
    if hasattr(fe, "apply_model_params"):
        fe.apply_model_params(rec["params"], source=("v0 boot GA baseline" if int(version) == 0 else f"rollback artifact v{version}"))
    return jsonify({"ok": True, "active_version": version})






def _score_params_for_applicant(params: dict, income: float, credit: float, ratio: float, default: Any, emp: float) -> float:
    """Score one applicant with a supplied version artifact without changing
    the live engine. Used by the admin-only Model Impact Test.
    """
    if not hasattr(fe, "_build_control_system_from_params"):
        raise RuntimeError("Version scoring helper is unavailable.")
    built = fe._build_control_system_from_params(params)
    sim = fe.ctrl.ControlSystemSimulation(built["risk_ctrl"])
    sim.input["income"] = float(income)
    sim.input["credit"] = float(credit)
    sim.input["ratio"] = float(ratio)
    sim.input["default"] = _default_to_value(default)
    sim.input["emp_exp"] = float(emp)
    sim.compute()
    return float(sim.output["risk"])


def _version_records_for_impact() -> list[dict]:
    """Return v0 + promoted versions, newest first, deduped by version."""
    records = []
    seen = set()
    for rec in storage.list_versions():
        try:
            v = int(rec.get("version", -1))
        except Exception:
            continue
        if v in seen:
            continue
        seen.add(v)
        records.append(rec)
    if 0 not in seen:
        records.append(retraining.version0_record())
    records.sort(key=lambda r: int(r.get("version", 0)), reverse=True)
    return records


@app.post("/api/admin/model-impact")
def api_admin_model_impact():
    """Compare how rollbackable versions score the same applicant.

    This does not promote/rollback anything. It is an operator verification tool
    to prove that version artifacts can change the served fuzzy score even when
    the same applicant remains in the same decision band.
    """
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    income = float(payload.get("income", 65000))
    credit = float(payload.get("credit", 720))
    loan_amount = float(payload.get("loan_amount", 11700))
    ratio = float(payload.get("ratio", (loan_amount / income if income else 0)))
    emp = float(payload.get("emp", 4))
    default = payload.get("default", "No")
    active_params, active_version = retraining.active_params_or_default()

    rows = []
    for rec in _version_records_for_impact():
        if not rec or "params" not in rec:
            continue
        version = int(rec.get("version", 0))
        try:
            score = _score_params_for_applicant(rec["params"], income, credit, ratio, default, emp)
        except Exception as exc:
            rows.append({"version": version, "error": str(exc)})
            continue
        decision = "APPROVE" if score < 40 else ("REVIEW" if score < 70 else "REJECT")
        rows.append({
            "version": version,
            "label": f"v{version}" + (" baseline" if version == 0 else ""),
            "active": version == active_version,
            "score": round(score, 4),
            "score_display": round(score, 1),
            "decision": decision,
            "validation_mse": rec.get("new_mse"),
        })
    active_row = next((r for r in rows if r.get("active")), None)
    if active_row:
        for r in rows:
            if "score" in r:
                r["delta_vs_active"] = round(float(r["score"]) - float(active_row["score"]), 4)
    return jsonify({
        "ok": True,
        "active_version": active_version,
        "applicant": {
            "income": income, "credit": credit, "loan_amount": loan_amount,
            "ratio": round(ratio, 4), "emp": emp, "default": default,
        },
        "versions": rows,
        "note": "Scores are computed from stored version artifacts without changing the active model.",
    })


@app.post("/api/admin/reset-state")
def api_admin_reset_state():
    """Admin-only demo reset: clear uploaded learning state and return to v0.

    This is intentionally separate from rollback. Rollback changes only the active
    model; reset clears accumulated uploaded rows and retraining history so a demo
    can start from a fresh base dataset again.
    """
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    if payload.get("confirm") != "RESET":
        return jsonify({"ok": False, "error": "Confirmation required."}), 400
    storage.reset_learning_state(keep_v0=True)
    # Restore active model pointer to v0 baseline. Prefer stored v0; fall back to
    # the current boot baseline if a v0 artifact does not exist.
    rec = retraining.version0_record()
    storage.set_active_params(rec)
    if hasattr(fe, "apply_model_params"):
        fe.apply_model_params(rec["params"], source="v0 baseline after demo reset")
    storage.set_status("idle", "Demo learning state reset to v0 baseline.")
    return jsonify({"ok": True, "active_version": 0, "message": "Demo learning state reset to v0 baseline."})


@app.get("/api/admin/download/<int:version>")
def api_admin_download(version):
    if not _admin_ok_nav(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = retraining.version0_zip() if int(version) == 0 else storage.read_version_zip(version)
    if not data:
        return jsonify({"ok": False, "error": "Version not found."}), 404
    from flask import Response
    return Response(
        data, mimetype="application/zip",
        headers={"Content-Disposition": f"attachment; filename=model_v{version}.zip"},
    )


@app.post("/tasks/retrain")
def task_retrain():
    """Cloud Tasks worker endpoint.

    Cloud Tasks calls this endpoint and waits for a 200 response after the GA
    finishes. This keeps the user/scheduler request short while giving the GA
    its own HTTP request lifecycle.
    """
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    job_id = payload.get("job_id")
    force = bool(payload.get("force", False))
    owner_token = payload.get("owner_token")
    status = storage.get_status()
    if status.get("state") != "running":
        return jsonify({"ok": True, "skipped": "no_running_job", "job_id": job_id})
    if job_id and status.get("job_id") and status.get("job_id") != job_id:
        return jsonify({"ok": True, "skipped": "stale_job", "job_id": job_id})
    try:
        result = _run_retraining_job(force=force, job_id=job_id, owner_token=owner_token)
        follow = maybe_chain_retraining_after_run(result, source="chain_task")
        return jsonify({"ok": True, "job_id": job_id, "result": result, "follow_up": follow})
    except Exception as exc:
        try:
            storage.restore_current_batch(job_id, reason="task failed")
        except Exception:
            pass
        storage.set_status("idle", "Retraining failed — locked batch restored for retry.", extra={"job_id": job_id})
        if owner_token:
            storage.release_retrain_lock(owner_token)
        storage.add_run({
            "run_id": job_id or "task_unknown",
            "time": storage.now_iso(),
            "status": "failed",
            "version": None,
            "old_mse": None,
            "new_mse": None,
            "train_rows": None,
            "validation_rows": None,
            "pending_rows": None,
            "new_rows_used": None,
            "reason": "Retraining task failed. Pending rows were kept for retry.",
        })
        print(f"[tasks/retrain] failed: {exc}", flush=True)
        return jsonify({"ok": False, "error": "Retraining failed."}), 500


@app.route("/cron/check-retrain", methods=["GET", "POST"])
def cron_check_retrain():
    """Pinged hourly by Cloud Scheduler. Phase 3: cron does NOT bypass the volume
    trigger — it re-checks the same 5%-of-base threshold via the shared function.
    Only fires when clean pending is sufficient and no run is already in progress."""
    if not _admin_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    trigger = maybe_trigger_retraining(source="cron")
    return jsonify(trigger)



# =============================================================================
# Telegram bot (webhook) — CONVERSATIONAL mode.
#
# Each incoming message is handed to the existing Gemini tool-calling agent,
# which extracts the five inputs from free-form text, asks brief follow-ups for
# anything missing, computes the debt-to-income ratio, calls the live fuzzy
# engine (calculate_loan_risk) when complete, then explains the decision.
#
# Per-chat conversation history is persisted in the shared storage layer
# (GCS/local), so context survives across Cloud Run instances and cold starts.
# Only the plain user/model TEXT turns are stored; the within-turn tool
# round-trips are ephemeral.
# =============================================================================
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")   # set a random string
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"
TG_HISTORY_LIMIT = 24   # max stored turns per chat (keeps context bounded)

# Best-effort in-instance dedup of Telegram update_id, so a retried update
# (Telegram resends if it doesn't get a fast 2xx) isn't processed twice.
_TG_SEEN_UPDATES = collections.OrderedDict()
_TG_SEEN_MAX = 512
_TG_SEEN_LOCK = threading.Lock()


def _tg_seen_update(update_id):
    if update_id is None:
        return False
    with _TG_SEEN_LOCK:
        if update_id in _TG_SEEN_UPDATES:
            return True
        _TG_SEEN_UPDATES[update_id] = True
        while len(_TG_SEEN_UPDATES) > _TG_SEEN_MAX:
            _TG_SEEN_UPDATES.popitem(last=False)
    return False


def _tg_spawn(fn, *args, **kwargs):
    """Run a heavy Telegram handler off the request thread so the webhook can
    return 200 immediately. NOTE: on Cloud Run this requires CPU to remain
    allocated after the response — set --no-cpu-throttling and --min-instances=1
    (which also removes cold starts), otherwise the background thread is frozen.
    """
    def _runner():
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"[telegram] background handler failed: {exc}", flush=True)
    threading.Thread(target=_runner, daemon=True).start()

TG_SYSTEM_PROMPT = """You are a friendly Malaysian loan officer assistant chatting on Telegram.

Through natural conversation, collect these FIVE inputs from the user:
1. annual income (RM)
2. CTOS credit score (300-850)
3. requested loan amount (RM)
4. years of employment experience
5. previous loan default? (Yes/No)

Rules:
- Extract whatever the user gives in free text, in any order. Don't force a rigid sequence.
- If something is missing or invalid, ask a short, friendly follow-up for ONLY the missing/invalid item(s).
- Compute debt_to_income_ratio = requested_loan_amount / annual_income. It must be between 0 and 1; if it is above 1, tell the user the requested loan is too large for their income and ask for a smaller amount.
- Once you have all five, call the calculate_loan_risk tool with income, credit, ratio, default, emp.
- After the tool returns, give a short verdict: the decision (APPROVE / REVIEW / REJECT), the risk score out of 100, and 1-2 practical tips to improve the outcome.
- Base every explanation ONLY on the tool output. Never invent numbers or reasons.
- Keep replies concise and plain (Telegram); reply in English. You may use the lookup_ctos_category tool to describe the credit band.
"""


def _tg_call(method, payload):
    """JSON Telegram API call for messages, chat actions and callback answers."""
    if not TG_TOKEN:
        return None
    try:
        req = urllib.request.Request(
            f"{TG_API}/{method}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            try:
                return json.loads(resp.read().decode("utf-8"))
            except Exception:
                return None
    except Exception as exc:
        print(f"[telegram] send failed ({method}): {exc}", flush=True)
        return None


def _tg_send(chat_id, text, reply_markup=None, parse_mode="HTML"):
    payload = {"chat_id": chat_id, "text": text[:4000]}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    return _tg_call("sendMessage", payload)


def _tg_typing(chat_id):
    _tg_call("sendChatAction", {"chat_id": chat_id, "action": "typing"})


def _tg_answer_callback(callback_id, text=None):
    payload = {"callback_query_id": callback_id}
    if text:
        payload["text"] = text
    return _tg_call("answerCallbackQuery", payload)


def _tg_send_photo(chat_id, image_path, caption=None, reply_markup=None):
    """Send a generated PNG chart to Telegram using multipart upload."""
    if not TG_TOKEN or requests is None:
        return None
    try:
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption[:1024]
            data["parse_mode"] = "HTML"
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        with open(image_path, "rb") as f:
            return requests.post(
                f"{TG_API}/sendPhoto",
                data=data,
                files={"photo": f},
                timeout=35,
            )
    except Exception as exc:
        print(f"[telegram] sendPhoto failed: {exc}", flush=True)
        return None


def _tg_send_document(chat_id, file_path, caption=None, filename=None):
    """Send a generated PDF/report to Telegram."""
    if not TG_TOKEN or requests is None:
        return None
    try:
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption[:1024]
        with open(file_path, "rb") as f:
            files = {"document": (filename or os.path.basename(file_path), f)}
            return requests.post(f"{TG_API}/sendDocument", data=data, files=files, timeout=35)
    except Exception as exc:
        print(f"[telegram] sendDocument failed: {exc}", flush=True)
        return None


def _tg_history_key(chat_id):
    return f"telegram/chat_{chat_id}.json"


def _tg_last_key(chat_id):
    return f"telegram/last_assessment_{chat_id}.json"


def _tg_get_history(chat_id):
    return storage.read_json(_tg_history_key(chat_id), default=[]) or []


def _tg_save_history(chat_id, history):
    storage.write_json(_tg_history_key(chat_id), history[-TG_HISTORY_LIMIT:])


def _tg_clear_history(chat_id):
    try:
        storage.delete_key(_tg_history_key(chat_id))
        storage.delete_key(_tg_last_key(chat_id))
    except Exception:
        pass


def _tg_save_last_assessment(chat_id, record):
    try:
        storage.write_json(_tg_last_key(chat_id), record)
    except Exception as exc:
        print(f"[telegram] save last assessment failed: {exc}", flush=True)


def _tg_get_last_assessment(chat_id):
    return storage.read_json(_tg_last_key(chat_id), default=None)


def _tg_buttons(include_report=True):
    rows = []
    if include_report:
        rows.append([
            {"text": "🧾 Download report", "callback_data": "download_report"},
            {"text": "🔄 Reset", "callback_data": "reset"},
        ])
    else:
        rows.append([{"text": "🔄 Reset", "callback_data": "reset"}])
    rows.append([
        {"text": "✅ Approve sample", "callback_data": "sample_approve"},
        {"text": "⚠️ Review sample", "callback_data": "sample_review"},
        {"text": "❌ Reject sample", "callback_data": "sample_reject"},
    ])
    return {"inline_keyboard": rows}


def _decision_level(score):
    score = float(score)
    if score < 40:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def _tg_format_summary(applicant, ctos, result):
    decision = result.get("decision", "REVIEW")
    score = float(result.get("risk_score", 0))
    risk_level = _decision_level(score)
    recommendation = "Approve loan" if decision == "APPROVE" else ("Manual review required" if decision == "REVIEW" else "Reject loan")
    emoji = "🟢" if decision == "APPROVE" else ("🟠" if decision == "REVIEW" else "🔴")
    return (
        "<b>Loan Assessment Result</b>\n\n"
        f"{emoji} <b>Decision:</b> {decision}\n"
        f"<b>Risk Score:</b> {score:.1f} / 100\n"
        f"<b>Risk Level:</b> {risk_level}\n"
        f"<b>Recommendation:</b> {recommendation}\n\n"
        f"<b>Applicant:</b> Income RM{float(applicant.get('income', 0)):,.0f} · "
        f"Loan RM{float(applicant.get('loan_amount', 0)):,.0f} · "
        f"DTI {float(applicant.get('ratio', 0)):.2f} · "
        f"CTOS {int(float(applicant.get('credit', 0)))} ({ctos.get('category', '—')})\n\n"
        "<i>Powered by AI + Fuzzy Risk Engine</i>"
    )


def _tg_generate_risk_gauge(score, decision, output_path):
    """Generate a Telegram-friendly semi-circle risk gauge PNG."""
    if plt is None or Wedge is None:
        return False
    score = max(0.0, min(100.0, float(score)))
    level = _decision_level(score)

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_aspect("equal")
    ax.axis("off")

    zones = [
        (0, 40, "#16A34A", "Low Risk"),
        (40, 70, "#F59E0B", "Medium Risk"),
        (70, 100, "#DC2626", "High Risk"),
    ]
    for start, end, color, _label in zones:
        theta1 = 180 - (end / 100.0) * 180
        theta2 = 180 - (start / 100.0) * 180
        ax.add_patch(Wedge((0, 0), 1.0, theta1, theta2, width=0.18,
                           facecolor=color, edgecolor="#FFFFFF", linewidth=2))

    angle = math.radians(180 - (score / 100.0) * 180)
    x = 0.78 * math.cos(angle)
    y = 0.78 * math.sin(angle)
    ax.plot([0, x], [0, y], color="#111827", linewidth=3.0, solid_capstyle="round")
    ax.scatter([0], [0], s=95, color="#111827", zorder=5)

    ax.text(0, -0.14, f"{score:.1f}/100", ha="center", va="center",
            fontsize=28, fontweight="bold", color="#111827")
    ax.text(0, -0.35, f"{decision} · {level} Risk", ha="center", va="center",
            fontsize=14, color="#374151")
    ax.text(0, -0.56, "Risk Score: Lower = Safer (<40 Approvable)", ha="center", va="center",
            fontsize=10.5, color="#6B7280")
    ax.text(-0.96, -0.77, "● Low Risk", fontsize=10, color="#16A34A")
    ax.text(-0.20, -0.77, "● Medium Risk", fontsize=10, color="#F59E0B")
    ax.text(0.65, -0.77, "● High Risk", fontsize=10, color="#DC2626")

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.93, 1.15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _tg_format_reasons_and_whatif(applicant, result, reasons, what_if):
    lines = ["<b>Main Reasons</b>"]
    for item in reasons[:4]:
        lines.append(f"• {item.get('text', '')}")
    lines.append("")
    lines.append("<b>What-if Suggestion</b>")
    if what_if:
        best = what_if[0]
        lines.append(f"• {best.get('delta_text', 'Risk can be reduced')}: {best.get('advice', '')}")
    elif result.get("decision") == "APPROVE":
        lines.append("• No major adjustment is required for this profile.")
    else:
        lines.append("• This profile needs major restructuring — consider reducing the loan amount significantly.")
    return "\n".join(lines)


def _tg_build_assessment_from_tool(args, result):
    applicant = {
        "income": float(args.get("income", 0)),
        "credit": float(args.get("credit", 0)),
        "ratio": float(args.get("ratio", 0)),
        "default": str(args.get("default", "No")),
        "emp": float(args.get("emp", 0)),
    }
    applicant["loan_amount"] = round(applicant["income"] * applicant["ratio"], 2)
    ctos = lookup_ctos_category(applicant["credit"])
    reasons = reasons_from_degrees(result.get("membership_degrees", {}))
    what_if = build_what_if(applicant["income"], applicant["credit"], applicant["ratio"], applicant["default"], applicant["emp"], current=result)
    return applicant, ctos, reasons, what_if


def _tg_send_rich_assessment(chat_id, applicant, ctos, result, reasons, what_if, ai_answer=None):
    """Send summary + gauge + reasons. Persist last assessment for PDF callback."""
    _tg_send(chat_id, _tg_format_summary(applicant, ctos, result))

    image_path = None
    try:
        fd, image_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        if _tg_generate_risk_gauge(result.get("risk_score", 0), result.get("decision", "REVIEW"), image_path):
            _tg_send_photo(chat_id, image_path, caption="📊 <b>Risk score visualization</b>")
    except Exception as exc:
        print(f"[telegram] gauge generation failed: {exc}", flush=True)
    finally:
        if image_path:
            try: os.remove(image_path)
            except Exception: pass

    _tg_send(chat_id, _tg_format_reasons_and_whatif(applicant, result, reasons, what_if), reply_markup=_tg_buttons(include_report=True))
    _tg_save_last_assessment(chat_id, {
        "applicant": applicant,
        "ctos": ctos,
        "result": result,
        "reasons": reasons,
        "what_if": what_if,
        "ai_answer": ai_answer or "",
        "created": storage.now_iso() if hasattr(storage, "now_iso") else "",
    })


def _tg_generate_pdf_report(record, output_path):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
    except Exception:
        return False

    applicant = record.get("applicant", {})
    ctos = record.get("ctos", {})
    result = record.get("result", {})
    reasons = record.get("reasons", [])
    what_if = record.get("what_if", [])

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    y = height - 56
    c.setFont("Helvetica-Bold", 16)
    c.drawString(48, y, "Malaysian Loan Approval Assessment Report")
    y -= 22
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(48, y, "Generated by AI + Fuzzy Risk Engine")
    c.setFillColor(colors.black)

    y -= 42
    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, y, "Decision Summary")
    y -= 22
    c.setFont("Helvetica", 10)
    c.drawString(58, y, f"Decision: {result.get('decision', '—')}")
    y -= 18
    c.drawString(58, y, f"Risk Score: {float(result.get('risk_score', 0)):.1f}/100")
    y -= 18
    c.drawString(58, y, f"Risk Level: {_decision_level(float(result.get('risk_score', 0)))}")

    y -= 34
    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, y, "Applicant Profile")
    y -= 22
    c.setFont("Helvetica", 10)
    profile_rows = [
        ("Annual Income", f"RM{float(applicant.get('income', 0)):,.0f}"),
        ("Requested Loan", f"RM{float(applicant.get('loan_amount', 0)):,.0f}"),
        ("Debt-to-Income Ratio", f"{float(applicant.get('ratio', 0)):.2f}"),
        ("CTOS", f"{int(float(applicant.get('credit', 0)))} ({ctos.get('category', '—')})"),
        ("Employment", f"{float(applicant.get('emp', 0)):.1f} years"),
        ("Previous Default", str(applicant.get('default', 'No'))),
    ]
    for label, value in profile_rows:
        c.drawString(58, y, f"{label}: {value}")
        y -= 18

    y -= 18
    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, y, "Main Reasons")
    y -= 22
    c.setFont("Helvetica", 10)
    for item in reasons[:5]:
        text = f"• {item.get('text', '')}"[:105]
        c.drawString(58, y, text)
        y -= 18
        if y < 90:
            c.showPage(); y = height - 56

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, y, "What-if Suggestion")
    y -= 22
    c.setFont("Helvetica", 10)
    if what_if:
        best = what_if[0]
        c.drawString(58, y, f"{best.get('delta_text', '')}: {best.get('advice', '')}"[:105])
    elif result.get("decision") == "APPROVE":
        c.drawString(58, y, "No major adjustment is required for this profile.")
    else:
        c.drawString(58, y, "This profile needs major restructuring — consider reducing loan amount significantly.")
    c.save()
    return True

def run_gemini_chat(history, system_prompt=TG_SYSTEM_PROMPT):
    """Conversational turn of the Gemini tool-calling agent.

    Returns {"text": ..., "events": [...]} so Telegram can send a visual risk
    gauge after the calculate_loan_risk tool is called. The web evaluate path
    reuses this with a single-turn history and the web SYSTEM_PROMPT.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None or types is None:
        return None
    try:
        client = genai.Client(api_key=api_key)
        model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        tools = _gemini_tools()

        contents = []
        for turn in history:
            role = "model" if turn.get("role") == "model" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=turn.get("text", ""))]))

        config_kwargs = dict(
            system_instruction=system_prompt,
            tools=[tools],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        _tc = _gemini_thinking_config()
        if _tc is not None:
            config_kwargs["thinking_config"] = _tc
        config = types.GenerateContentConfig(**config_kwargs)

        final_text_parts = []
        events = []
        for _ in range(8):
            response = client.models.generate_content(model=model, contents=contents, config=config)
            candidate = response.candidates[0]
            parts = candidate.content.parts or []
            function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
            text_parts = [p.text for p in parts if getattr(p, "text", None)]
            final_text_parts.extend([t.strip() for t in text_parts if t.strip()])

            if function_calls:
                contents.append(candidate.content)
                response_parts = []
                for call in function_calls:
                    args = dict(call.args)
                    result = execute_agent_tool(call.name, args)
                    events.append({"tool": call.name, "args": args, "result": result})
                    response_parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
                contents.append(types.Content(role="user", parts=response_parts))
                continue
            break

        return {"text": "\n".join(final_text_parts).strip(), "events": events, "provider": "Gemini"}
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc(limit=2)}


# ----------------------------------------------------------------------------
# OpenAI-compatible fallback providers (Cerebras gpt-oss-120b -> Groq llama-3.3)
# Both expose the same /chat/completions schema, so one implementation serves
# both; only base_url, key and model differ. These fire only when the primary
# Gemini turn fails, which is what kills the "temporary error" dead-ends.
# ----------------------------------------------------------------------------
def _openai_tools():
    """Same tools as _gemini_tools(), in OpenAI function-calling schema."""
    return [
        {"type": "function", "function": {
            "name": "calculate_loan_risk",
            "description": "Run the GA-optimised fuzzy logic engine. Returns risk score, decision and membership degrees.",
            "parameters": {"type": "object", "properties": {
                "income": {"type": "number", "description": "Annual income in RM"},
                "credit": {"type": "number", "description": "CTOS score from 300 to 850"},
                "ratio": {"type": "number", "description": "Loan fraction of income from 0 to 1"},
                "default": {"type": "string", "description": "Yes or No"},
                "emp": {"type": "number", "description": "Employment experience in years"},
            }, "required": ["income", "credit", "ratio", "default", "emp"]},
        }},
        {"type": "function", "function": {
            "name": "lookup_ctos_category",
            "description": "Look up the CTOS credit category for a credit score.",
            "parameters": {"type": "object", "properties": {
                "credit": {"type": "number", "description": "CTOS credit score"},
            }, "required": ["credit"]},
        }},
    ]


def _openai_fallback_providers():
    """Ordered fallback legs after Gemini, each OpenAI-compatible.

    Configure via env: CEREBRAS_API_KEY / GROQ_API_KEY (legs without a key are
    skipped). Models overridable with CEREBRAS_MODEL / GROQ_CHAT_MODEL.
    """
    return [
        {
            "name": "cerebras",
            "base_url": os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
            "api_key": os.environ.get("CEREBRAS_API_KEY", ""),
            "model": os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b"),
            "extra": {"reasoning_effort": "low"},   # gpt-oss low-latency mode
        },
        {
            "name": "groq",
            "base_url": os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            "api_key": os.environ.get("GROQ_API_KEY", ""),
            "model": os.environ.get("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"),
            "extra": {},
        },
    ]


def run_openai_compat_chat(history, provider, system_prompt=TG_SYSTEM_PROMPT):
    """One agent turn against an OpenAI-compatible chat endpoint.

    Returns {"text", "events", "model"} on success, {"error": ...} on failure,
    or None when the provider has no API key (so the caller skips to the next).
    """
    if requests is None:
        return None
    api_key = provider.get("api_key")
    if not api_key:
        return None

    url = provider["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        role = "assistant" if turn.get("role") == "model" else "user"
        messages.append({"role": role, "content": turn.get("text", "")})

    tools = _openai_tools()
    events = []
    try:
        for _ in range(8):
            payload = {
                "model": provider["model"],
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.3,
            }
            payload.update(provider.get("extra") or {})
            resp = requests.post(url, headers=headers, json=payload, timeout=40)
            if resp.status_code != 200:
                return {"error": f"{provider['name']} HTTP {resp.status_code}: {resp.text[:200]}"}

            message = resp.json()["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                # Echo the assistant turn (with its tool_calls), then tool results.
                messages.append({
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                })
                for call in tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    result = execute_agent_tool(name, args)
                    events.append({"tool": name, "args": args, "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": json.dumps({"result": result}),
                    })
                continue

            text = (message.get("content") or "").strip()
            return {"text": text, "events": events, "model": provider["model"], "provider": provider["name"].capitalize()}

        # Loop budget exhausted (rare) — return whatever tool events we gathered.
        return {"text": "", "events": events, "model": provider["model"], "provider": provider["name"].capitalize()}
    except Exception as exc:
        return {"error": f"{provider['name']}: {exc}"}


def run_agent_chain(history, system_prompt=TG_SYSTEM_PROMPT):
    """Agent turn with provider fallback: Gemini -> Cerebras -> Groq.

    Shared by the Telegram bot (conversational, TG_SYSTEM_PROMPT) and the web
    evaluate path (single-turn, web SYSTEM_PROMPT). All three legs do real
    tool-calling against the fuzzy engine. Returns the first successful reply
    ({"text", "events", ...}); if every configured provider fails, returns an
    aggregated {"error": ...}.
    """
    errors = []

    primary = run_gemini_chat(history, system_prompt=system_prompt)
    if primary is not None and not primary.get("error"):
        return primary
    if primary is not None and primary.get("error"):
        errors.append(f"gemini: {primary.get('error')}")

    for provider in _openai_fallback_providers():
        reply = run_openai_compat_chat(history, provider, system_prompt=system_prompt)
        if reply is None:
            continue  # provider not configured — skip silently
        if not reply.get("error"):
            print(f"[agent] served by fallback provider: {provider['name']}", flush=True)
            return reply
        errors.append(reply.get("error"))

    return {"error": "; ".join(e for e in errors if e) or "No LLM provider configured."}


def _tg_latest_risk_event(reply):
    for event in reversed(reply.get("events") or []):
        if event.get("tool") == "calculate_loan_risk" and isinstance(event.get("result"), dict):
            return event
    return None


def _tg_sample_text(kind):
    if kind == "sample_approve":
        return "Annual income 90000, loan amount 9000, CTOS 780, employment 10 years, no previous default."
    if kind == "sample_review":
        return "Annual income 36000, loan amount 14400, CTOS 610, employment 2 years, no previous default."
    if kind == "sample_reject":
        return "Annual income 30000, loan amount 16500, CTOS 580, employment 1 year, previous default yes."
    return ""


def _tg_process_text(chat_id, text, reset_history=False):
    if reset_history:
        _tg_clear_history(chat_id)
    history = _tg_get_history(chat_id)
    history.append({"role": "user", "text": text})

    _tg_typing(chat_id)
    reply = run_agent_chain(history)

    if reply is None:
        _tg_send(chat_id, "The AI assistant isn't configured (missing GEMINI_API_KEY). Please contact the administrator.", parse_mode=None)
        return
    if reply.get("error"):
        print(f"[telegram] gemini chat error: {reply.get('error')}", flush=True)
        _tg_send(chat_id, "Sorry, I hit a temporary error. Please try again in a moment.", parse_mode=None)
        return

    answer = reply.get("text") or "Could you give me a bit more detail about the applicant?"
    history.append({"role": "model", "text": answer})
    _tg_save_history(chat_id, history)

    risk_event = _tg_latest_risk_event(reply)
    if risk_event:
        try:
            applicant, ctos, reasons, what_if = _tg_build_assessment_from_tool(risk_event.get("args", {}), risk_event.get("result", {}))
            _tg_send_rich_assessment(chat_id, applicant, ctos, risk_event.get("result", {}), reasons, what_if, ai_answer=answer)
        except Exception as exc:
            print(f"[telegram] rich assessment failed: {exc}", flush=True)
            _tg_send(chat_id, answer, parse_mode=None)
    else:
        # Missing-field follow-ups remain plain text.
        _tg_send(chat_id, answer, parse_mode=None)


def _tg_handle_callback(callback):
    callback_id = callback.get("id")
    msg = callback.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    data = callback.get("data")
    _tg_answer_callback(callback_id)
    if not chat_id:
        return

    if data == "reset":
        _tg_clear_history(chat_id)
        _tg_send(chat_id, "Cleared. Tell me about a new applicant whenever you're ready.", parse_mode=None)
        return

    if data in {"sample_approve", "sample_review", "sample_reject"}:
        sample = _tg_sample_text(data)
        _tg_send(chat_id, f"Using sample profile:\n{sample}", parse_mode=None)
        _tg_process_text(chat_id, sample, reset_history=True)
        return

    if data == "download_report":
        record = _tg_get_last_assessment(chat_id)
        if not record:
            _tg_send(chat_id, "No assessment report is available yet. Send an applicant profile first.", parse_mode=None)
            return
        pdf_path = None
        try:
            fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            if _tg_generate_pdf_report(record, pdf_path):
                _tg_send_document(chat_id, pdf_path, caption="Loan assessment report", filename="loan_assessment_report.pdf")
            else:
                _tg_send(chat_id, "PDF generation is not available on this deployment. Please check reportlab in requirements.txt.", parse_mode=None)
        except Exception as exc:
            print(f"[telegram] pdf failed: {exc}", flush=True)
            _tg_send(chat_id, "Could not generate the PDF report. Please try again.", parse_mode=None)
        finally:
            if pdf_path:
                try: os.remove(pdf_path)
                except Exception: pass

@app.post("/telegram/webhook")
def telegram_webhook():
    # Security: verify the secret header configured during setWebhook.
    if TG_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != TG_SECRET:
        return "forbidden", 403

    update = request.get_json(force=True, silent=True) or {}

    # Ignore Telegram retries of an update we've already accepted.
    if _tg_seen_update(update.get("update_id")):
        return "ok", 200

    if "callback_query" in update:
        _tg_spawn(_tg_handle_callback, update["callback_query"])
        return "ok", 200

    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        return "ok", 200

    chat_id = msg["chat"]["id"]
    text = msg["text"].strip()

    if text in ("/start", "/help"):
        _tg_clear_history(chat_id)
        _tg_send(chat_id,
                 "Hi! I'm the Malaysian Loan Approval assistant. Just tell me about the "
                 "applicant in your own words — annual income, CTOS score, the loan amount "
                 "they want, years of employment, and whether they've defaulted before. "
                 "I'll ask for anything I'm missing and then send a decision, risk chart, "
                 "main reasons, and an optional PDF report.\n\n"
                 "Example:\nAnnual income 65000, loan amount 11700, CTOS 720, employment 4 years, no previous default.\n\n"
                 "Send /reset anytime to start a new applicant.",
                 reply_markup=_tg_buttons(include_report=False),
                 parse_mode=None)
        return "ok", 200

    if text in ("/reset", "/cancel"):
        _tg_clear_history(chat_id)
        _tg_send(chat_id, "Cleared. Tell me about a new applicant whenever you're ready.", parse_mode=None)
        return "ok", 200

    if text in ("/sample_approve", "/sample_review", "/sample_reject"):
        kind = text[1:]
        _tg_spawn(_tg_process_text, chat_id, _tg_sample_text(kind), reset_history=True)
        return "ok", 200

    _tg_spawn(_tg_process_text, chat_id, text)
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
