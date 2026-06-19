# ================================================================
# retraining.py — continuous-learning loop for the fuzzy engine.
#
# Two-gate intake + a Retraining Agent that owns the retrain task:
#   check_data_freshness -> read_and_validate_data
#   -> retrain_fuzzy_engine -> evaluate_and_promote
#
# It NEVER edits fuzzy_engine.py. It reuses the engine's own MF math
# (build_mfs_from_params, fuzzy_risk_score_vec, decode_chromosome,
# INC/RAT/EMP bounds) so a retrained model is numerically identical
# in form to the production engine — only the data-derived breakpoints
# change. Promotion is gated on MSE over a FIXED validation slice.
# ================================================================

import io
import json
import math
import random
import zipfile

import numpy as np
import pandas as pd

import fuzzy_engine as fe
import storage

# Phase 3 — volume-based auto-retrain trigger.
# Auto retrain fires when CLEAN pending rows reach a fraction of the FIXED
# reference base (predictable + demo-friendly). Replaces the old 30-row trigger.
# RETRAIN_TRIGGER_MIN_ROWS can be overridden directly (e.g. lowered for a live
# demo); otherwise it is ceil(RETRAIN_TRIGGER_RATIO * REFERENCE_BASE_ROWS).
REFERENCE_BASE_ROWS = int(__import__("os").environ.get("REFERENCE_BASE_ROWS", "40500"))
RETRAIN_TRIGGER_RATIO = float(__import__("os").environ.get("RETRAIN_TRIGGER_RATIO", "0.05"))
RETRAIN_TRIGGER_MIN_ROWS = int(__import__("os").environ.get(
    "RETRAIN_TRIGGER_MIN_ROWS",
    str(math.ceil(REFERENCE_BASE_ROWS * RETRAIN_TRIGGER_RATIO))))
# Backward-compatible alias for existing freshness / UI code.
RETRAIN_THRESHOLD = RETRAIN_TRIGGER_MIN_ROWS

REQUIRED_COLUMNS = [
    "person_income", "credit_score", "loan_percent_income",
    "person_emp_exp", "previous_loan_defaults_on_file", "loan_status",
]

# --- GA config for retraining ---
# Keep retraining aligned with the v0 / boot GA in fuzzy_engine.py.
# This makes model-version comparisons fair: v0 and retraining use the same
# GA hyperparameters and operators; only the dataset stage changes.
GA_POP = int(fe.GA_POP)
GA_GENS = int(fe.GA_GENS)
GA_CR = float(fe.GA_CR)
GA_MR_BASE = float(fe.GA_MR_BASE)
GA_TOURNAMENT = int(fe.GA_TOURNAMENT)
GA_SEED = int(fe.GA_SEED)
# Phase 5 — multi-seed restart count (3–5). Reduces single-seed GA variance.
GA_RESTARTS = int(__import__("os").environ.get("GA_RESTARTS", "3"))
VALIDATION_FRACTION = 0.10
VALIDATION_SEED = GA_SEED

# Phase 1: promotion is gated by a margin so noise-level "improvements" do not
# bump the active version. A candidate must beat the active model on the fixed
# validation MSE by at least this much to be promoted.
PROMOTION_MARGIN = 1e-4

# Phase 2: blended (anchor + recent) promotion gate.
#   anchor  = the fixed original-distribution holdout (stable cross-version ruler,
#             protects against catastrophic forgetting).
#   recent  = the drift-aware holdout (checks adaptation to new data).
# These weights / tolerance are POLICY choices, not universal truths — tune them
# to how much old-distribution regression you will accept to chase recent drift.
ANCHOR_WEIGHT = 0.6
RECENT_WEIGHT = 0.4
MAX_ANCHOR_REGRESSION = 1.08    # candidate may not worsen anchor MSE beyond this x
MIN_RECENT_HOLDOUT = 200        # below this, fall back to anchor-only gate (no NaN)

# Optional speed cap on the rows the GA evaluates fitness against. Default 0 =
# no cap (train on all holdout-excluded rows). Set e.g. 12000 to speed up demos;
# applies equally to baseline and candidates so comparisons stay fair.
TRAIN_SAMPLE_CAP = int(__import__("os").environ.get("RETRAIN_TRAIN_CAP", "0"))

# Dedicated, fixed validation set carved once from loan_data.csv (stratified by
# loan_status). It is NEVER used for training, so old and new models are always
# scored on the exact same held-out data — a stable ruler across every retrain.
VALIDATION_FILE = "validation_holdout.csv"
VALIDATION_INDEX_FILE = "validation_holdout_index.json"


# ----------------------------------------------------------------
# Gate 1 — outlier test (loose: only block clearly impossible values)
# ----------------------------------------------------------------
def outlier_reasons(row: dict):
    reasons = []
    try:
        income = float(row.get("person_income"))
        if income <= 0 or income > 5_000_000:
            reasons.append("income out of plausible range")
    except (TypeError, ValueError):
        reasons.append("income not numeric")
    try:
        credit = float(row.get("credit_score"))
        if credit < 300 or credit > 850:
            reasons.append("credit score outside 300-850")
    except (TypeError, ValueError):
        reasons.append("credit score not numeric")
    try:
        ratio = float(row.get("loan_percent_income"))
        if ratio < 0 or ratio > 1:
            reasons.append("loan ratio outside 0-1")
    except (TypeError, ValueError):
        reasons.append("loan ratio not numeric")
    try:
        emp = float(row.get("person_emp_exp"))
        if emp < 0 or emp > 60:
            reasons.append("employment years out of range")
    except (TypeError, ValueError):
        reasons.append("employment years not numeric")
    d = str(row.get("previous_loan_defaults_on_file", "")).strip().lower()
    if d not in ("yes", "no"):
        reasons.append("default flag must be Yes/No")
    s = row.get("loan_status")
    if str(s) not in ("0", "1", "0.0", "1.0"):
        reasons.append("loan_status must be 0 or 1")
    return reasons


def normalise_row(row: dict) -> dict:
    return {
        "person_income": float(row["person_income"]),
        "credit_score": float(row["credit_score"]),
        "loan_percent_income": float(row["loan_percent_income"]),
        "person_emp_exp": float(row["person_emp_exp"]),
        "previous_loan_defaults_on_file": "Yes" if str(row["previous_loan_defaults_on_file"]).strip().lower() == "yes" else "No",
        "loan_status": int(float(row["loan_status"])),
    }


def intake_rows(rows):
    """Run Gate 1 on incoming rows. Clean rows -> pending pool; bad -> review queue."""
    accepted, rejected = [], []
    for raw in rows:
        reasons = outlier_reasons(raw)
        if reasons:
            rejected.append({"row": raw, "reasons": reasons, "stage": "outlier"})
        else:
            accepted.append(normalise_row(raw))
    if accepted:
        storage.add_pending(accepted)
    if rejected:
        storage.add_review(rejected)
    return {"accepted": len(accepted), "rejected": len(rejected),
            "pending_total": len(storage.get_pending())}


# ----------------------------------------------------------------
# Tool 1 — check_data_freshness
# ----------------------------------------------------------------
def check_data_freshness():
    pending = storage.get_pending()
    n = len(pending)
    return {"new_rows": n, "threshold": RETRAIN_TRIGGER_MIN_ROWS,
            "ratio": RETRAIN_TRIGGER_RATIO, "reference_base": REFERENCE_BASE_ROWS,
            "ready": n >= RETRAIN_TRIGGER_MIN_ROWS}


def trigger_decision(force: bool = False) -> dict:
    """Phase 3 — pure auto-retrain trigger policy (reads only, no side effects).

    Decides whether a retrain should START. Kept separate from the queueing /
    threading layer (web app) so it is unit-testable. Rules:
      * force bypasses the VOLUME threshold but NEVER the lock;
      * otherwise require clean pending >= RETRAIN_TRIGGER_MIN_ROWS;
      * never start if a run is already in progress.
    The lock here is the basic status-based guard; Phase 4 replaces it with an
    atomic lease lock (heartbeat + CAS + stale recovery).
    """
    clean_pending = len(storage.get_pending())
    threshold = RETRAIN_TRIGGER_MIN_ROWS
    enough = clean_pending >= threshold
    status = storage.get_status() or {}
    locked = status.get("state") == "running"
    freshness = {"new_rows": clean_pending, "threshold": threshold,
                 "ratio": RETRAIN_TRIGGER_RATIO, "reference_base": REFERENCE_BASE_ROWS,
                 "ready": enough}
    if locked:
        decision = "skipped_already_running"
    elif force or enough:
        decision = "trigger"
    else:
        decision = "skipped_not_enough_data"
    return {"decision": decision, "force": bool(force), "locked": locked,
            "clean_pending": clean_pending, "freshness": freshness}


# ----------------------------------------------------------------
# Tool 2 — read_and_validate_data (Gate 2: schema/type/logic quality)
# ----------------------------------------------------------------
def read_and_validate_data():
    pending = storage.get_pending()
    clean, bad = [], []
    for row in pending:
        problems = []
        for col in REQUIRED_COLUMNS:
            if col not in row or row[col] is None:
                problems.append(f"missing {col}")
        if not problems:
            # logic consistency
            if row["loan_percent_income"] > 0 and row["person_income"] <= 0:
                problems.append("ratio present but income is zero")
            if row["loan_status"] not in (0, 1):
                problems.append("loan_status not 0/1")
        (bad if problems else clean).append((row, problems))

    bad_rows = [{"row": r, "reasons": p, "stage": "quality"} for r, p in bad]
    if bad_rows:
        storage.add_review(bad_rows)
        # keep only clean rows in pending
        storage.set_pending([r for r, _ in clean])
    return {"clean": len(clean), "sent_to_review": len(bad_rows)}


# ----------------------------------------------------------------
# Build the combined training frame: base csv + clean pending rows
# ----------------------------------------------------------------
def _holdout_indices():
    """Row positions in loan_data.csv reserved for validation (if the file exists)."""
    import os
    path = os.path.join(os.path.dirname(__file__), VALIDATION_INDEX_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return set(json.load(fh))
    except Exception:
        return None


def _load_base_frame() -> pd.DataFrame:
    """Training base = loan_data.csv (MINUS the fixed validation holdout)
    + all previously-accumulated accepted rows."""
    df = pd.read_csv(fe.DATASET_PATH).reset_index(drop=True)
    holdout = _holdout_indices()
    if holdout:
        df = df.drop(index=[i for i in holdout if i < len(df)], errors="ignore")
    base = df[REQUIRED_COLUMNS].copy()
    acc = storage.get_accumulated()
    if acc:
        base = pd.concat([base, pd.DataFrame(acc)[REQUIRED_COLUMNS]], ignore_index=True)
    return base


def _load_validation_frame():
    """The fixed validation set. Falls back to None if the file is absent."""
    import os
    path = os.path.join(os.path.dirname(__file__), VALIDATION_FILE)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)[REQUIRED_COLUMNS].copy()


def _pending_frame(rows=None) -> pd.DataFrame:
    rows = storage.get_pending() if rows is None else (rows or [])
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.DataFrame([normalise_row(r) for r in rows])[REQUIRED_COLUMNS].copy()


def _frame_to_arrays(df: pd.DataFrame):
    income = df["person_income"].values.astype(float)
    credit = df["credit_score"].values.astype(float)
    ratio = df["loan_percent_income"].values.astype(float)
    emp = df["person_emp_exp"].values.astype(float)
    default = df["previous_loan_defaults_on_file"].map({"No": 0, "Yes": 1}).values.astype(float)
    status = df["loan_status"].values.astype(float)
    return income, credit, ratio, emp, default, status


def _mse_for_params(params: dict, arrays) -> float:
    income, credit, ratio, emp, default, status = arrays
    mfs = fe.build_mfs_from_params(params)
    pred = fe.fuzzy_risk_score_vec(income, ratio, emp, default, credit, mfs)
    return float(np.mean((status - pred) ** 2))


def _recent_holdout_arrays():
    """Arrays for the recent drift-aware holdout, or None if it doesn't yet hold
    enough rows (cold-start -> gate falls back to anchor-only)."""
    recent = storage.get_recent_holdout()
    if len(recent) < MIN_RECENT_HOLDOUT:
        return None, len(recent)
    df = pd.DataFrame(recent)[REQUIRED_COLUMNS].copy()
    return _frame_to_arrays(df), len(recent)


def _assert_no_holdout_leakage():
    """Leakage guard (record identity, by row_id). The recent holdout must never
    overlap the accumulated training base. Fail loudly rather than silently train
    on validation data."""
    acc_ids = {r.get("__rid") for r in storage.get_accumulated()
               if isinstance(r, dict) and r.get("__rid")}
    rec_ids = {r.get("__rid") for r in storage.get_recent_holdout()
               if isinstance(r, dict) and r.get("__rid")}
    overlap = acc_ids & rec_ids
    if overlap:
        raise RuntimeError(
            f"Holdout leakage: {len(overlap)} row_id(s) are in BOTH the accumulated "
            f"training base and the recent holdout. Aborting retrain to avoid "
            f"training on validation data.")


# ----------------------------------------------------------------
# Tool 3 — retrain_fuzzy_engine (self-contained GA, engine MF math)
# ----------------------------------------------------------------
def _smart_chrom():
    """Same smart initialisation style as fuzzy_engine.run_ga().

    Each numeric variable contributes three normalised boundary genes:
    early boundary, middle boundary, and upper boundary. This keeps the
    membership functions meaningful from generation 0 instead of starting
    from fully random breakpoints.
    """
    return [
        random.uniform(0.10, 0.35), random.uniform(0.35, 0.65), random.uniform(0.65, 0.90),
        random.uniform(0.10, 0.35), random.uniform(0.35, 0.65), random.uniform(0.65, 0.90),
        random.uniform(0.10, 0.35), random.uniform(0.35, 0.65), random.uniform(0.65, 0.90),
    ]


def _shape_penalty(params: dict) -> float:
    """Same MF-shape penalty concept as fuzzy_engine.fitness()."""
    penalty = 0.0
    for var_name, breakpoints in params.items():
        a, b, c = breakpoints
        span = 1.0 if var_name == "rat" else (100000 if var_name == "inc" else 30)

        width1 = b - a
        width2 = c - b
        total_width = c - a

        if width1 < 0.08 * span:
            penalty += (0.08 * span - width1) ** 2
        if width2 < 0.08 * span:
            penalty += (0.08 * span - width2) ** 2
        if total_width < 0.15 * span:
            penalty += (0.15 * span - total_width) ** 2 * 2
    return float(penalty)


def _fitness_on(arrays, chrom) -> float:
    """GA training fitness aligned with fuzzy_engine.fitness().

    Promotion still uses pure validation MSE in evaluate_and_promote(); this
    fitness value is only for GA search and convergence history.
    """
    income, credit, ratio, emp, default, status = arrays
    params = fe.decode_chromosome(chrom)
    mfs = fe.build_mfs_from_params(params)
    pred = fe.fuzzy_risk_score_vec(income, ratio, emp, default, credit, mfs)
    squared_error = float(np.sum((status - pred) ** 2))
    penalty_weight = 5000
    return float(squared_error + penalty_weight * _shape_penalty(params))


def _build_warm_start_chrom():
    """Phase 5 — encode the current active model into a GA chromosome for
    warm-starting, with a round-trip check. Returns (chrom_or_None, info)."""
    active = storage.get_active_params()
    active_params = active["params"] if active and "params" in active else _current_engine_params()
    chrom = fe.encode_params(active_params)
    rt_back = fe.decode_chromosome(chrom)
    max_err = 0.0
    for k in ("inc", "rat", "emp"):
        for a, b in zip(rt_back[k], active_params[k]):
            # combined absolute + relative tolerance (scales across inc/rat/emp).
            err = abs(a - b)
            max_err = max(max_err, err)
            if err > 1e-3 + 1e-4 * abs(b):
                return None, {"warm_started": False, "round_trip_ok": False,
                              "round_trip_max_err": round(float(max_err), 6)}
    return chrom, {"warm_started": True, "round_trip_ok": True,
                   "round_trip_max_err": round(float(max_err), 6)}


def _run_ga_once(train_arr, run_seed: int, warm_chrom=None):
    """One GA run: smart-init population (optionally warm-started with the active
    chromosome in generation 0), best-so-far elitism, adaptive mutation, two-point
    crossover. The active chromosome is NOT pinned — best-so-far elitism replaces
    it the moment something beats it. Returns a result dict."""
    random.seed(int(run_seed))
    np.random.seed(int(run_seed))
    pop = [_smart_chrom() for _ in range(GA_POP)]
    if warm_chrom is not None:
        pop[0] = list(warm_chrom)  # inject active model as a gen-0 individual
    scores = [_fitness_on(train_arr, c) for c in pop]
    best_chrom = pop[int(np.argmin(scores))][:]
    best_score = min(scores)
    init_score = best_score
    prev_avg = float(np.mean(scores))
    fit_history = [best_score]
    avg_history = [prev_avg]

    for _ in range(GA_GENS):
        new_pop = [best_chrom[:]]  # best-so-far elitism (not active-lock)
        while len(new_pop) < GA_POP:
            def tour():
                cand = random.sample(list(range(GA_POP)), min(GA_TOURNAMENT, GA_POP))
                return pop[min(cand, key=lambda i: scores[i])][:]

            p1, p2 = tour(), tour()
            if random.random() < GA_CR:
                pt1 = random.randint(1, fe.N_GENES - 2)
                pt2 = random.randint(pt1 + 1, fe.N_GENES - 1)
                child1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
                child2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
            else:
                child1, child2 = p1[:], p2[:]
            new_pop.extend([child1, child2])

        pop = new_pop[:GA_POP]
        scores = [_fitness_on(train_arr, c) for c in pop]
        new_avg = float(np.mean(scores))
        mr = GA_MR_BASE * 0.5 if new_avg < prev_avg else GA_MR_BASE * 2.0

        for i in range(1, GA_POP):  # adaptive mutation, skip elite at index 0
            ind = pop[i][:]
            mutated = False
            for j in range(fe.N_GENES):
                if random.random() < mr:
                    ind[j] = float(np.clip(ind[j] + np.random.uniform(-0.12, 0.12), 0.0, 1.0))
                    mutated = True
            if mutated:
                pop[i] = ind

        scores = [_fitness_on(train_arr, c) for c in pop]
        new_avg = float(np.mean(scores))
        gen_best = int(np.argmin(scores))
        if scores[gen_best] < best_score:
            best_score = scores[gen_best]
            best_chrom = pop[gen_best][:]
        fit_history.append(best_score)
        avg_history.append(new_avg)
        prev_avg = new_avg

    return {"best_chrom": best_chrom, "best_score": best_score, "init_score": init_score,
            "fit_history": fit_history, "avg_history": avg_history, "seed": int(run_seed)}


def retrain_fuzzy_engine(include_pending: bool = True, seed=None, pending_rows=None):
    # Phase 2 leakage guard: recent holdout must never overlap the training base.
    _assert_no_holdout_leakage()
    # Validation = the dedicated fixed holdout file (never trained on).
    # Training    = historical base (minus holdout) + new pending rows.
    base_df = _load_base_frame().reset_index(drop=True)
    pending_df = _pending_frame(pending_rows).reset_index(drop=True) if include_pending else pd.DataFrame(columns=REQUIRED_COLUMNS)
    val_df = _load_validation_frame()

    if val_df is not None and len(val_df):
        # base_df already excludes the holdout rows, so just append pending.
        train_df = pd.concat([base_df, pending_df], ignore_index=True)
    else:
        # Fallback (holdout file missing): deterministic seed-slice of the base.
        rng = np.random.default_rng(VALIDATION_SEED)
        base_idx = np.arange(len(base_df))
        rng.shuffle(base_idx)
        n_val = max(1, int(len(base_idx) * VALIDATION_FRACTION))
        val_idx = base_idx[:n_val]
        train_base_idx = base_idx[n_val:]
        train_df = pd.concat([base_df.iloc[train_base_idx], pending_df], ignore_index=True)
        val_df = base_df.iloc[val_idx].reset_index(drop=True)

    # Speed cap: sample the training rows the GA optimises against. The final
    # candidate is still scored on the full fixed validation holdout below.
    if TRAIN_SAMPLE_CAP and len(train_df) > TRAIN_SAMPLE_CAP:
        train_df = train_df.sample(n=TRAIN_SAMPLE_CAP, random_state=VALIDATION_SEED).reset_index(drop=True)

    train_arr = _frame_to_arrays(train_df)
    val_arr = _frame_to_arrays(val_df)

    # Phase 1 — per-run base GA seed (deterministic per data-state, distinct from
    # the boot/v0 seed so the candidate never re-derives v0's exact chromosome).
    if seed is not None:
        effective_seed = int(seed) % (2**31 - 1)
    else:
        effective_seed = (int(GA_SEED) * 2654435761
                          + len(train_df) * 40503
                          + len(pending_df)) % (2**31 - 1)
    if effective_seed == int(GA_SEED):
        effective_seed = (effective_seed + 1) % (2**31 - 1)

    # Phase 5 — warm-start from the active model + multi-seed restarts.
    #   * warm-start injects the active chromosome into generation 0 of EVERY
    #     restart, so (with best-so-far elitism) each restart's best is no worse
    #     than the active model on TRAINING fitness.
    #   * run GA_RESTARTS seeds, select the candidate with the best TRAINING
    #     fitness (NOT the fixed validation MSE — that stays purely for the gate,
    #     avoiding validation selection bias).
    warm_chrom, warm_info = _build_warm_start_chrom()
    restart_seeds = [(effective_seed + r * 2246822519) % (2**31 - 1) for r in range(max(1, GA_RESTARTS))]
    runs = [_run_ga_once(train_arr, s, warm_chrom=warm_chrom) for s in restart_seeds]
    winner = min(runs, key=lambda r: r["best_score"])  # select on training fitness

    best_chrom = winner["best_chrom"]
    best_score = winner["best_score"]
    init_score = winner["init_score"]
    fit_history = winner["fit_history"]
    avg_history = winner["avg_history"]
    selected_seed = winner["seed"]

    candidate_params = fe.decode_chromosome(best_chrom)
    improvement_pct = round((init_score - best_score) / init_score * 100, 1) if init_score > 0 else 0.0
    train_status = train_arr[5]
    return {
        "params": candidate_params,
        "effective_seed": int(effective_seed),
        "selected_seed": int(selected_seed),
        "warm_started": bool(warm_info["warm_started"]),
        "restarts": int(len(restart_seeds)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "pending_rows": int(len(pending_df)),
        "n_total": int(len(base_df) + len(pending_df)),
        "val_arrays": val_arr,
        "ga": {
            "init_fitness": round(float(init_score), 5),
            "final_fitness": round(float(best_score), 5),
            "improvement_pct": improvement_pct,
            "fit_history": [round(float(x), 5) for x in fit_history],
            "avg_history": [round(float(x), 5) for x in avg_history],
            "generations": int(GA_GENS),
            "population": int(GA_POP),
            "tournament_size": int(GA_TOURNAMENT),
            "crossover_rate": float(GA_CR),
            "mutation_base": float(GA_MR_BASE),
            "mutation_strategy": "adaptive",
            "crossover_type": "two-point",
            "initialization": "smart+warm" if warm_info["warm_started"] else "smart",
            "warm_started": bool(warm_info["warm_started"]),
            "warm_round_trip_ok": bool(warm_info["round_trip_ok"]),
            "warm_round_trip_max_err": warm_info["round_trip_max_err"],
            "restarts": int(len(restart_seeds)),
            "restart_seeds": [int(s) for s in restart_seeds],
            "restart_fitness": [round(float(r["best_score"]), 5) for r in runs],
            "selected_seed": int(selected_seed),
            "seed": int(selected_seed),
            "base_seed": int(effective_seed),
            "train_rows": int(len(train_df)),
            "approved": int((train_status == 0).sum()),
            "rejected": int((train_status == 1).sum()),
            "metric": "MSE",
            "fitness_objective": "squared_error_plus_shape_penalty",
        },
    }


# ----------------------------------------------------------------
# Tool 4 — evaluate_and_promote (blended anchor + recent validation gate)
# ----------------------------------------------------------------
def evaluate_and_promote(candidate):
    # Both models are scored on the EXACT same held-out arrays.
    #   anchor = candidate["val_arrays"] = the fixed original-distribution holdout
    #            (stable cross-version ruler; protects old-distribution accuracy).
    #   recent = the drift-aware holdout (checks adaptation to new data).
    anchor = candidate["val_arrays"]
    new_params = candidate["params"]
    active = storage.get_active_params()
    old_params = active["params"] if active and "params" in active else _current_engine_params()

    anchor_old = _mse_for_params(old_params, anchor)
    anchor_new = _mse_for_params(new_params, anchor)

    recent_arr, recent_n = _recent_holdout_arrays()
    if recent_arr is not None:
        gate_mode = "blended"
        recent_old = _mse_for_params(old_params, recent_arr)
        recent_new = _mse_for_params(new_params, recent_arr)
        blended_old = ANCHOR_WEIGHT * anchor_old + RECENT_WEIGHT * recent_old
        blended_new = ANCHOR_WEIGHT * anchor_new + RECENT_WEIGHT * recent_new
    else:
        # Cold-start: not enough recent rows yet -> anchor-only gate.
        gate_mode = "anchor_only"
        recent_old = recent_new = None
        blended_old, blended_new = anchor_old, anchor_new

    # Never let a non-finite MSE poison the gate (e.g. empty array -> NaN).
    for v in (anchor_old, anchor_new, blended_old, blended_new):
        if not math.isfinite(v):
            raise RuntimeError("Non-finite MSE in promotion gate; aborting retrain.")

    # Promote only if the candidate beats the active model on the blended gate by
    # the margin AND does not regress the anchor (old distribution) beyond the
    # allowed tolerance — adapt to drift without catastrophic forgetting.
    beat_margin = blended_new < blended_old - PROMOTION_MARGIN
    anchor_guard_pass = anchor_new <= anchor_old * MAX_ANCHOR_REGRESSION
    improved = bool(beat_margin and anchor_guard_pass)

    if improved:
        skip_reason = ""
    elif beat_margin and not anchor_guard_pass:
        skip_reason = (f"Candidate improved the blended drift-aware gate but exceeded the "
                       f"allowed anchor regression ({MAX_ANCHOR_REGRESSION:.2f}x); kept to "
                       f"prevent catastrophic forgetting on the original distribution.")
    else:
        skip_reason = "Candidate did not beat the active model on the blended validation gate by the margin."

    return {
        "improved": improved,
        "gate_mode": gate_mode,
        # ruler separation: new_mse / old_mse stay the ANCHOR mse (the stable,
        # cross-version-comparable ruler). The blended numbers are the
        # promotion DECISION values only — not comparable across versions.
        "new_mse": round(anchor_new, 5),
        "old_mse": round(anchor_old, 5),
        "anchor_old_mse": round(anchor_old, 5),
        "anchor_new_mse": round(anchor_new, 5),
        "recent_old_mse": (round(recent_old, 5) if recent_old is not None else None),
        "recent_new_mse": (round(recent_new, 5) if recent_new is not None else None),
        "blended_old_mse": round(blended_old, 5),
        "blended_new_mse": round(blended_new, 5),
        "recent_holdout_size": int(recent_n),
        "anchor_guard_pass": bool(anchor_guard_pass),
        "anchor_regression": (round(anchor_new / anchor_old, 4) if anchor_old > 0 else None),
        "promotion_margin": PROMOTION_MARGIN,
        "skip_reason": skip_reason,
        "new_params": new_params,
        "effective_seed": candidate.get("effective_seed"),
        "train_rows": candidate["train_rows"],
        "validation_rows": candidate["validation_rows"],
        "pending_rows": candidate["pending_rows"],
        "n_total": candidate["n_total"],
        "ga": candidate.get("ga")}


def _current_engine_params():
    """Params the live engine is currently using (from GA at boot)."""
    return {"inc": list(fe.best_params["inc"]),
            "rat": list(fe.best_params["rat"]),
            "emp": list(fe.best_params["emp"])}


# Baseline model produced by fuzzy_engine.py's boot GA, now trained on the
# HOLDOUT-EXCLUDED base (the fixed validation rows are dropped before the GA
# runs). Captured once at import; this is the model the v0 baseline adopts.
BOOT_BASELINE_PARAMS = _current_engine_params()


def build_baseline_if_missing(apply_live: bool = True):
    """Build the v0 baseline by ADOPTING the boot engine's model.

    fuzzy_engine.py now runs its GA on the HOLDOUT-EXCLUDED base (the fixed
    validation rows are dropped before training), so the boot model has never
    seen the validation set. The v0 baseline simply adopts that exact model.
    This guarantees that the served baseline, the cold-start cache
    (boot_ga.json) and the GA Performance page all describe ONE identical
    model — no second, divergent GA run. Candidates (v1+) are still scored
    against this v0 on the same fixed holdout, so promotion gating stays fair.
    Idempotent: runs only when no active model exists.
    """
    if storage.get_active_params():
        return None

    # Adopt the holdout-excluded boot model as the v0 baseline.
    params = _current_engine_params()  # == fe.best_params

    # Score it on the fixed validation holdout (the same ruler used for gating).
    val_df = _load_validation_frame()
    if val_df is not None and len(val_df):
        val_arr = _frame_to_arrays(val_df)
        val_rows = int(len(val_df))
    else:
        base_df = _load_base_frame().reset_index(drop=True)
        rng = np.random.default_rng(VALIDATION_SEED)
        idx = np.arange(len(base_df)); rng.shuffle(idx)
        n_val = max(1, int(len(base_df) * VALIDATION_FRACTION))
        val_arr = _frame_to_arrays(base_df.iloc[idx[:n_val]].reset_index(drop=True))
        val_rows = n_val
    mse = round(_mse_for_params(params, val_arr), 5)

    # Reuse the boot GA's convergence history so the GA Performance page and the
    # slide reflect the model that is actually served.
    boot_ga = getattr(fe, "ga_result", {}) or {}
    ref_status = getattr(fe, "REF_STATUS", None)
    train_rows = int(len(ref_status)) if ref_status is not None else None
    ga_block = {
        "init_fitness": round(float(boot_ga.get("init_fitness", 0.0)), 5),
        "final_fitness": round(float(boot_ga.get("best_fitness", 0.0)), 5),
        "improvement_pct": round(float(boot_ga.get("improvement_pct", 0.0)), 1),
        "fit_history": [round(float(x), 5) for x in boot_ga.get("fit_history", [])],
        "avg_history": [round(float(x), 5) for x in boot_ga.get("avg_history", [])],
        "generations": int(boot_ga.get("generations", 0)),
        "population": int(boot_ga.get("pop_size", 0)),
        "tournament_size": int(getattr(fe, "GA_TOURNAMENT", GA_TOURNAMENT)),
        "crossover_rate": float(boot_ga.get("crossover_rate", GA_CR)),
        "mutation_base": float(boot_ga.get("mutation_base", GA_MR_BASE)),
        "mutation_strategy": "adaptive",
        "crossover_type": "two-point",
        "initialization": "smart",
        "seed": int(getattr(fe, "GA_SEED", GA_SEED)),
        "train_rows": train_rows,
        "approved": int((ref_status == 0).sum()) if ref_status is not None else None,
        "rejected": int((ref_status == 1).sum()) if ref_status is not None else None,
        "metric": "MSE",
    }

    record = {
        "version": 0,
        "params": params,
        "new_mse": mse,
        "old_mse": None,
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "pending_rows": 0,
        "total_rows": (train_rows + val_rows) if train_rows is not None else None,
        "created": storage.now_iso(),
        "source": "Holdout-excluded GA baseline (boot engine)",
        "status": "baseline",
        "ga": ga_block,
    }
    changelog = (
        f"Model v0 — {storage.now_iso()}\n"
        f"  Source        : holdout-excluded GA baseline (boot engine, adopted)\n"
        f"  Training rows : {train_rows} (base minus fixed holdout)\n"
        f"  Validation    : {val_rows} fixed holdout rows\n"
        f"  Validation MSE: {mse}\n"
        f"  Decision      : BASELINE (fair comparison anchor; never saw the holdout)\n"
        f"  Income MF     : {params['inc']}\n"
        f"  Ratio  MF     : {params['rat']}\n"
        f"  Emp    MF     : {params['emp']}\n"
    )
    storage.save_version(0, record, changelog, _make_version_zip(0, record, changelog))
    storage.set_active_params(record)
    storage.append_changelog(f"[{storage.now_iso()}] v0 BASELINE built (holdout-excluded boot engine) mse={mse}")
    if apply_live and hasattr(fe, "apply_model_params"):
        try:
            fe.apply_model_params(params, source="v0 holdout-excluded baseline")
        except Exception as exc:
            print(f"[retrain] baseline activation failed: {exc}", flush=True)
    return record


def _ga_block_for_params(params: dict) -> dict:
    """Compute a correct MSE-based `ga` summary for the given params using the
    REAL holdout-excluded training data (base minus fixed holdout + pending) —
    NOT the boot-time full 45k dataset.

    Used to heal active model records that were stored without a usable `ga`
    block (older deploys, or a synthesized v0 rollback). Without this, the GA
    page silently fell back to the boot engine and showed the misleading
    45,000-row / large-error numbers instead of the served model's MSE.
    """
    base_df = _load_base_frame().reset_index(drop=True)
    pending_df = _pending_frame().reset_index(drop=True)
    val_df = _load_validation_frame()
    if val_df is not None and len(val_df):
        train_df = pd.concat([base_df, pending_df], ignore_index=True)
    else:
        rng = np.random.default_rng(VALIDATION_SEED)
        idx = np.arange(len(base_df))
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * VALIDATION_FRACTION))
        val_df = base_df.iloc[idx[:n_val]].reset_index(drop=True)
        train_df = pd.concat([base_df.iloc[idx[n_val:]], pending_df], ignore_index=True)

    train_arr = _frame_to_arrays(train_df)
    val_arr = _frame_to_arrays(val_df)
    mse = round(float(_mse_for_params(params, val_arr)), 5)
    status = train_arr[5]
    return {
        "init_fitness": mse,
        "final_fitness": mse,
        "improvement_pct": 0.0,
        "fit_history": [mse],   # flat: no evolution history available for an already-served model
        "avg_history": [mse],
        "generations": int(GA_GENS),
        "population": int(GA_POP),
        "tournament_size": int(GA_TOURNAMENT),
        "crossover_rate": float(GA_CR),
        "mutation_base": float(GA_MR_BASE),
        "mutation_strategy": "adaptive",
        "crossover_type": "two-point",
        "initialization": "smart",
        "seed": int(GA_SEED),
        "train_rows": int(len(train_df)),
        "approved": int((status == 0).sum()),
        "rejected": int((status == 1).sum()),
        "metric": "MSE",
    }


def _active_ga_is_healthy(active: dict | None) -> bool:
    ga = (active or {}).get("ga") or {}
    return bool(ga.get("fit_history"))


def repair_active_ga(allow_rebuild: bool = False):
    """Self-heal the active model record so it always carries a valid MSE-based
    `ga` block. Returns the (possibly updated) active record.

    - Healthy record (has ga.fit_history) -> untouched.
    - Stale v0 / baseline with allow_rebuild=True -> rebuild a fair holdout-
      excluded baseline (full GA, real convergence curve) and re-store it.
    - Any other record missing `ga` -> backfill a correct flat MSE summary from
      its stored params (cheap; safe to run inside a web request).

    This is what stops the GA page from reverting to the boot 45,000-row engine.
    """
    active = storage.get_active_params()
    if not active or "params" not in active:
        return active
    if _active_ga_is_healthy(active):
        return active

    version = active.get("version")
    is_baseline = (version in (0, None)) or str(active.get("status", "")).lower() == "baseline"

    try:
        if allow_rebuild and is_baseline:
            # Rebuild a real v0 baseline on holdout-excluded base data only.
            # Pending rows belong to candidate retraining, not baseline reconstruction.
            candidate = retrain_fuzzy_engine(include_pending=False)
            params = candidate["params"]
            mse = round(float(_mse_for_params(params, candidate["val_arrays"])), 5)
            record = {
                "version": 0,
                "params": params,
                "new_mse": mse,
                "old_mse": None,
                "train_rows": candidate["train_rows"],
                "validation_rows": candidate["validation_rows"],
                "pending_rows": 0,
                "total_rows": candidate["n_total"],
                "created": storage.now_iso(),
                "source": "Holdout-excluded GA baseline",
                "status": "baseline",
                "ga": candidate.get("ga"),
            }
            storage.save_version(0, record, f"[{storage.now_iso()}] v0 baseline rebuilt from holdout-excluded data", _make_version_zip(0, record, "v0 baseline"))
            storage.set_active_params(record)
            storage.append_changelog(f"[{storage.now_iso()}] v0 BASELINE rebuilt (mse={mse})")
            if hasattr(fe, "apply_model_params"):
                try:
                    fe.apply_model_params(params, source="v0 holdout-excluded baseline")
                except Exception as exc:
                    print(f"[retrain] healed baseline activation failed: {exc}", flush=True)
            return record

        # Cheap backfill for promoted records (or when rebuild isn't allowed).
        active["ga"] = _ga_block_for_params(active["params"])
        active.setdefault("train_rows", active["ga"]["train_rows"])
        active.setdefault("total_rows", active["ga"]["train_rows"])
        storage.set_active_params(active)
        storage.append_changelog(f"[{storage.now_iso()}] HEALED active model: backfilled ga block (was missing fit_history)")
    except Exception as exc:
        print(f"[retrain] repair_active_ga failed: {exc}", flush=True)
    return active


def version0_record() -> dict:
    """v0 baseline shown in the admin console. Prefer the stored holdout-excluded
    baseline; fall back to the boot full-data model only if it hasn't been built."""
    stored = storage.read_version_params(0)
    if stored:
        return stored
    return {
        "version": 0,
        "params": BOOT_BASELINE_PARAMS,
        "created": "Boot-time",
        "source": "Boot-time GA baseline (full data)",
        "new_mse": None,
        "old_mse": None,
        "train_rows": int(len(fe.REF_STATUS)) if hasattr(fe, "REF_STATUS") else 0,
        "validation_rows": None,
        "pending_rows": 0,
        "total_rows": int(len(fe.REF_STATUS)) if hasattr(fe, "REF_STATUS") else 0,
        "status": "baseline",
    }


def version0_zip() -> bytes:
    stored = storage.read_version_zip(0)
    if stored:
        return stored
    changelog = (
        "Model v0 — Boot-time GA baseline (full data)\n"
        "  Source       : fuzzy_engine.py initial GA result\n"
        "  Decision     : BASELINE (rollback target)\n"
        f"  Income MF    : {BOOT_BASELINE_PARAMS['inc']}\n"
        f"  Ratio  MF    : {BOOT_BASELINE_PARAMS['rat']}\n"
        f"  Emp    MF    : {BOOT_BASELINE_PARAMS['emp']}\n"
    )
    return _make_version_zip(0, version0_record(), changelog)


def _current_run_id() -> str:
    status = storage.get_status()
    return str(status.get("job_id") or f"run_{storage.now_iso().replace(' ', '_').replace(':', '')}")


def _make_version_zip(version: int, params: dict, changelog: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mf_params.json", json.dumps(params, ensure_ascii=False, indent=2))
        zf.writestr("changelog.txt", changelog)
    return buf.getvalue()


def promote(evaluation):
    version = storage.next_version_number()
    params = evaluation["new_params"]
    record = {
        "version": version,
        "params": params,
        "new_mse": evaluation["new_mse"],
        "old_mse": evaluation["old_mse"],
        "gate_mode": evaluation.get("gate_mode"),
        "anchor_old_mse": evaluation.get("anchor_old_mse"),
        "anchor_new_mse": evaluation.get("anchor_new_mse"),
        "recent_old_mse": evaluation.get("recent_old_mse"),
        "recent_new_mse": evaluation.get("recent_new_mse"),
        "blended_old_mse": evaluation.get("blended_old_mse"),
        "blended_new_mse": evaluation.get("blended_new_mse"),
        "recent_holdout_size": evaluation.get("recent_holdout_size"),
        "anchor_guard_pass": evaluation.get("anchor_guard_pass"),
        "anchor_regression": evaluation.get("anchor_regression"),
        "effective_seed": evaluation.get("effective_seed"),
        "train_rows": evaluation["train_rows"],
        "validation_rows": evaluation.get("validation_rows"),
        "pending_rows": evaluation.get("pending_rows"),
        "total_rows": evaluation["n_total"],
        "created": storage.now_iso(),
        "ga": evaluation.get("ga"),
    }
    _gm = evaluation.get("gate_mode")
    changelog = (
        f"Model v{version} — {storage.now_iso()}\n"
        f"  Trigger      : retraining agent\n"
        f"  Training rows: {evaluation['train_rows']} (pending {evaluation.get('pending_rows')}, total {evaluation['n_total']})\n"
        f"  Validation  : {evaluation.get('validation_rows')} fixed anchor rows"
        f" + {evaluation.get('recent_holdout_size')} recent holdout rows\n"
        f"  Gate mode    : {_gm}\n"
        f"  Anchor MSE   : old {evaluation.get('anchor_old_mse')} -> new {evaluation.get('anchor_new_mse')} (stable ruler)\n"
        f"  Recent MSE   : old {evaluation.get('recent_old_mse')} -> new {evaluation.get('recent_new_mse')}\n"
        f"  Blended MSE  : old {evaluation.get('blended_old_mse')} -> new {evaluation.get('blended_new_mse')} (decision value)\n"
        f"  Decision     : PROMOTED (beat blended gate; anchor regression within {MAX_ANCHOR_REGRESSION:.2f}x)\n"
        f"  Income MF    : {params['inc']}\n"
        f"  Ratio  MF    : {params['rat']}\n"
        f"  Emp    MF    : {params['emp']}\n"
    )
    zip_bytes = _make_version_zip(version, record, changelog)
    storage.save_version(version, record, changelog, zip_bytes)
    storage.set_active_params(record)
    storage.append_changelog(
        f"[{storage.now_iso()}] v{version} PROMOTED  old_mse={evaluation['old_mse']} "
        f"new_mse={evaluation['new_mse']} rows={evaluation['n_total']}")
    run_id = _current_run_id()
    storage.add_run({
        "run_id": run_id,
        "time": storage.now_iso(),
        "status": "promoted",
        "version": version,
        "old_mse": evaluation["old_mse"],
        "new_mse": evaluation["new_mse"],
        "gate_mode": evaluation.get("gate_mode"),
        "blended_old_mse": evaluation.get("blended_old_mse"),
        "blended_new_mse": evaluation.get("blended_new_mse"),
        "recent_old_mse": evaluation.get("recent_old_mse"),
        "recent_new_mse": evaluation.get("recent_new_mse"),
        "recent_holdout_size": evaluation.get("recent_holdout_size"),
        "anchor_guard_pass": evaluation.get("anchor_guard_pass"),
        "effective_seed": evaluation.get("effective_seed"),
        "train_rows": evaluation["train_rows"],
        "validation_rows": evaluation.get("validation_rows"),
        "pending_rows": evaluation.get("pending_rows"),
        "new_rows_used": evaluation.get("_batch_row_count", evaluation.get("pending_rows")),
        "reason": "Candidate improved blended validation MSE while anchor regression stayed within tolerance.",
    })
    evaluation_with_version = dict(evaluation)
    evaluation_with_version["version"] = version
    storage.consume_pending_batch(run_id, "promoted", evaluation_with_version, f"Promoted v{version}; locked batch consumed; newer uploads remain queued.", rows=evaluation.get("_batch_rows"))

    # Activate the promoted model on this instance immediately, so promotion is
    # self-contained no matter how run_retraining_agent() is invoked. Other
    # Cloud Run instances pick it up from storage on their next restart.
    if hasattr(fe, "apply_model_params"):
        try:
            fe.apply_model_params(params, source=f"promoted artifact v{version}")
        except Exception as exc:  # never let activation failure break the promotion record
            print(f"[retrain] live activation failed for v{version}: {exc}", flush=True)

    return record


def rollback(version: int):
    version = int(version)
    if version == 0:
        rec = version0_record()
        rec["created"] = storage.now_iso()
        storage.set_active_params(rec)
        storage.append_changelog(f"[{storage.now_iso()}] ROLLBACK to v0 baseline")
        storage.add_run({
            "run_id": f"rollback_{storage.now_iso().replace(' ', '_').replace(':', '')}",
            "time": storage.now_iso(),
            "status": "rollback",
            "version": 0,
            "old_mse": None,
            "new_mse": None,
            "train_rows": None,
            "validation_rows": None,
            "pending_rows": None,
            "new_rows_used": None,
            "reason": "Admin rolled back to v0 baseline.",
        })
        return rec

    rec = storage.read_version_params(version)
    if not rec:
        return None
    storage.set_active_params(rec)
    storage.append_changelog(f"[{storage.now_iso()}] ROLLBACK to v{version}")
    storage.add_run({
        "run_id": f"rollback_{storage.now_iso().replace(' ', '_').replace(':', '')}",
        "time": storage.now_iso(),
        "status": "rollback",
        "version": version,
        "old_mse": rec.get("old_mse"),
        "new_mse": rec.get("new_mse"),
        "train_rows": rec.get("train_rows"),
        "validation_rows": rec.get("validation_rows"),
        "pending_rows": rec.get("pending_rows"),
        "new_rows_used": None,
        "reason": f"Admin rolled back to v{version}.",
    })
    return rec


# ----------------------------------------------------------------
# The Retraining Agent loop (observe -> reason -> tools -> explain)
# ----------------------------------------------------------------
def run_retraining_agent(force: bool = False, owner_token: str = None):
    """Returns a trace of the agent's decisions. Designed to run async."""
    trace = []

    def step(name, detail):
        trace.append({"step": name, "detail": detail})

    fresh = check_data_freshness()
    step("observe", f"{fresh['new_rows']} new rows pending (threshold {fresh['threshold']}).")

    if not fresh["ready"] and not force:
        step("reason", "Not enough new data — skip retraining.")
        storage.set_status("idle", f"Waiting for data ({fresh['new_rows']}/{fresh['threshold']}).")
        return {"action": "skip", "reason": "threshold_not_met", "trace": trace, "freshness": fresh}

    step("reason", "Threshold met — proceeding to validate and retrain." if fresh["ready"]
         else "Manual force — proceeding despite threshold.")

    quality = read_and_validate_data()
    step("validate", f"Quality check: {quality['clean']} clean, {quality['sent_to_review']} sent to manual review.")

    post_quality_fresh = check_data_freshness()
    if not force and not post_quality_fresh["ready"]:
        step("reason", "Not enough clean rows left after validation — abort.")
        storage.add_run({
            "run_id": _current_run_id(),
            "time": storage.now_iso(),
            "status": "skipped",
            "version": None,
            "old_mse": None,
            "new_mse": None,
            "train_rows": post_quality_fresh.get("new_rows", 0),
            "validation_rows": None,
            "pending_rows": post_quality_fresh.get("new_rows", 0),
            "new_rows_used": 0,
            "reason": "Clean rows fell below the retraining threshold after Gate 2 validation; rows remain pending until more clean data arrives.",
        })
        storage.set_status("idle", f"Waiting for data ({post_quality_fresh['new_rows']}/{post_quality_fresh['threshold']}).")
        return {"action": "skip", "reason": "threshold_not_met_after_validation", "trace": trace, "freshness": post_quality_fresh}

    run_id = _current_run_id()
    batch_rows = storage.begin_pending_batch(run_id)
    if not batch_rows:
        step("reason", "No clean rows were available to lock for this run — abort.")
        storage.set_status("idle", "No queued clean rows available for retraining.")
        return {"action": "skip", "reason": "empty_batch", "trace": trace, "freshness": check_data_freshness()}

    storage.set_status(
        "running",
        "Retraining in progress…",
        extra={"job_id": run_id, "batch_rows": len(batch_rows), "queued_clean_rows": len(storage.get_pending())},
    )
    step("call_tool", f"retrain_fuzzy_engine() — running GA on locked batch ({len(batch_rows)} rows) + base data.")
    candidate = retrain_fuzzy_engine(pending_rows=batch_rows)

    step("call_tool", "evaluate_and_promote() — blended (anchor + recent) validation gate.")
    evaluation = evaluate_and_promote(candidate)
    evaluation["_batch_rows"] = batch_rows
    evaluation["_batch_row_count"] = len(batch_rows)
    step("explain",
         f"Gate [{evaluation['gate_mode']}]: anchor old={evaluation['anchor_old_mse']} new={evaluation['anchor_new_mse']} | "
         f"recent old={evaluation['recent_old_mse']} new={evaluation['recent_new_mse']} | "
         f"blended old={evaluation['blended_old_mse']} new={evaluation['blended_new_mse']} "
         f"({'improved' if evaluation['improved'] else 'not improved'}).")

    # Phase 4 — final-commit ownership check. If this run was declared stale and a
    # newer run took over the lease, abort BEFORE mutating any state (no promote,
    # no pending consume) so the taking-over run owns the batch cleanly.
    if owner_token is not None and not storage.still_own_retrain_lock(owner_token):
        step("act", "Superseded by a newer retrain (lock lost) — restoring locked batch to pending.")
        storage.restore_current_batch(run_id, reason="lock lost before commit")
        storage.set_status("idle", "Retrain superseded by a newer run; locked batch restored.")
        return {"action": "superseded", "reason": "lock_lost", "evaluation": evaluation, "trace": trace}

    if evaluation["improved"]:
        record = promote(evaluation)
        step("act", f"Promoted model v{record['version']}.")
        storage.set_status("done",
                           f"Promoted v{record['version']} (blended MSE {evaluation['blended_new_mse']}; anchor MSE {evaluation['new_mse']}).",
                           extra={"version": record["version"]})
        return {"action": "promote", "version": record["version"], "evaluation": evaluation, "trace": trace}

    storage.append_changelog(
        f"[{storage.now_iso()}] SKIP [{evaluation['gate_mode']}] "
        f"blended old={evaluation['blended_old_mse']} new={evaluation['blended_new_mse']} "
        f"anchor_guard_pass={evaluation['anchor_guard_pass']}")
    run_id = _current_run_id()
    storage.add_run({
        "run_id": run_id,
        "time": storage.now_iso(),
        "status": "skipped",
        "version": None,
        "old_mse": evaluation["old_mse"],
        "new_mse": evaluation["new_mse"],
        "gate_mode": evaluation.get("gate_mode"),
        "blended_old_mse": evaluation.get("blended_old_mse"),
        "blended_new_mse": evaluation.get("blended_new_mse"),
        "recent_old_mse": evaluation.get("recent_old_mse"),
        "recent_new_mse": evaluation.get("recent_new_mse"),
        "recent_holdout_size": evaluation.get("recent_holdout_size"),
        "anchor_guard_pass": evaluation.get("anchor_guard_pass"),
        "anchor_regression": evaluation.get("anchor_regression"),
        "effective_seed": evaluation.get("effective_seed"),
        "train_rows": evaluation.get("train_rows"),
        "validation_rows": evaluation.get("validation_rows"),
        "pending_rows": evaluation.get("pending_rows"),
        "new_rows_used": evaluation.get("_batch_row_count", evaluation.get("pending_rows")),
        "reason": evaluation.get("skip_reason") or "Candidate did not beat the active model on the blended validation gate. Clean rows archived as attempted.",
    })
    storage.consume_pending_batch(run_id, "skipped", evaluation, "Candidate skipped; locked batch consumed; newer uploads remain queued.", rows=evaluation.get("_batch_rows"))
    step("act", "Kept current model — candidate did not pass the blended gate.")
    storage.set_status("done", "Retrained — no improvement, current model kept.")
    return {"action": "skip", "reason": "no_improvement", "evaluation": evaluation, "trace": trace}


# ----------------------------------------------------------------
# Active-params accessor for the live engine (hot-load without edit)
# ----------------------------------------------------------------
def active_params_or_default():
    active = storage.get_active_params()
    if active and "params" in active:
        return active["params"], active.get("version")
    return _current_engine_params(), 0
