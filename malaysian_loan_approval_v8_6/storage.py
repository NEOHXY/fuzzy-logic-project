# ================================================================
# storage.py — persistence layer for the continuous-learning loop.
# Uses Google Cloud Storage when GCS_BUCKET is set (production on
# Cloud Run, whose container FS is ephemeral); otherwise falls back
# to a local ./_store directory (local dev / demo).
#
# Layout (keys are the same in GCS and local):
#   pending.json                 list of candidate rows (passed Gate 1)
#   review.json                  list of rows sent to manual review
#   status.json                  retraining lock + last run info
#   active/mf_params.json        currently-promoted model params
#   versions/v{n}/mf_params.json
#   versions/v{n}/changelog.txt
#   versions/v{n}/model_v{n}.zip
#   changelog.txt                appended global log
# ================================================================

import os
import io
import json
import uuid
import time
import hashlib
import threading
from datetime import datetime, timezone

_LOCK = threading.Lock()

# Phase 2 — recent (drift-aware) holdout pool.
# A deterministic ~20% of every consumed clean batch is split into this pool
# BEFORE the rest is folded into the training base, so the recent holdout is
# never trained on. Split is by row_id hash (record identity, not content).
RECENT_HOLDOUT_BUCKET = 5          # 1-in-5 -> ~20% of consumed rows to holdout
MAX_RECENT_HOLDOUT = 5000          # cap; oldest overflow is evicted INTO training
_BUCKET = os.environ.get("GCS_BUCKET", "").strip()
_LOCAL_ROOT = os.environ.get("STORE_DIR", os.path.join(os.path.dirname(__file__), "_store"))

_gcs_client = None
_gcs_bucket = None
if _BUCKET:
    try:
        from google.cloud import storage as gcs_storage  # type: ignore
        _gcs_client = gcs_storage.Client()
        _gcs_bucket = _gcs_client.bucket(_BUCKET)
    except Exception as exc:  # pragma: no cover - falls back to local
        print(f"[storage] GCS init failed ({exc}); using local store", flush=True)
        _gcs_client = None
        _gcs_bucket = None


def using_gcs() -> bool:
    return _gcs_bucket is not None


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ----------------------------------------------------------------
# Low-level blob IO
# ----------------------------------------------------------------
def _local_path(key: str) -> str:
    path = os.path.join(_LOCAL_ROOT, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def read_bytes(key: str):
    if using_gcs():
        blob = _gcs_bucket.blob(key)
        if not blob.exists():
            return None
        return blob.download_as_bytes()
    path = os.path.join(_LOCAL_ROOT, key)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return fh.read()


def write_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    if using_gcs():
        _gcs_bucket.blob(key).upload_from_string(data, content_type=content_type)
        return
    with open(_local_path(key), "wb") as fh:
        fh.write(data)


def read_text(key: str):
    data = read_bytes(key)
    return data.decode("utf-8") if data is not None else None


def write_text(key: str, text: str, content_type: str = "text/plain") -> None:
    write_bytes(key, text.encode("utf-8"), content_type=content_type)


def read_json(key: str, default=None):
    text = read_text(key)
    if text is None:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def write_json(key: str, obj) -> None:
    write_text(key, json.dumps(obj, ensure_ascii=False, indent=2), content_type="application/json")


def list_prefix(prefix: str):
    if using_gcs():
        return sorted({b.name for b in _gcs_client.list_blobs(_BUCKET, prefix=prefix)})
    root = os.path.join(_LOCAL_ROOT, prefix)
    out = []
    base = os.path.dirname(root) if not prefix.endswith("/") else root
    for dirpath, _dirs, files in os.walk(os.path.join(_LOCAL_ROOT, prefix.split("/")[0]) if "/" in prefix else _LOCAL_ROOT):
        for f in files:
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, _LOCAL_ROOT).replace(os.sep, "/")
            if rel.startswith(prefix):
                out.append(rel)
    return sorted(out)


# ----------------------------------------------------------------
# Domain helpers
# ----------------------------------------------------------------
def get_pending():
    return read_json("pending.json", default=[]) or []


def set_pending(rows) -> None:
    write_json("pending.json", rows)


def _ensure_rid(rows):
    """Stamp a unique synthetic row_id on any row missing one.

    Membership (training vs recent holdout) is tracked by this id, NOT by row
    content: the dataset has only a few coarse fields, and demo/retraining data
    is often sampled or synthesised from base rows, so two distinct records can
    have identical feature values. Content hashing would falsely treat them as
    the same record.
    """
    for r in rows or []:
        if isinstance(r, dict) and not r.get("__rid"):
            r["__rid"] = uuid.uuid4().hex
    return rows


def add_pending(rows) -> int:
    with _LOCK:
        cur = get_pending()
        cur.extend(_ensure_rid(rows))
        set_pending(cur)
        return len(cur)


def clear_pending() -> None:
    set_pending([])


CURRENT_BATCH_KEY = "current_batch.json"


def get_current_batch():
    return read_json(CURRENT_BATCH_KEY, default={}) or {}


def clear_current_batch() -> None:
    delete_key(CURRENT_BATCH_KEY)


def begin_pending_batch(run_id: str) -> list:
    """Atomically lock the current pending rows for one retraining run.

    Rows uploaded after this point stay in pending.json and are NOT consumed by
    the running job. If the worker fails, restore_current_batch() puts the locked
    rows back into pending for retry.
    """
    with _LOCK:
        rows = _ensure_rid(get_pending())
        set_pending([])
        write_json(CURRENT_BATCH_KEY, {
            "run_id": run_id,
            "time": now_iso(),
            "row_count": len(rows),
            "rows": rows,
        })
        return rows


def restore_current_batch(run_id: str = None, reason: str = "") -> int:
    """Return the locked run batch to pending after a system failure.

    New uploads that arrived during the failed run remain in pending; the failed
    batch is prepended so it can be retried with the next run.
    """
    with _LOCK:
        batch = get_current_batch()
        if not batch:
            return 0
        if run_id and batch.get("run_id") and batch.get("run_id") != run_id:
            return 0
        rows = _ensure_rid(batch.get("rows") or [])
        cur = _ensure_rid(get_pending())
        set_pending(rows + cur)
        delete_key(CURRENT_BATCH_KEY)
        return len(rows)


# Accumulated training data: clean rows that have been consumed by a retrain are
# folded here so they permanently join the training base. This is what makes the
# loop "continuous" — every retrain trains on base + ALL accepted data so far,
# not just the latest batch.
def get_accumulated():
    return read_json("accumulated.json", default=[]) or []


def add_accumulated(rows) -> int:
    with _LOCK:
        cur = get_accumulated()
        cur.extend(rows or [])
        write_json("accumulated.json", cur)
        return len(cur)


# ---- Phase 2: recent (drift-aware) holdout pool -----------------------------
def get_recent_holdout():
    """Rows reserved for the recent/drift-aware validation holdout.

    NEVER trained on. Populated by splitting ~20% of each consumed batch off
    BEFORE the remainder is added to accumulated training data.
    """
    return read_json("recent_holdout.json", default=[]) or []


def _is_recent_holdout_row(rid) -> bool:
    """Deterministic ~20% bucket by row_id (record identity, not content)."""
    h = int(hashlib.md5(str(rid).encode("utf-8")).hexdigest(), 16)
    return (h % RECENT_HOLDOUT_BUCKET) == 0


def add_recent_holdout(rows) -> int:
    """Append rows to the recent holdout pool, capped at MAX_RECENT_HOLDOUT.

    Oldest overflow is EVICTED INTO accumulated training data (never discarded —
    evicted rows are still valid labelled records, just no longer "recent").
    """
    overflow = []
    with _LOCK:
        cur = get_recent_holdout()
        cur.extend(rows or [])
        if len(cur) > MAX_RECENT_HOLDOUT:
            cut = len(cur) - MAX_RECENT_HOLDOUT
            overflow = cur[:cut]          # oldest
            cur = cur[cut:]
        write_json("recent_holdout.json", cur)
    if overflow:
        add_accumulated(overflow)         # evicted recent rows -> training base
    return len(get_recent_holdout())


def get_attempted_batches():
    """Batches of clean rows that have already been used in a retraining attempt.

    These rows are no longer counted as `new_rows` for Scheduler/normal retrain.
    They are kept for auditability and, because they are valid labelled records,
    folded into the accumulated historical training base.
    """
    return read_json("attempted_batches.json", default=[]) or []


def add_attempted_batch(record: dict, limit: int = 100) -> None:
    with _LOCK:
        rows = get_attempted_batches()
        rows.insert(0, record)
        write_json("attempted_batches.json", rows[:limit])


def consume_pending_batch(run_id: str, outcome: str, evaluation=None, reason: str = "", rows=None) -> int:
    """Archive ONLY the rows locked for this retraining attempt.

    Older versions cleared the whole pending pool at the end of a run. That was
    wrong: if the admin uploaded a new CSV while GA was still running, those new
    rows were never trained but were still consumed. This function consumes the
    locked current_batch.json rows only and leaves new pending uploads queued for
    the next retraining run.
    """
    batch = get_current_batch()
    if rows is None:
        if batch and (not run_id or batch.get("run_id") == run_id):
            rows = batch.get("rows") or []
        else:
            # Backward-compatible fallback for old/local states with no
            # current_batch.json. Avoid clearing pending if a different run owns
            # the batch.
            rows = get_pending()
    rows = _ensure_rid(rows or [])
    n = len(rows)
    if n:
        # Phase 2: split the consumed run batch by row_id BEFORE folding into training.
        holdout_rows = [r for r in rows if _is_recent_holdout_row(r.get("__rid"))]
        train_rows = [r for r in rows if not _is_recent_holdout_row(r.get("__rid"))]
        add_accumulated(train_rows)
        add_recent_holdout(holdout_rows)
        add_attempted_batch({
            "run_id": run_id,
            "time": now_iso(),
            "outcome": outcome,
            "row_count": n,
            "to_train": len(train_rows),
            "to_recent_holdout": len(holdout_rows),
            "old_mse": (evaluation or {}).get("old_mse") if isinstance(evaluation, dict) else None,
            "candidate_mse": (evaluation or {}).get("new_mse") if isinstance(evaluation, dict) else None,
            "promoted_version": (evaluation or {}).get("version") if isinstance(evaluation, dict) else None,
            "reason": reason,
        })
    if batch and (not run_id or batch.get("run_id") == run_id):
        clear_current_batch()
    elif rows is get_pending():
        clear_pending()
    return n



def get_review():
    return read_json("review.json", default=[]) or []


def add_review(rows) -> None:
    with _LOCK:
        cur = get_review()
        cur.extend(rows)
        write_json("review.json", cur)


def set_review(rows) -> None:
    write_json("review.json", rows)


def get_status():
    return read_json("status.json", default={"state": "idle", "message": "Ready", "updated": now_iso()})


def set_status(state: str, message: str, extra=None) -> None:
    payload = {"state": state, "message": message, "updated": now_iso()}
    if extra:
        payload.update(extra)
    write_json("status.json", payload)


# ----------------------------------------------------------------
# Phase 4 — atomic retrain lease lock.
#
#   * Liveness is judged ONLY by heartbeat freshness, never by started_at — a
#     long but actively-heartbeating run must not be killed.
#   * acquire / stale-recovery / heartbeat / release all use compare-and-set on
#     the storage object generation (GCS if-generation-match; local mtime under
#     the process lock) so two concurrent triggers cannot both win.
#   * The running job pumps heartbeats on a side thread, verifies it still owns
#     the lock before its final commit, and releases on exit.
# ----------------------------------------------------------------
RETRAIN_LOCK_KEY = "retrain_lock.json"
HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "30"))
HEARTBEAT_STALE_SECONDS = int(os.environ.get("HEARTBEAT_STALE_SECONDS", "120"))


def _read_lock_with_gen():
    """Return (lock_dict_or_None, generation). generation is an opaque CAS token;
    0 means 'object absent' (used as the create-if-absent precondition).
    GCS uses the real object generation; local uses an embedded monotonic
    counter (`_gen`) so CAS is correct regardless of filesystem mtime resolution."""
    if using_gcs():
        blob = _gcs_bucket.blob(RETRAIN_LOCK_KEY)
        if not blob.exists():
            return None, 0
        try:
            blob.reload()
            gen = int(blob.generation or 0)
            data = blob.download_as_bytes()
            return json.loads(data), gen
        except Exception:
            return None, 0
    path = os.path.join(_LOCAL_ROOT, RETRAIN_LOCK_KEY)
    if not os.path.exists(path):
        return None, 0
    try:
        with open(path) as fh:
            obj = json.load(fh)
        return obj, int(obj.get("_gen", 0))
    except Exception:
        return None, 0


def _cas_write_lock(obj, expected_gen) -> bool:
    """Write the lock ONLY if the current generation still equals expected_gen."""
    if using_gcs():
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        blob = _gcs_bucket.blob(RETRAIN_LOCK_KEY)
        try:
            blob.upload_from_string(data, content_type="application/json",
                                    if_generation_match=expected_gen)
            return True
        except Exception:
            return False  # PreconditionFailed -> someone else changed it first
    with _LOCK:
        path = _local_path(RETRAIN_LOCK_KEY)
        cur_gen = 0
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    cur_gen = int(json.load(fh).get("_gen", 0))
            except Exception:
                cur_gen = 0
        if cur_gen != expected_gen:
            return False
        out = dict(obj)
        out["_gen"] = int(expected_gen) + 1  # monotonic bump
        with open(path, "wb") as fh:
            fh.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
        return True


def _cas_delete_lock(expected_gen) -> bool:
    if using_gcs():
        blob = _gcs_bucket.blob(RETRAIN_LOCK_KEY)
        try:
            blob.delete(if_generation_match=expected_gen)
            return True
        except Exception:
            return False
    with _LOCK:
        path = os.path.join(_LOCAL_ROOT, RETRAIN_LOCK_KEY)
        cur_gen = 0
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    cur_gen = int(json.load(fh).get("_gen", 0))
            except Exception:
                cur_gen = 0
        if cur_gen != expected_gen:
            return False
        if os.path.exists(path):
            os.remove(path)
        return True


def _make_lock(run_id, owner_token, source, recovered_from=None):
    now = time.time()
    d = {"run_id": run_id, "owner_token": owner_token, "source": source,
         "started_at": now, "started_iso": now_iso(), "heartbeat_at": now}
    if recovered_from:
        d["recovered_from"] = recovered_from
    return d


def _lock_is_stale(lock) -> bool:
    # Liveness by HEARTBEAT ONLY. started_at is never used here.
    return (time.time() - float(lock.get("heartbeat_at", 0))) > HEARTBEAT_STALE_SECONDS


def get_retrain_lock():
    lock, _ = _read_lock_with_gen()
    return lock


def acquire_retrain_lock(run_id, owner_token, source=""):
    """Atomically acquire the lease. Returns the lock dict on success, else None.

    Wins if: no lock exists (atomic create), OR the existing lock is stale
    (atomic takeover only if its generation is unchanged since we read it).
    A fresh, live lock is never stolen.
    """
    lock, gen = _read_lock_with_gen()
    if lock is None:
        new = _make_lock(run_id, owner_token, source)
        return new if _cas_write_lock(new, gen) else None
    if not _lock_is_stale(lock):
        return None
    new = _make_lock(run_id, owner_token, source, recovered_from=lock.get("run_id"))
    return new if _cas_write_lock(new, gen) else None


def heartbeat_retrain_lock(owner_token) -> bool:
    """Refresh heartbeat_at iff we still own the lock. False if taken over/gone."""
    lock, gen = _read_lock_with_gen()
    if not lock or lock.get("owner_token") != owner_token:
        return False
    lock["heartbeat_at"] = time.time()
    return _cas_write_lock(lock, gen)


def still_own_retrain_lock(owner_token) -> bool:
    lock, _ = _read_lock_with_gen()
    return bool(lock and lock.get("owner_token") == owner_token)


def release_retrain_lock(owner_token) -> bool:
    lock, gen = _read_lock_with_gen()
    if lock and lock.get("owner_token") == owner_token:
        return _cas_delete_lock(gen)
    return False


def recover_stale_retrain_status() -> bool:
    """Clear a stale Running UI state when the worker is no longer alive.

    A Cloud Task / browser refresh can leave status.json at state=running after
    the worker request has already stopped. The lease heartbeat is the source of
    truth: if there is no lock, or the lock heartbeat is stale, the admin console
    should become retryable without consuming pending rows.
    """
    status = get_status()
    if str(status.get("state", "")).lower() != "running":
        return False

    lock, gen = _read_lock_with_gen()
    if lock is None:
        restore_current_batch(reason="recovered interrupted run")
        set_status("idle", "Previous retraining job was interrupted. Locked batch restored to pending.",
                   extra={"recovered": True})
        return True

    if _lock_is_stale(lock):
        _cas_delete_lock(gen)
        restore_current_batch(lock.get("run_id"), reason="stale retrain lock")
        set_status("idle", "Previous retraining job was interrupted. Locked batch restored to pending.",
                   extra={"recovered": True, "recovered_run_id": lock.get("run_id")})
        return True

    return False


def get_active_params():
    return read_json("active/mf_params.json", default=None)


def set_active_params(params: dict) -> None:
    write_json("active/mf_params.json", params)


def list_versions():
    """Return version metadata sorted by version number descending."""
    metas = []
    keys = list_prefix("versions/")
    seen = set()
    for k in keys:
        # versions/v{n}/mf_params.json
        parts = k.split("/")
        if len(parts) >= 3 and parts[2] == "mf_params.json":
            vdir = parts[1]
            if vdir in seen:
                continue
            seen.add(vdir)
            meta = read_json(f"versions/{vdir}/mf_params.json", default={})
            metas.append(meta)
    metas.sort(key=lambda m: m.get("version", 0), reverse=True)
    return metas


def next_version_number() -> int:
    versions = list_versions()
    return (versions[0]["version"] + 1) if versions else 1


def save_version(version: int, params: dict, changelog: str, zip_bytes: bytes) -> None:
    base = f"versions/v{version}"
    write_json(f"{base}/mf_params.json", params)
    write_text(f"{base}/changelog.txt", changelog)
    write_bytes(f"{base}/model_v{version}.zip", zip_bytes, content_type="application/zip")


def read_version_params(version: int):
    return read_json(f"versions/v{version}/mf_params.json", default=None)


def read_version_zip(version: int):
    return read_bytes(f"versions/v{version}/model_v{version}.zip")


def append_changelog(line: str) -> None:
    with _LOCK:
        existing = read_text("changelog.txt") or ""
        write_text("changelog.txt", existing + line.rstrip("\n") + "\n")


def read_changelog() -> str:
    return read_text("changelog.txt") or ""


# ----------------------------------------------------------------
# Retraining history / candidate-run audit records
# ----------------------------------------------------------------
def get_runs():
    """Return recent retraining/rollback audit records, newest first.

    These are not production versions. They include promoted, skipped, failed,
    and rollback events so skipped candidates can be audited without becoming
    rollbackable model versions.
    """
    return read_json("runs.json", default=[]) or []


def set_runs(rows) -> None:
    write_json("runs.json", rows or [])


def add_run(record: dict, limit: int = 200) -> None:
    with _LOCK:
        rows = get_runs()
        rows.insert(0, record)
        write_json("runs.json", rows[:limit])


# ----------------------------------------------------------------
# Demo/admin reset helpers (admin-only endpoint calls these)
# ----------------------------------------------------------------
def delete_key(key: str) -> None:
    """Delete a single persisted key if it exists."""
    if using_gcs():
        blob = _gcs_bucket.blob(key)
        try:
            if blob.exists():
                blob.delete()
        except Exception:
            pass
        return
    path = os.path.join(_LOCAL_ROOT, key)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def delete_prefix(prefix: str) -> None:
    """Delete every object/file under a prefix."""
    keys = list_prefix(prefix)
    for key in keys:
        delete_key(key)


def reset_learning_state(keep_v0: bool = True) -> None:
    """Reset demo/test learning state while preserving code and dataset.

    This clears uploaded/candidate data, review rows, accumulated accepted rows,
    retraining runs, global changelog, status, and active model pointer. Promoted
    versions are removed so the console returns to a clean v0 baseline state.
    If keep_v0=True, the stored versions/v0 artifact is preserved.
    """
    for key in [
        "pending.json", "review.json", "accumulated.json", "attempted_batches.json",
        "recent_holdout.json", "current_batch.json", "retrain_lock.json",
        "runs.json", "changelog.txt", "status.json", "active/mf_params.json"
    ]:
        delete_key(key)

    # Delete promoted versions. Preserve v0 by default as the rollback baseline.
    for key in list_prefix("versions/"):
        if keep_v0 and key.startswith("versions/v0/"):
            continue
        delete_key(key)

    set_status("idle", "Demo learning state reset to v0 baseline.")
