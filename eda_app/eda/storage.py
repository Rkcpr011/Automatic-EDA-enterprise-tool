"""Thread-safe persistence: metadata.json (report registry) + per-job state files."""
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timedelta

import config

_LOCK = threading.RLock()
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------- helpers
def new_id() -> str:
    return uuid.uuid4().hex


def valid_id(job_id: str) -> bool:
    return bool(job_id and _ID_RE.match(job_id))


def job_dir(job_id):
    return config.UPLOAD_DIR / job_id


def report_dir(job_id):
    return config.REPORT_DIR / job_id


def _atomic_write_json(path, data):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------- metadata
def _read_meta():
    if not config.METADATA_FILE.exists():
        return {"reports": {}}
    try:
        with open(config.METADATA_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("reports", {})
        return data
    except (json.JSONDecodeError, OSError):
        # keep a copy of the unreadable file and start fresh rather than crash
        try:
            shutil.copy(config.METADATA_FILE, str(config.METADATA_FILE) + ".corrupt")
        except OSError:
            pass
        return {"reports": {}}


def create_entry(entry: dict):
    with _LOCK:
        data = _read_meta()
        data["reports"][entry["id"]] = entry
        _atomic_write_json(config.METADATA_FILE, data)


def get_entry(job_id):
    with _LOCK:
        return _read_meta()["reports"].get(job_id)


def update_entry(job_id, **fields):
    with _LOCK:
        data = _read_meta()
        if job_id not in data["reports"]:
            return None
        data["reports"][job_id].update(fields)
        _atomic_write_json(config.METADATA_FILE, data)
        return data["reports"][job_id]


def remove_entry(job_id):
    with _LOCK:
        data = _read_meta()
        data["reports"].pop(job_id, None)
        _atomic_write_json(config.METADATA_FILE, data)


def all_entries():
    with _LOCK:
        entries = list(_read_meta()["reports"].values())
    entries.sort(key=lambda e: e.get("upload_date", ""), reverse=True)
    return entries


def delete_job_files(job_id):
    """Remove everything on disk for a job (never touches paths outside our dirs)."""
    if not valid_id(job_id):
        return
    for base in (config.UPLOAD_DIR, config.REPORT_DIR):
        shutil.rmtree(base / job_id, ignore_errors=True)


def purge_job(job_id):
    delete_job_files(job_id)
    remove_entry(job_id)


def cleanup_stale():
    """Drop abandoned wizard drafts; flag builds interrupted by a restart."""
    cutoff = datetime.now() - timedelta(hours=config.STALE_DRAFT_HOURS)
    for e in all_entries():
        try:
            created = datetime.fromisoformat(e.get("upload_date", ""))
        except ValueError:
            continue
        if e.get("status") == "draft" and created < cutoff:
            purge_job(e["id"])


def mark_interrupted():
    for e in all_entries():
        if e.get("status") == "processing":
            update_entry(e["id"], status="failed", step="outliers",
                         error="Report build was interrupted by a server restart. Please re-run it.")


# ---------------------------------------------------------------- per-job state
def _state_path(job_id):
    return job_dir(job_id) / "state.json"


def get_state(job_id) -> dict:
    p = _state_path(job_id)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(job_id, state: dict):
    with _LOCK:
        _atomic_write_json(_state_path(job_id), state)


def update_state(job_id, **fields):
    with _LOCK:
        st = get_state(job_id)
        st.update(fields)
        save_state(job_id, st)
        return st
