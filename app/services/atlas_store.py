"""
Stored Intervention Atlas results, one per (disease, PC groups) pair.

A risk area depends only on the disease and its contributing PCs, not on the patient, so one
stored result serves every patient with the same risk and PCs. Kept in SQLite on a Docker volume
so results survive restarts and redeploys.
"""

import json
import logging
import os
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

STORE_PATH = os.getenv("ATLAS_STORE_PATH", "/app/atlas_store/atlas.db")
FORMAT_VERSION = "10"
MAX_AGE_SECONDS = int(os.getenv("ATLAS_STORE_MAX_AGE_DAYS", "7")) * 86400

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
        _conn = sqlite3.connect(STORE_PATH, check_same_thread=False)
        _conn.execute("""CREATE TABLE IF NOT EXISTS risk_area (
            disease TEXT NOT NULL, pc_groups TEXT NOT NULL, built_at REAL NOT NULL, result TEXT NOT NULL,
            PRIMARY KEY (disease, pc_groups))""")
        _conn.execute("""CREATE TABLE IF NOT EXISTS patient_risks (
            patient_id TEXT PRIMARY KEY, fetched_at REAL NOT NULL, patient TEXT NOT NULL)""")
        _conn.execute("""CREATE TABLE IF NOT EXISTS paper_check (
            pmid TEXT NOT NULL, item TEXT NOT NULL, disease TEXT NOT NULL, verdict INTEGER NOT NULL,
            reason TEXT, checked_at REAL NOT NULL, PRIMARY KEY (pmid, item, disease))""")
        _conn.execute("""CREATE TABLE IF NOT EXISTS precompute_run (
            finished_at REAL NOT NULL, summary TEXT NOT NULL)""")
        _conn.commit()
    return _conn


def _pc_key(pc_groups: list[str]) -> str:
    return f"v{FORMAT_VERSION}:" + ",".join(sorted(pc_groups))


def get(disease: str, pc_groups: list[str]) -> dict | None:
    """The stored result for this disease + PCs, or None when missing or older than the max age."""
    try:
        with _lock:
            row = _db().execute("SELECT built_at, result FROM risk_area WHERE disease = ? AND pc_groups = ?",
                                (disease, _pc_key(pc_groups))).fetchone()
    except Exception as e:
        logger.warning(f"atlas_store | read failed for {disease!r}: {e}")
        return None
    if not row or time.time() - row[0] > MAX_AGE_SECONDS:
        return None
    return json.loads(row[1])


def put(disease: str, pc_groups: list[str], result: dict) -> None:
    try:
        with _lock:
            _db().execute("INSERT OR REPLACE INTO risk_area VALUES (?, ?, ?, ?)",
                          (disease, _pc_key(pc_groups), time.time(), json.dumps(result)))
            _db().commit()
    except Exception as e:
        logger.warning(f"atlas_store | write failed for {disease!r}: {e}")


def record_run(summary: dict) -> None:
    try:
        with _lock:
            _db().execute("INSERT INTO precompute_run VALUES (?, ?)", (time.time(), json.dumps(summary)))
            _db().commit()
    except Exception as e:
        logger.warning(f"atlas_store | recording precompute run failed: {e}")


def last_run_at() -> float | None:
    """When the last precompute run finished, or None if none has."""
    try:
        with _lock:
            row = _db().execute("SELECT MAX(finished_at) FROM precompute_run").fetchone()
    except Exception as e:
        logger.warning(f"atlas_store | reading precompute runs failed: {e}")
        return None
    return row[0] if row else None


def put_patients(patients: list[dict]) -> None:
    """Saves each patient's gender and risks (as ALIS returns them) so /atlas needn't call ALIS."""
    rows = [(p["id"], time.time(), json.dumps({"gender": p.get("gender"), "clinic_id": p.get("clinic_id"),
                                               "risks": p.get("risks") or []}))
            for p in patients if p.get("id")]
    try:
        with _lock:
            _db().executemany("INSERT OR REPLACE INTO patient_risks VALUES (?, ?, ?)", rows)
            _db().commit()
    except Exception as e:
        logger.warning(f"atlas_store | saving patient risks failed: {e}")


def get_patient(patient_id: str) -> dict | None:
    """The patient's saved gender and risks, or None when this patient isn't saved yet."""
    try:
        with _lock:
            row = _db().execute("SELECT patient FROM patient_risks WHERE patient_id = ?", (patient_id,)).fetchone()
    except Exception as e:
        logger.warning(f"atlas_store | reading patient {patient_id} failed: {e}")
        return None
    return json.loads(row[0]) if row else None


def get_paper_verdict(pmid: str, item: str, disease: str) -> bool | None:
    try:
        with _lock:
            row = _db().execute("SELECT verdict FROM paper_check WHERE pmid = ? AND item = ? AND disease = ?",
                                (pmid, item.lower(), disease.lower())).fetchone()
    except Exception as e:
        logger.warning(f"atlas_store | reading paper verdict {pmid} failed: {e}")
        return None
    return bool(row[0]) if row else None


def put_paper_verdict(pmid: str, item: str, disease: str, verdict: bool, reason: str | None) -> None:
    try:
        with _lock:
            _db().execute("INSERT OR REPLACE INTO paper_check VALUES (?, ?, ?, ?, ?, ?)",
                          (pmid, item.lower(), disease.lower(), int(verdict), reason, time.time()))
            _db().commit()
    except Exception as e:
        logger.warning(f"atlas_store | saving paper verdict {pmid} failed: {e}")


def get_paper_record(pmid: str, item: str, disease: str) -> tuple[bool, str | None] | None:
    """(verdict, reason) of a stored check, or None when it hasn't been checked."""
    try:
        with _lock:
            row = _db().execute("SELECT verdict, reason FROM paper_check WHERE pmid = ? AND item = ? AND disease = ?",
                                (pmid, item.lower(), disease.lower())).fetchone()
    except Exception as e:
        logger.warning(f"atlas_store | reading check {pmid} failed: {e}")
        return None
    return (bool(row[0]), row[1]) if row else None
