"""
Daily precompute of the Intervention Atlas for every patient, so /atlas only reads stored results.

Every ATLAS_PRECOMPUTE_HOURS (default 24) it fetches all patients' risks from ALIS's internal
endpoint (service key, no user token), collects the unique (disease, PC groups) pairs, and builds
only the ones not in the Atlas store yet: new patients, new risks, changed PCs, and results past
the store's max age (ATLAS_STORE_MAX_AGE_DAYS). A pair shared by many patients is built once.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.services import atlas_store
from app.services.backend_api import fetch_all_patient_risks

logger = logging.getLogger(__name__)

_INTERVAL_SECONDS = float(os.getenv("ATLAS_PRECOMPUTE_HOURS", "24")) * 3600
_WORKERS = int(os.getenv("ATLAS_PRECOMPUTE_WORKERS", "2"))
_STARTUP_DELAY_SECONDS = 60

_run_lock = threading.Lock()


def _pairs(patients: list[dict]) -> dict[tuple[str, tuple[str, ...]], tuple[str, list[str]]]:
    """Unique (disease, sorted PC groups) -> (disease, PC groups in the backend's order)."""
    pairs = {}
    for patient in patients:
        suffix = "F" if (patient.get("gender") or "").lower() == "female" else "M"
        for risk in patient.get("risks") or []:
            disease = risk.get("disease_name")
            if not disease:
                continue
            pc_groups = [f"{pc}{suffix}" for pc in risk.get("contributing_pcs") or []]
            pairs.setdefault((disease, tuple(sorted(pc_groups))), (disease, pc_groups))
    return pairs


def run_precompute() -> dict | None:
    """Builds every patient's risk areas that aren't stored yet; None if a run is already going."""
    if not _run_lock.acquire(blocking=False):
        logger.info("atlas_precompute | a run is already in progress; skipping")
        return None
    try:
        from app.services.atlas_service import combined_mechanisms
        from app.services.atlas_v2 import build_risk_area

        started = time.time()
        patients = fetch_all_patient_risks()
        atlas_store.put_patients(patients)
        all_pairs = list(_pairs(patients).values())
        from app.services.disease_terms import is_mapped
        unmapped = sorted({d for d, _ in all_pairs if not is_mapped(d)})
        if unmapped:
            logger.warning(f"atlas_precompute | risk names with no search terms (add them to disease_terms.py): {unmapped}")
        pairs = [(d, pcs) for d, pcs in all_pairs if atlas_store.get(d, pcs) is None]
        logger.info(f"atlas_precompute | started | patients={len(patients)} | unique risk areas={len(all_pairs)} "
                    f"| already stored={len(all_pairs) - len(pairs)} | to build={len(pairs)}")

        done, failed = 0, []
        def _build(pair: tuple[str, list[str]]) -> None:
            nonlocal done
            disease, pc_groups = pair
            try:
                mechanisms, _ = combined_mechanisms(pc_groups)
                build_risk_area(disease, pc_groups, mechanisms)
            except Exception as e:
                logger.error(f"atlas_precompute | {disease} {pc_groups} failed: {e}")
                failed.append(disease)
            done += 1
            if done % 10 == 0 or done == len(pairs):
                logger.info(f"atlas_precompute | {done}/{len(pairs)} built | {time.time() - started:.0f}s")

        with ThreadPoolExecutor(max_workers=_WORKERS) as executor:
            list(executor.map(_build, pairs))

        summary = {"patients": len(patients), "risk_areas": len(all_pairs), "built": len(pairs), "failed": len(failed),
                   "seconds": round(time.time() - started)}
        atlas_store.record_run(summary)
        logger.info(f"atlas_precompute | finished | {summary}")
        return summary
    except Exception as e:
        logger.error(f"atlas_precompute | run failed: {e}")
        return None
    finally:
        _run_lock.release()


def precompute_loop() -> None:
    """Runs the precompute every _INTERVAL_SECONDS, timed from the last finished run so that
    restarts and redeploys don't trigger an extra run."""
    if os.getenv("ATLAS_PRECOMPUTE", "1") == "0":
        logger.info("atlas_precompute | disabled (ATLAS_PRECOMPUTE=0)")
        return
    if not os.getenv("ALIS_SERVICE_KEY"):
        logger.warning("atlas_precompute | ALIS_SERVICE_KEY is not set; daily precompute is off")
        return
    time.sleep(_STARTUP_DELAY_SECONDS)
    while True:
        last = atlas_store.last_run_at()
        wait = 0 if last is None else max(0, last + _INTERVAL_SECONDS - time.time())
        if wait:
            logger.info(f"atlas_precompute | next run in {wait / 3600:.1f}h")
            time.sleep(wait)
        if run_precompute() is None:
            time.sleep(3600)
