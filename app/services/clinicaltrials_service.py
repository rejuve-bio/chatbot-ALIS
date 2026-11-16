"""
Live query against ClinicalTrials.gov's real API 
"""

import logging
import threading
import time

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
_STATUSES = "RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION"


def get_active_trials_for_compound(compound_name: str, condition: str | None = None, limit: int = 3) -> list[dict]:
    """
    Active/recruiting trials testing this compound as an intervention, real-time.
    """
    params = {
        "query.intr": compound_name,
        "filter.overallStatus": _STATUSES,
        "pageSize": limit,
        "fields": "NCTId,BriefTitle,OverallStatus,Condition,Phase",
    }
    if condition:
        params["query.cond"] = condition
    try:
        resp = httpx.get(BASE_URL, params=params, timeout=15.0)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"ClinicalTrials.gov | failed for compound={compound_name}, condition={condition}: {e}")
        return []

    studies = resp.json().get("studies", [])
    trials = []
    for s in studies:
        proto = s.get("protocolSection", {})
        trials.append({
            "nct_id": proto.get("identificationModule", {}).get("nctId"),
            "title": proto.get("identificationModule", {}).get("briefTitle"),
            "status": proto.get("statusModule", {}).get("overallStatus"),
            "conditions": proto.get("conditionsModule", {}).get("conditions", []),
            "phase": proto.get("designModule", {}).get("phases", []),
        })
    logger.info(f"ClinicalTrials.gov | compound={compound_name} | condition={condition} | "
                f"found {len(trials)} active/recruiting trial(s)")
    return trials


_RATE_LIMIT = threading.Semaphore(2)
_SEARCH_CACHE: dict[tuple, tuple[float, tuple[list[dict], int]]] = {}
_SEARCH_CACHE_TTL_SECONDS = 86400


def search_active_trials(interventions: list[str], condition: str | None, limit: int = 2) -> tuple[list[dict], int]:
    """Active trials testing any of these interventions for the condition, plus the total count on
    ClinicalTrials.gov (used to order drug classes by real trial activity)."""
    params = {
        "query.intr": " OR ".join(interventions),
        "filter.overallStatus": _STATUSES,
        "pageSize": limit,
        "countTotal": "true",
        "fields": "NCTId,BriefTitle,OverallStatus,Phase",
    }
    if condition:
        params["query.cond"] = condition
    cache_key = (tuple(interventions), condition, limit)
    cached = _SEARCH_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _SEARCH_CACHE_TTL_SECONDS:
        return cached[1]
    resp = None
    for attempt in range(6):
        try:
            with _RATE_LIMIT:
                resp = httpx.get(BASE_URL, params=params, timeout=15.0)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            break
        except Exception as e:
            logger.warning(f"ClinicalTrials.gov | failed for interventions={interventions}, condition={condition}: {e}")
            return [], 0
    if resp is None or resp.status_code == 429:
        logger.warning(f"ClinicalTrials.gov | still rate-limited for interventions={interventions}")
        return [], 0

    body = resp.json()
    trials = []
    for s in body.get("studies", []):
        proto = s.get("protocolSection", {})
        nct_id = proto.get("identificationModule", {}).get("nctId")
        trials.append({
            "nct_id": nct_id,
            "title": proto.get("identificationModule", {}).get("briefTitle"),
            "status": proto.get("statusModule", {}).get("overallStatus"),
            "phase": proto.get("designModule", {}).get("phases", []),
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
        })
    result = (trials, body.get("totalCount", len(trials)))
    _SEARCH_CACHE[cache_key] = (time.time(), result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = get_active_trials_for_compound("Rapamycin")
    print(f"Found {len(result)} active trials for Rapamycin")
    for t in result:
        print(t)
