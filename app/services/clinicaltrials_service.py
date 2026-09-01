"""
Live query against ClinicalTrials.gov's real API 
"""

import logging

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = get_active_trials_for_compound("Rapamycin")
    print(f"Found {len(result)} active trials for Rapamycin")
    for t in result:
        print(t)
