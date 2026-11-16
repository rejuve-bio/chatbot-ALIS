"""
NLM RxClass: drug classes (from DailyMed labels) and MED-RT drug–disease relations.
"""

import logging
import threading
import time

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://rxnav.nlm.nih.gov/REST/rxclass"

_CACHE: dict[tuple, tuple[float, list[str]]] = {}
_CACHE_TTL_SECONDS = 7 * 86400
_cache_lock = threading.Lock()


def _cached(key: tuple, fetch) -> list[str]:
    with _cache_lock:
        cached = _CACHE.get(key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]
    try:
        value = fetch()
    except Exception as e:
        logger.warning(f"RxClass | lookup {key} failed: {e}")
        return []
    with _cache_lock:
        _CACHE[key] = (time.time(), value)
    return value


def get_drug_classes(drug_name: str) -> list[str]:
    """Established pharmacologic classes of a drug, e.g. ["Biguanide"]."""
    def fetch():
        resp = httpx.get(f"{BASE_URL}/class/byDrugName.json",
                         params={"drugName": drug_name, "relaSource": "DAILYMED", "relas": "has_epc"},
                         timeout=30.0)
        resp.raise_for_status()
        infos = resp.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
        return sorted({i["rxclassMinConceptItem"]["className"] for i in infos})
    return _cached(("epc", drug_name.lower()), fetch)


def get_disease_mesh_ids(disease_heading: str) -> list[str]:
    """MeSH disease IDs NLM RxClass knows under this heading, e.g. "Asthma" -> ["D001249"]."""
    def fetch():
        resp = httpx.get(f"{BASE_URL}/class/byName.json",
                         params={"className": disease_heading, "classTypes": "DISEASE"}, timeout=30.0)
        resp.raise_for_status()
        concepts = resp.json().get("rxclassMinConceptList", {}).get("rxclassMinConcept", [])
        return [c["classId"] for c in concepts]
    return _cached(("mesh", disease_heading.lower()), fetch)


def get_contraindicated_drugs(mesh_id: str) -> list[str]:
    """Drugs MED-RT lists as contraindicated with this disease."""
    def fetch():
        resp = httpx.get(f"{BASE_URL}/classMembers.json",
                         params={"classId": mesh_id, "relaSource": "MEDRT", "rela": "CI_with", "ttys": "IN"},
                         timeout=30.0)
        resp.raise_for_status()
        members = resp.json().get("drugMemberGroup", {}).get("drugMember", [])
        return sorted({m["minConcept"]["name"].lower() for m in members})
    return _cached(("ci", mesh_id), fetch)
