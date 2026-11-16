"""
Approved drugs per disease from ChEMBL's curated drug_indication data (EMBL-EBI).
"""

import logging
import threading
import time

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

_SALT_WORDS = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium", "besylate", "benzoate",
    "mesylate", "maleate", "fumarate", "phosphate", "succinate", "tartrate", "propanediol",
    "pidolate", "monohydrate", "dihydrate", "hydrobromide", "bromide", "acetate", "sulfate",
    "citrate", "lysine", "medoxomil", "cilexetil", "kamedoxomil", "bisulfate", "xinafoate",
}

_CACHE: dict[str, tuple[float, list[str]]] = {}
_CACHE_TTL_SECONDS = 7 * 86400
_cache_lock = threading.Lock()


_ION_WORDS = {"ferric", "ferrous", "iron", "zinc", "calcium", "magnesium", "potassium", "sodium", "lithium"}


def base_drug_name(name: str) -> str:
    words = name.lower().split()
    while len(words) > 1 and words[-1] in _SALT_WORDS and not (len(words) == 2 and words[0] in _ION_WORDS):
        words.pop()
    return " ".join(words)


_WITHDRAWN_CACHE: dict[str, bool] = {}


def is_withdrawn(name: str) -> bool:
    """True when ChEMBL flags the drug as withdrawn from the market (e.g. phenformin)."""
    key = name.strip().lower()
    if key in _WITHDRAWN_CACHE:
        return _WITHDRAWN_CACHE[key]
    try:
        resp = httpx.get(f"{BASE_URL}/molecule.json", params={"pref_name__iexact": key, "limit": 5}, timeout=30.0)
        resp.raise_for_status()
        withdrawn = any(m.get("withdrawn_flag") for m in resp.json().get("molecules", []))
    except Exception as e:
        logger.warning(f"ChEMBL | withdrawn check failed for {name!r}: {e}")
        raise ChemblUnavailable(name) from e
    _WITHDRAWN_CACHE[key] = withdrawn
    return withdrawn


_TARGET_CACHE: dict[str, set[str]] = {}


_HUMAN_PROTEIN_TARGETS = ("SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY")


def get_target_genes(name: str) -> set[str]:
    """Gene symbols of the drug's targets from ChEMBL's curated mechanism-of-action records
    (e.g. sirolimus -> FKBP1A, acarbose -> AMY2A, MGAM). Empty when ChEMBL has no mechanism."""
    key = name.strip().lower()
    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key]
    genes: set[str] = set()
    try:
        mols = httpx.get(f"{BASE_URL}/molecule.json", params={"pref_name__iexact": key, "limit": 1},
                         timeout=30.0).json().get("molecules", [])
        if mols:
            parent = (mols[0].get("molecule_hierarchy") or {}).get("parent_chembl_id") or mols[0]["molecule_chembl_id"]
            mechs = httpx.get(f"{BASE_URL}/mechanism.json",
                              params={"parent_molecule_chembl_id": parent, "limit": 50},
                              timeout=30.0).json().get("mechanisms", [])
            for target_id in {m["target_chembl_id"] for m in mechs if m.get("target_chembl_id")}:
                target = httpx.get(f"{BASE_URL}/target/{target_id}.json", timeout=30.0).json()
                if target.get("organism") != "Homo sapiens" or target.get("target_type") not in _HUMAN_PROTEIN_TARGETS:
                    continue
                for component in target.get("target_components", []):
                    genes.update(s["component_synonym"] for s in component.get("target_component_synonyms", [])
                                 if s.get("syn_type") == "GENE_SYMBOL")
    except Exception as e:
        logger.warning(f"ChEMBL | target lookup failed for {name!r}: {e}")
        raise ChemblUnavailable(name) from e
    _TARGET_CACHE[key] = genes
    return genes


def _get_with_retry(url: str, params: dict, attempts: int = 4) -> httpx.Response:
    """ChEMBL's API returns occasional 5xx errors under load; retry those with a growing pause."""
    for attempt in range(attempts):
        resp = httpx.get(url, params=params, timeout=60.0)
        if resp.status_code < 500 or attempt == attempts - 1:
            break
        time.sleep(2 ** attempt)
    resp.raise_for_status()
    return resp


class ChemblUnavailable(Exception):
    """ChEMBL couldn't be reached; unlike an empty list, this doesn't mean "no approved drugs"."""


def get_approved_drugs(mesh_heading: str) -> list[str]:
    """Base names of drugs ChEMBL lists as approved (max phase 4) for this MeSH disease heading."""
    with _cache_lock:
        cached = _CACHE.get(mesh_heading)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        resp = _get_with_retry(f"{BASE_URL}/drug_indication.json",
                               {"mesh_heading": mesh_heading, "max_phase_for_ind": 4, "limit": 1000})
        ids = sorted({d["molecule_chembl_id"] for d in resp.json().get("drug_indications", [])})
        names: set[str] = set()
        for i in range(0, len(ids), 50):
            mol = _get_with_retry(f"{BASE_URL}/molecule.json",
                                  {"molecule_chembl_id__in": ",".join(ids[i:i + 50]), "limit": 50})
            names.update(base_drug_name(m["pref_name"]) for m in mol.json().get("molecules", [])
                         if m.get("pref_name"))
    except Exception as e:
        logger.warning(f"ChEMBL | approved-drug lookup failed for {mesh_heading!r}: {e}")
        raise ChemblUnavailable(mesh_heading) from e

    drugs = sorted(names)
    logger.info(f"ChEMBL | {mesh_heading!r} -> {len(drugs)} approved drug(s)")
    with _cache_lock:
        _CACHE[mesh_heading] = (time.time(), drugs)
    return drugs
