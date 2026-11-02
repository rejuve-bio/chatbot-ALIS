"""
Risk mechanisms -> curated GO / Reactome pathways (data/pathway_mapping.py) -> the human genes in
each pathway, from the Reactome Content Service, or QuickGO for GO-only pathways.
"""

import logging
import threading
import time

import httpx

from data.pathway_mapping import PATHWAY_MAPPING

logger = logging.getLogger(__name__)

REACTOME_URL = "https://reactome.org/ContentService/data/participants/{}/referenceEntities"
QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"

MAX_GENES_FOR_MATCHING = 400

_CACHE: dict[str, tuple[float, set[str]]] = {}
_CACHE_TTL_SECONDS = 7 * 86400
_cache_lock = threading.Lock()


def _cached(key: str, fetch) -> set[str]:
    with _cache_lock:
        cached = _CACHE.get(key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]
    try:
        genes = fetch()
    except Exception as e:
        logger.warning(f"pathway genes | {key} lookup failed: {e}")
        return set()
    with _cache_lock:
        _CACHE[key] = (time.time(), genes)
    return genes


def _reactome_genes(stable_id: str) -> set[str]:
    def fetch():
        resp = httpx.get(REACTOME_URL.format(stable_id), timeout=60.0)
        resp.raise_for_status()
        return {e["geneName"][0] for e in resp.json() if e.get("geneName")}
    return _cached(stable_id, fetch)


def _go_genes(go_id: str) -> set[str]:
    def fetch():
        resp = httpx.get(QUICKGO_URL, params={
            "goId": go_id, "taxonId": 9606, "goUsage": "descendants",
            "geneProductSubset": "Swiss-Prot", "geneProductType": "protein", "limit": 200,
        }, headers={"Accept": "application/json"}, timeout=60.0)
        resp.raise_for_status()
        return {a["symbol"] for a in resp.json().get("results", []) if a.get("symbol")}
    return _cached(go_id, fetch)


def get_risk_pathways(mechanisms: list[str]) -> list[dict]:
    """The curated pathways behind these mechanisms, each with its human gene set."""
    rows: dict[str, dict] = {}
    for mechanism in mechanisms:
        for row in PATHWAY_MAPPING.get(mechanism, []):
            key = row["reactome_id"] or ",".join(row["go_ids"])
            entry = rows.setdefault(key, {**row, "mechanisms": []})
            if mechanism not in entry["mechanisms"]:
                entry["mechanisms"].append(mechanism)

    pathways = []
    for row in rows.values():
        genes = _reactome_genes(row["reactome_id"]) if row["reactome_id"] else set()
        if not genes:
            for go_id in row["go_ids"]:
                genes |= _go_genes(go_id)
        pathways.append({
            "name": row["reactome_name"] or row["go_name"],
            "pathway_term": row["pathway"],
            "mechanisms": row["mechanisms"],
            "reactome_id": row["reactome_id"],
            "go_ids": row["go_ids"],
            "url": (f"https://reactome.org/content/detail/{row['reactome_id']}" if row["reactome_id"]
                    else f"https://www.ebi.ac.uk/QuickGO/term/{row['go_ids'][0]}"),
            "confidence": row["confidence"],
            "gene_count": len(genes),
            "_genes": genes,
        })
    return pathways


def pathways_acted_on(drug_genes: set[str], pathways: list[dict]) -> list[dict]:
    """Which of the risk's pathways contain genes this drug acts on (broad pathways excluded)."""
    hits = []
    for p in pathways:
        if p["gene_count"] > MAX_GENES_FOR_MATCHING:
            continue
        overlap = sorted(drug_genes & p["_genes"])
        if overlap:
            hits.append({"name": p["name"], "url": p["url"], "genes": overlap})
    return hits
