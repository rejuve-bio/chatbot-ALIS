"""
NCBI E-utilities (PubMed): citations for PubMed IDs, and recent human trials / meta-analyses for
an intervention in a disease — newer evidence than the stored Evipedia and DrugAge records.
"""

import datetime
import logging
import os
import threading
import time

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

_MIN_INTERVAL = 0.11 if NCBI_API_KEY else 0.35
_rate_lock = threading.Lock()
_last_call = [0.0]

_CITATIONS: dict[str, dict] = {}
_SEARCH_CACHE: dict[tuple, tuple[float, tuple[list[dict], int]]] = {}
_SEARCH_TTL_SECONDS = 86400

_HUMAN_EVIDENCE_TYPES = '(randomized controlled trial[pt] OR meta-analysis[pt] OR systematic review[pt]) AND humans[mh]'


def _get(endpoint: str, params: dict) -> dict:
    if NCBI_API_KEY:
        params = {**params, "api_key": NCBI_API_KEY}
    for attempt in range(5):
        with _rate_lock:
            wait = _MIN_INTERVAL - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        resp = httpx.get(f"{BASE_URL}/{endpoint}", params={**params, "retmode": "json"}, timeout=30.0)
        if resp.status_code != 429:
            break
        time.sleep(2 ** attempt)
    resp.raise_for_status()
    return resp.json()


def _summaries(pmids: list[str]) -> dict[str, dict]:
    missing = [p for p in pmids if p not in _CITATIONS]
    if missing:
        try:
            result = _get("esummary.fcgi", {"db": "pubmed", "id": ",".join(missing[:200])}).get("result", {})
        except Exception as e:
            logger.warning(f"NCBI | esummary failed for {len(missing)} id(s): {e}")
            result = {}
        for pmid in result.get("uids", []):
            r = result[pmid]
            _CITATIONS[pmid] = {
                "pmid": pmid,
                "title": r.get("title", "").rstrip("."),
                "journal": r.get("source"),
                "year": (r.get("pubdate") or "")[:4],
                "type": ", ".join(t for t in r.get("pubtype", []) if t != "Journal Article") or "Journal Article",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
    return {p: _CITATIONS[p] for p in pmids if p in _CITATIONS}


def get_citations(pmids: list[str], limit: int = 5) -> list[dict]:
    """Title, journal, year and PubMed link for the first `limit` IDs that resolve."""
    ids = list(dict.fromkeys(str(p).strip() for p in pmids if str(p).strip().isdigit()))[:limit]
    found = _summaries(ids) if ids else {}
    return [found[p] for p in ids if p in found]


_PATHWAY_CACHE: dict[tuple, tuple[float, list[dict]]] = {}


def _population_filter(sex: str | None) -> str:
    """No pregnancy and no child-only studies; for a male (female) patient, no study of women (men)
    only. Papers indexed for both sexes, or for neither, are kept."""
    q = " NOT pregnancy[mh] NOT (child[mh] NOT adult[mh])"
    if sex == "M":
        q += " NOT (female[mh] NOT male[mh])"
    elif sex == "F":
        q += " NOT (male[mh] NOT female[mh])"
    return q


def _disease_query(disease_terms: list[str], mesh: bool) -> str:
    """With MeSH headings: papers whose main topic is treating or preventing the disease. Without
    (a risk name that isn't a MeSH heading): the disease in the title."""
    if mesh:
        return " OR ".join(f'"{d}"[majr]' if "/" in d else f'"{d}/therapy"[majr] OR "{d}/prevention and control"[majr]'
                           for d in disease_terms)
    return " OR ".join(f"{_quote(d)}[ti]" for d in disease_terms)


def get_pathway_studies(pathway_term: str, disease_terms: list[str], years: int = 5, limit: int = 8,
                        sex: str | None = None, mesh: bool = False) -> list[dict]:
    """Newest human RCTs / meta-analyses / systematic reviews on this pathway in this disease, each
    with the substances PubMed indexes for it (its MeSH chemical list) — the interventions studied."""
    cache_key = (pathway_term, tuple(disease_terms), years, limit, sex, mesh)
    cached = _PATHWAY_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _SEARCH_TTL_SECONDS:
        return cached[1]

    disease_q = _disease_query(disease_terms, mesh)
    year = datetime.date.today().year
    try:
        ids = _get("esearch.fcgi", {
            "db": "pubmed",
            "term": f'"{pathway_term}"[tiab] AND ({disease_q}) AND {_HUMAN_EVIDENCE_TYPES}{_population_filter(sex)}',
            "sort": "pub_date", "retmax": limit, "datetype": "pdat",
            "mindate": str(year - years), "maxdate": str(year),
        }).get("esearchresult", {}).get("idlist", [])
        xml = _get_xml("efetch.fcgi", {"db": "pubmed", "id": ",".join(ids)}) if ids else None
    except Exception as e:
        logger.warning(f"NCBI | pathway search failed for {pathway_term!r}: {e}")
        return []

    studies = []
    for article in (xml.findall(".//PubmedArticle") if xml is not None else []):
        pmid = article.findtext(".//MedlineCitation/PMID")
        pub_types = [t.text for t in article.findall(".//PublicationTypeList/PublicationType") if t.text]
        year_text = (article.findtext(".//Journal/JournalIssue/PubDate/Year")
                     or (article.findtext(".//Journal/JournalIssue/PubDate/MedlineDate") or "")[:4])
        citation = {
            "pmid": pmid,
            "title": "".join(article.find(".//ArticleTitle").itertext()).rstrip(".") if article.find(".//ArticleTitle") is not None else "",
            "journal": article.findtext(".//Journal/ISOAbbreviation") or article.findtext(".//Journal/Title"),
            "year": year_text,
            "type": ", ".join(t for t in pub_types if t != "Journal Article") or "Journal Article",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
        _CITATIONS[pmid] = citation
        studies.append({**citation, "substances": [c.text for c in article.findall(".//ChemicalList/Chemical/NameOfSubstance") if c.text]})
    _PATHWAY_CACHE[cache_key] = (time.time(), studies)
    return studies


def _get_xml(endpoint: str, params: dict):
    import xml.etree.ElementTree as ET
    if NCBI_API_KEY:
        params = {**params, "api_key": NCBI_API_KEY}
    for attempt in range(5):
        with _rate_lock:
            wait = _MIN_INTERVAL - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        resp = httpx.get(f"{BASE_URL}/{endpoint}", params={**params, "retmode": "xml"}, timeout=60.0)
        if resp.status_code != 429:
            break
        time.sleep(2 ** attempt)
    resp.raise_for_status()
    return ET.fromstring(resp.text)


def _quote(term: str) -> str:
    return f'"{term}"' if " " in term or "-" in term else term


def get_recent_human_studies(
    interventions: list[str], disease_terms: list[str], years: int = 3, limit: int = 3,
    sex: str | None = None, mesh: bool = False,
) -> tuple[list[dict], int]:
    """Newest randomized trials, meta-analyses and systematic reviews with the intervention in the
    title and the disease as a MeSH heading (or in title/abstract when it has no MeSH heading),
    from the last `years` years. Returns (studies, total count)."""
    names = [n for n in interventions if n][:8]
    if not names or not disease_terms:
        return [], 0
    cache_key = (tuple(names), tuple(disease_terms), years, limit, sex, mesh)
    cached = _SEARCH_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _SEARCH_TTL_SECONDS:
        return cached[1]

    drug_q = " OR ".join(f"{_quote(n)}[ti]" for n in names)
    disease_q = _disease_query(disease_terms, mesh)
    year = datetime.date.today().year
    try:
        found = _get("esearch.fcgi", {
            "db": "pubmed", "term": f"({drug_q}) AND ({disease_q}) AND {_HUMAN_EVIDENCE_TYPES}{_population_filter(sex)}",
            "sort": "pub_date", "retmax": limit, "datetype": "pdat",
            "mindate": str(year - years), "maxdate": str(year),
        }).get("esearchresult", {})
    except Exception as e:
        logger.warning(f"NCBI | esearch failed for {names}: {e}")
        return [], 0
    ids = found.get("idlist", [])
    studies = list(_summaries(ids).values()) if ids else []
    result = (studies, int(found.get("count", 0)))
    _SEARCH_CACHE[cache_key] = (time.time(), result)
    return result


_ABSTRACTS: dict[str, str] = {}


def get_abstracts(pmids: list[str]) -> dict[str, str]:
    """Abstract text for each PubMed ID (empty when the paper has none); raises when PubMed fails."""
    missing = [p for p in pmids if p and p not in _ABSTRACTS]
    if missing:
        root = _get_xml("efetch.fcgi", {"db": "pubmed", "id": ",".join(missing), "rettype": "abstract"})
        for article in root.iter("PubmedArticle"):
            pmid = article.findtext(".//PMID")
            _ABSTRACTS[pmid] = " ".join("".join(t.itertext()) for t in article.iter("AbstractText"))
        for p in missing:
            _ABSTRACTS.setdefault(p, "")
    return {p: _ABSTRACTS.get(p, "") for p in pmids}
