import logging
import os
import re
import threading
import time

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY")

_LABEL_FIELDS = ("indications_and_usage", "contraindications", "boxed_warning", "warnings_and_cautions", "warnings")

_LABEL_CACHE: dict[str, tuple[float, dict | None]] = {}
_LABEL_CACHE_TTL_SECONDS = 7 * 86400
_cache_lock = threading.Lock()


def _single_ingredient_labels(results: list[dict], name: str) -> list[tuple[int, dict]]:
    """openFDA's generic_name search also returns combination products (e.g. "DICLOFENAC SODIUM AND
    MISOPROSTOL"), whose indications belong to the combination; only single-ingredient labels count."""
    name_upper = name.upper()
    single = []
    for r in results:
        for generic in r.get("openfda", {}).get("generic_name", []):
            g = generic.upper()
            if name_upper in g and " AND " not in g and "," not in g and "/" not in g and ";" not in g:
                single.append((len(generic), r))
                break
    single.sort(key=lambda s: s[0])
    return single


def _pick_single_ingredient(results: list[dict], name: str) -> dict | None:
    single = _single_ingredient_labels(results, name)
    if not single:
        return None
    return next((r for _, r in single if r.get("boxed_warning")), single[0][1])


LOOKUP_FAILED = {"lookup_failed": True}


def get_drug_label(drug_name: str) -> dict | None:
    """The FDA label's indication, contraindication and warning text for a drug, or None when the
    name isn't an FDA-labelled drug (supplements, research compounds, drug classes). Returns
    LOOKUP_FAILED when openFDA couldn't be reached, so callers never mistake "unknown" for "no label"."""
    key = drug_name.strip().lower().replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-")
    with _cache_lock:
        cached = _LABEL_CACHE.get(key)
    if cached and (time.time() - cached[0]) < _LABEL_CACHE_TTL_SECONDS:
        return cached[1]

    params = {"search": f'openfda.generic_name:"{key.replace(chr(34), "")}"', "limit": 25}
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY
    try:
        resp = httpx.get(BASE_URL, params=params, timeout=15.0)
        if resp.status_code in (400, 404):
            label = None
        else:
            resp.raise_for_status()
            body = resp.json()
            results = body.get("results", [])
            picked = _pick_single_ingredient(results, key)
            label = None if picked is None else {
                "generic_name": picked["openfda"]["generic_name"][0],
                "label_count": body.get("meta", {}).get("results", {}).get("total", 0),
                "route": [r.lower() for r in picked["openfda"].get("route", [])],
                **{f: " ".join(picked.get(f, [])) for f in _LABEL_FIELDS},
                "all_indications": [" ".join(r.get("indications_and_usage", []))
                                    for _, r in _single_ingredient_labels(results, key)],
            }
    except Exception as e:
        logger.warning(f"openFDA | label lookup failed for {drug_name!r}: {e}")
        return LOOKUP_FAILED

    logger.info(f"openFDA | {drug_name!r} -> {label['generic_name'] if label else 'no label'}")
    with _cache_lock:
        _LABEL_CACHE[key] = (time.time(), label)
    return label


_MARKETED_CACHE: dict[str, bool | None] = {}


def is_marketed(drug_name: str) -> bool | None:
    """False when FDA lists products for this ingredient but none is currently sold (all
    discontinued); True when at least one is; None when FDA has no record (supplements, research
    compounds) or the lookup failed."""
    key = drug_name.strip().lower()
    if key in _MARKETED_CACHE:
        return _MARKETED_CACHE[key]
    params = {"search": f'products.active_ingredients.name:"{key.upper().replace(chr(34), "")}"', "limit": 100}
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY
    try:
        resp = httpx.get("https://api.fda.gov/drug/drugsfda.json", params=params, timeout=15.0)
        if resp.status_code in (400, 404):
            marketed = None
        else:
            resp.raise_for_status()
            statuses = [p.get("marketing_status") for r in resp.json().get("results", []) for p in r.get("products", [])
                        if any(key.upper() in (a.get("name") or "").upper() for a in p.get("active_ingredients", []))]
            marketed = None if not statuses else any(s in ("Prescription", "Over-the-counter") for s in statuses)
    except Exception as e:
        logger.warning(f"openFDA | marketing status lookup failed for {drug_name!r}: {e}")
        return None
    _MARKETED_CACHE[key] = marketed
    return marketed


_RXNORM_CACHE: dict[str, str | None] = {}


def rxnorm_ingredient_name(name: str) -> str | None:
    """The RxNorm ingredient name for a drug synonym (e.g. rapamycin -> sirolimus), via NLM RxNav."""
    key = name.strip().lower()
    if key in _RXNORM_CACHE:
        return _RXNORM_CACHE[key]
    ingredient = None
    try:
        ids = httpx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json", params={"name": key, "search": 2},
                        timeout=15.0).json().get("idGroup", {}).get("rxnormId", [])
        if ids:
            props = httpx.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{ids[0]}/properties.json",
                              timeout=15.0).json().get("properties", {})
            if props.get("tty") == "IN":
                ingredient = props.get("name", "").lower() or None
    except Exception as e:
        logger.warning(f"RxNorm | name lookup failed for {name!r}: {e}")
        return None
    _RXNORM_CACHE[key] = ingredient
    return ingredient


_SENTENCE_SPLIT = re.compile(r"(?<=[.;•])\s+|\(\s*\d+(?:\.\d+)?\s*\)")
_TREATS = re.compile(r"\b(treat|management|improve|reduc|slow|therap|replacement|control|relie[fv])", re.IGNORECASE)
_NEGATED = re.compile(r"\bnot (been )?(indicated|established|shown|recommended)\b|\b(did|does) not\b|not (to be )?effective|ineffective|not (been )?demonstrated|limitations? of use"
                      r"|\b(imaging|diagnos)|secondary causes?|should be (excluded|ruled out)|medical problems such as|contributing to"
                      r"|independent of|\(for example", re.IGNORECASE)


def mentions_term(text: str, term: str) -> bool:
    """The term as whole words in lowercase text: "pneumonia" is not in "streptococcus pneumoniae"."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def is_indicated_for(label: dict | None, disease_terms: list[str], exclude_terms: list[str] = ()) -> bool:
    """True when a sentence of the label's indications says the drug treats one of these terms.
    Sentence-level so "not indicated for hypertension" or "exclude hypothyroidism" don't count;
    exclude_terms drop sentences about a different condition with the same word (e.g. "pulmonary
    arterial hypertension")."""
    if not label or label is LOOKUP_FAILED or not disease_terms:
        return False
    texts = label.get("all_indications") or [label.get("indications_and_usage", "")]
    for sentence in (s for text in texts for s in _SENTENCE_SPLIT.split(text.lower())):
        if _NEGATED.search(sentence) or not _TREATS.search(sentence):
            continue
        if any(x in sentence for x in exclude_terms):
            continue
        if any(mentions_term(sentence, term) for term in disease_terms):
            return True
    return False


_BOXED_PREFIX = re.compile(r"^(boxed warning\s*)?(warnings?\s*:?\s*)", re.IGNORECASE)
_BOXED_END = re.compile(r"\s+•|\s+see full prescribing|[.;]|\s+(?=[A-Z][a-z]+\s+[a-z])")


def boxed_warning_title(label: dict | None) -> str | None:
    """The heading of the label's boxed warning, e.g. "Risk of thyroid c-cell tumors"."""
    if not label or label is LOOKUP_FAILED:
        return None
    text = _BOXED_PREFIX.sub("", (label.get("boxed_warning") or "").strip())
    if not text:
        return None
    heading = _BOXED_END.split(text, maxsplit=1)[0].strip(" ,:")
    heading = re.split(r"\s+warnings?\s*:", heading, maxsplit=1, flags=re.IGNORECASE)[0]
    cut = len(heading) > 90
    if cut:
        heading = heading[:90].rsplit(" ", 1)[0].rstrip(" ,;:")
    return heading[:1].upper() + heading[1:].lower() + ("…" if cut else "")
