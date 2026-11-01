"""
Body systems of a disease from its place in NLM's MeSH tree (used by the UI avatar).

A disease's MeSH tree numbers (e.g. Hypertension -> C14.907.489) start with the branch of the
body system it belongs to (C14 = Cardiovascular Diseases). MeSH files many diseases in several
branches (Stroke: C10 nervous system + C14 cardiovascular; Lung Neoplasms: C04 neoplasms + C08
respiratory), so a disease gets every system it is filed under, most-filed first, at most 3.
"""

import logging
import threading
import time
from collections import Counter

import httpx

logger = logging.getLogger(__name__)

LOOKUP_URL = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
DESCRIPTOR_URL = "https://id.nlm.nih.gov/mesh/{}.json"
UNLOCALIZED = "unlocalized"

_BRANCH_SYSTEM = (
    ("C06.552", "hepatic"),
    ("C06.130", "hepatic"),
    ("C06", "digestive"),
    ("C14", "cardiovascular"),
    ("C08", "respiratory"),
    ("C10", "neurological"),
    ("F03", "neurological"),
    ("C12.950", "renal"),
    ("C12.050.351.968", "renal"),
    ("C12.200.777", "renal"),
    ("C12.100", "reproductive"),
    ("C12.050.351.500", "reproductive"),
    ("C12.050.703", "reproductive"),
    ("C12.200.294", "reproductive"),
    ("C05", "musculoskeletal"),
    ("C17.800", "skin"),
    ("C20", "immune"),
    ("C18", "metabolic"),
    ("C19", "metabolic"),
    ("C11", "vision"),
    ("C09.218", "hearing"),
)
_SYSTEM_ORDER = ["cardiovascular", "respiratory", "neurological", "metabolic", "hepatic", "digestive", "renal",
                 "musculoskeletal", "reproductive", "immune", "vision", "hearing", "skin"]
_MAX_SYSTEMS = 3

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL_SECONDS = 30 * 86400
_cache_lock = threading.Lock()


def _cached(key: str, fetch):
    with _cache_lock:
        cached = _CACHE.get(key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]
    try:
        value = fetch()
    except Exception as e:
        logger.warning(f"MeSH | lookup {key!r} failed: {e}")
        return None
    with _cache_lock:
        _CACHE[key] = (time.time(), value)
    return value


def get_descriptor_id(heading: str) -> str | None:
    """MeSH descriptor ID for an exact heading, e.g. "Hypertension" -> "D006973"."""
    def fetch():
        resp = httpx.get(LOOKUP_URL, params={"label": heading, "match": "exact", "limit": 1}, timeout=20.0)
        resp.raise_for_status()
        hits = resp.json()
        return hits[0]["resource"].rsplit("/", 1)[-1] if hits else None
    return _cached(f"id:{heading.lower()}", fetch)


def get_tree_numbers(descriptor_id: str) -> list[str]:
    """The descriptor's positions in the MeSH tree, e.g. ["C14.907.489"]."""
    def fetch():
        resp = httpx.get(DESCRIPTOR_URL.format(descriptor_id), timeout=20.0)
        resp.raise_for_status()
        trees = resp.json().get("treeNumber", [])
        trees = trees if isinstance(trees, list) else [trees]
        return [t.rsplit("/", 1)[-1] for t in trees]
    return _cached(f"tree:{descriptor_id}", fetch) or []


def _system_of(tree_number: str) -> str | None:
    return next((system for prefix, system in _BRANCH_SYSTEM if tree_number.startswith(prefix)), None)


def get_body_systems(mesh_headings: list[str]) -> list[str]:
    """The avatar body systems for a disease, from the MeSH tree numbers of its headings: the
    systems filed under by the most headings first, then by the most tree numbers, then deepest.
    ["unlocalized"] when MeSH files it under no organ system (e.g. "Cancer" without a site)."""
    descriptors = dict.fromkeys(d for h in mesh_headings for d in [get_descriptor_id(h)] if d)
    headings, trees, depth = Counter(), Counter(), {}
    for descriptor in descriptors:
        found = set()
        for tree in get_tree_numbers(descriptor):
            system = _system_of(tree)
            if system:
                found.add(system)
                trees[system] += 1
                depth[system] = max(depth.get(system, 0), tree.count("."))
        headings.update(found)
    if not trees:
        return [UNLOCALIZED]
    ranked = sorted(trees, key=lambda s: (-headings[s], -trees[s], -depth[s], _SYSTEM_ORDER.index(s)))
    return ranked[:_MAX_SYSTEMS]
