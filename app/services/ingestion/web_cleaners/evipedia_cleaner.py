"""
Cleans Evipedia's evidence reviews (evipedia.ai) into biology-evidence
chunks, via the site's own structured data (reviews.json index, per-review
.md body, .meta.json citations). CC BY 4.0 licensed. Evidence tier is
inferred per-review (see _infer_evidence_tier), not assumed from the
site's general human-evidence framing — some reviews are animal-only.
"""

import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://evipedia.ai"
REVIEWS_INDEX_URL = f"{BASE_URL}/reviews.json"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ALIS-research-bot/1.0)"}
_REQUEST_DELAY_SECONDS = 0.3

# Allowlist, not a blocklist — excludes cosmetic/dental and borderline
# categories; a new site category defaults to excluded, not included.
_INCLUDED_CATEGORIES = {
    "compound", "botanical", "medication", "peptide", "animal", "cancer",
    "probiotic", "diet", "exercise", "mechanistic", "blood",
    "hormones_compound", "brain", "hormones_hormone", "senolytic",
    "therapy", "hormones_procedure", "targeted",
}

# Excludes boilerplate (Recommended Reading, Grokipedia, etc. — near-
# identical across reviews, would dilute the embedding) and low-value
# secondary detail.
_BODY_SECTIONS = [
    "Motivation", "Mechanism of Action", "Expected Benefits",
    "Potential Risks & Side Effects", "Therapeutic Protocol", "Conclusion",
]
_INTERACTIONS_SECTION = "Key Interactions & Contraindications"
_MAX_RAW_TEXT_CHARS = 20_000  # comfortably under the confirmed-working ~25k threshold

_ANIMAL_ONLY_MARKERS = [
    r"no human data", r"not (?:yet )?(?:been )?tested in humans?",
    r"no (?:direct )?evidence in humans?",
    r"only (?:in|from) (?:mice|rodents|animals|rats)\b",
    r"never (?:been )?tested in humans?", r"no human (?:trials?|studies|research)",
]
_ANIMAL_ONLY_PATTERN = re.compile("|".join(_ANIMAL_ONLY_MARKERS), re.IGNORECASE)


def _infer_evidence_tier(text: str) -> str:
    return "animal_model" if _ANIMAL_ONLY_PATTERN.search(text) else "human"


_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def _extract_section(md_text: str, heading: str) -> str | None:
    """Text of one '## Heading' section, up to the next '## ' heading (or end of doc)."""
    pattern = re.compile(
        r"^## " + re.escape(heading) + r"\s*\n(.*?)(?=\n## |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(md_text)
    if not match:
        return None
    cleaned = _HTML_COMMENT_PATTERN.sub("", match.group(1))
    return cleaned.strip() or None


def _fetch_reviews_index() -> list[dict]:
    resp = httpx.get(REVIEWS_INDEX_URL, headers=_HEADERS, timeout=30.0)
    resp.raise_for_status()
    reviews = resp.json()
    logger.info(f"Evipedia | reviews.json listed {len(reviews)} total reviews")
    return reviews


def load_evipedia_chunks(limit: int | None = None) -> list[dict]:
    """2 requests per review — slow for the full set. Per-review failures are skipped, not fatal."""
    all_reviews = _fetch_reviews_index()
    in_scope = [r for r in all_reviews if r.get("category") in _INCLUDED_CATEGORIES]
    logger.info(f"Evipedia | {len(in_scope)}/{len(all_reviews)} reviews in scope "
                f"after category filter")
    if limit:
        in_scope = in_scope[:limit]

    chunks = []
    for i, review in enumerate(in_scope):
        name = review.get("canonical_name")
        md_url = review.get("permalink_md")
        meta_url = review.get("permalink_meta")
        if not name or not md_url:
            continue

        try:
            md_resp = httpx.get(md_url, headers=_HEADERS, timeout=30.0)
            md_resp.raise_for_status()
            md_text = md_resp.text
        except Exception as e:
            logger.warning(f"Evipedia | failed to fetch body for {name}: {e}")
            continue

        pmids, nct_ids = [], []
        if meta_url:
            try:
                meta_resp = httpx.get(meta_url, headers=_HEADERS, timeout=30.0)
                meta_resp.raise_for_status()
                citations = meta_resp.json().get("citation", [])
                pmids = [c["pmid"] for c in citations if c.get("pmid")]
                nct_ids = [c["name"] for c in citations if c.get("name", "").startswith("NCT")]
            except Exception as e:
                logger.warning(f"Evipedia | failed to fetch citations for {name}: {e}")

        body_parts = []
        for heading in _BODY_SECTIONS:
            section = _extract_section(md_text, heading)
            if section:
                body_parts.append(f"{heading}: {section}")
        interactions = _extract_section(md_text, _INTERACTIONS_SECTION)

        if not body_parts:
            logger.warning(f"Evipedia | no recognized sections found for {name}, skipping")
            continue

        full_text_for_tier_check = "\n".join(body_parts) + "\n" + (interactions or "")
        evidence_tier = _infer_evidence_tier(full_text_for_tier_check)
        raw_text = f"Intervention: {name}. Evidence tier: {evidence_tier}.\n\n" + "\n\n".join(body_parts)
        if len(raw_text) > _MAX_RAW_TEXT_CHARS:
            # A few reviews get rejected outright by the embedder past this length.
            raw_text = raw_text[:_MAX_RAW_TEXT_CHARS].rsplit("\n\n", 1)[0]

        chunks.append({
            "type": "intervention",
            "name": name,
            "alternate_names": review.get("alternate_names", []),
            "category": review.get("category"),
            "source": "Evipedia",
            "evidence_tier": evidence_tier,
            "url": review.get("permalink"),
            "date_modified": review.get("dateModified"),
            "interactions_and_contraindications": interactions,
            "pmids": pmids,
            "nct_ids": nct_ids,
            "target_pc_groups": [],
            "target_conditions": [],
            "raw_text": raw_text,
        })

        if i % 25 == 0:
            logger.info(f"Evipedia | processed {i + 1}/{len(in_scope)} in-scope reviews")
        time.sleep(_REQUEST_DELAY_SECONDS)

    logger.info(f"Evipedia | loaded {len(chunks)} chunks from {len(in_scope)} in-scope review(s)")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = load_evipedia_chunks(limit=5)
    print(f"Loaded {len(result)} Evipedia chunks (test run, limited to 5)")
    print()
    if result:
        print(result[0])
