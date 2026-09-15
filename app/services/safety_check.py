

import json
import logging
import re
import time

from ddgs import DDGS

from app.services.llm_service import call_llm

logger = logging.getLogger(__name__)

SAFETY_SYSTEM_PROMPT = """
You are a clinical safety gate for a longevity platform's investigational-
evidence list. You are given a compound name, a disease/condition, and a
handful of real web search results about that compound. Your only job is
to decide whether this compound should be shown to a doctor as a candidate
intervention for that disease.

CRITICAL RULES:
- Base your verdict ONLY on the search results given. Never use outside
  knowledge, and never invent a finding that isn't in the results.
- If the results show a clear, serious safety concern for this compound in
  this disease/population (e.g. a black-box warning, a stated
  contraindication, documented increased mortality risk) — verdict
  "exclude".
- If the results show a real but milder caution (e.g. "use with caution",
  "monitor closely", a manageable interaction) — verdict "warn".
- If the results show no safety concern for this compound in this context
  — verdict "ok".
- If the results are thin, unclear, or don't actually address this
  compound/disease combination — verdict "exclude". When you cannot
  confirm a compound is safe, the default is to leave it out, not include
  it with a caveat.
- Reply with ONLY a JSON object in this exact format, nothing else:
{"verdict": "exclude" | "warn" | "ok", "reason": "one short sentence"}
"""

_SAFETY_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_SAFETY_CACHE_TTL_SECONDS = 86400  # a day — this is a compound+disease fact, not a per-patient one


def _search_compound_safety(compound_name: str, disease_name: str) -> list[dict]:
    query = f"{compound_name} safety contraindication {disease_name}"
    try:
        results = DDGS().text(query, max_results=5)
        return [{"title": r.get("title"), "body": r.get("body"), "url": r.get("href")} for r in results]
    except Exception as e:
        logger.warning(f"safety_check | search failed for {compound_name!r} / {disease_name!r}: {e}")
        return []


def _judge_compound_safety(compound_name: str, disease_name: str, snippets: list[dict]) -> dict:
    if not snippets:
        return {"verdict": "unknown", "reason": "no search results available — safety check could not run"}

    prompt = f"""
Compound: {compound_name}
Disease/condition: {disease_name}

Search results:
{json.dumps(snippets, indent=2)}
"""
    try:
        response = call_llm(prompt, system_prompt=SAFETY_SYSTEM_PROMPT)
        match = re.search(r"\{.*\}", response, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        verdict = parsed.get("verdict")
        if verdict not in ("exclude", "warn", "ok"):
            return {"verdict": "unknown", "reason": "safety check returned an unrecognized verdict"}
        return {"verdict": verdict, "reason": parsed.get("reason", "")}
    except Exception as e:
        logger.error(f"safety_check | judgment failed for {compound_name!r} / {disease_name!r}: {e}")
        return {"verdict": "unknown", "reason": f"safety check failed to run: {e}"}


def check_compound_safety(compound_name: str, disease_name: str | None) -> dict:
    """Returns {"verdict": "exclude"|"warn"|"ok"|"unknown", "reason": str}."""
    if not disease_name:
        return {"verdict": "unknown", "reason": "no disease context given"}

    cache_key = (compound_name, disease_name)
    cached = _SAFETY_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _SAFETY_CACHE_TTL_SECONDS:
        return cached[1]

    snippets = _search_compound_safety(compound_name, disease_name)
    verdict = _judge_compound_safety(compound_name, disease_name, snippets)
    logger.info(f"safety_check | {compound_name!r} for {disease_name!r} -> {verdict}")
    _SAFETY_CACHE[cache_key] = (time.time(), verdict)
    return verdict
