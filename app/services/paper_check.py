"""
Checks a PubMed paper before it counts as evidence: the item must be given to treat or prevent
the disease, and the paper must not report that it failed. A paper about the item causing or
worsening the disease (a risk factor, an exposure, a toxin), or about another disease, is rejected.

The LLM reads the title and abstract and may only reject; it never adds an item. Verdicts are
stored per (PMID, item, disease) with the reason, so each paper is checked once and can be audited.
"""

import json
import logging
import re

from app.services import atlas_store
from app.services.llm_service import call_llm
from app.services.ncbi_service import get_abstracts

logger = logging.getLogger(__name__)

CHECK_SYSTEM_PROMPT = """
You check whether a medical paper is evidence that an intervention TREATS or PREVENTS a disease in
the patient described. Read the title and abstract and answer every question, in JSON only:
{"population_has_disease": true|false,
 "human_clinical_evidence": true|false,
 "intervention_is_tested_treatment": true|false,
 "main_outcome_is_disease": true|false,
 "population_fits_patient": true|false,
 "benefit": "yes"|"no"|"unclear",
 "reason": "<one short sentence>"}

- population_has_disease: the people studied have this disease, or are at risk of it in a
  prevention trial. False for healthy volunteers, and for a different disease, even one with a
  similar name (pulmonary hypertension is not hypertension).
- human_clinical_evidence: a human clinical trial, or a systematic review / meta-analysis of human
  trials, that reports results. False for a trial protocol without results, a cost-effectiveness
  model, and a review that is mostly mechanistic or animal / cell studies.
- intervention_is_tested_treatment: the intervention is given to the patients as the treatment
  being tested (alone or as the tested part of a combination). False when it is the comparator,
  background therapy given to every group, a coating on a device, or a risk factor, exposure or toxin.
- main_outcome_is_disease: the main outcome measures this disease itself (its control, progression
  or events). False when the main outcome is something else (uric acid, imaging of another organ,
  muscle mass, pain after surgery) and the disease is only mentioned.
- population_fits_patient: false for pregnant women, children, a population of only the other
  sex (only women for a male patient, only men for a female patient), or a population defined by a
  rare inherited high-risk condition (Lynch syndrome, BRCA mutation carriers, familial adenomatous
  polyposis) when the disease asked about is not that condition.
- benefit: "yes" only when the paper reports the intervention helped this disease.
"""


_NOT_EFFICACY = re.compile(r"cost[- ]effective|cost[- ]utility|economic evaluation|budget impact|\bprotocol\b"
                           r"|study design|rationale and design|preclinical|mechanis(m|tic)", re.IGNORECASE)
_DEVICE = re.compile(r"drug[- ]coated|[a-z]+-coated balloon|coated balloon|eluting|stent", re.IGNORECASE)
_CHECKLIST = ("population_has_disease", "human_clinical_evidence", "intervention_is_tested_treatment",
              "main_outcome_is_disease", "population_fits_patient")


class PaperCheckUnavailable(Exception):
    """The paper couldn't be checked (PubMed or the LLM unreachable); don't store the result."""


def _parse(content: str) -> dict:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in LLM answer: {content[:200]!r}")
    return json.loads(match.group(0))


def treats_disease(paper: dict, item: str, disease: str, sex: str | None = None) -> bool:
    """True when this paper tests `item` as a treatment for `disease` and doesn't report failure."""
    pmid = paper.get("pmid")
    if not pmid:
        return False
    if _NOT_EFFICACY.search(paper.get("title") or "") or _DEVICE.search(paper.get("title") or ""):
        return False
    patient = {"M": "an adult man", "F": "an adult woman"}.get(sex, "an adult")
    disease_key = f"c3|{disease}|{sex or '-'}"
    stored = atlas_store.get_paper_verdict(pmid, item, disease_key)
    if stored is not None:
        return stored
    try:
        abstract = get_abstracts([pmid]).get(pmid, "")
        answer = _parse(call_llm(
            f"Intervention: {item}\nDisease: {disease}\nPatient: {patient}\n\n"
            f"Title: {paper.get('title')}\n\nAbstract: {abstract or '(none)'}",
            system_prompt=CHECK_SYSTEM_PROMPT,
        ))
    except Exception as e:
        logger.warning(f"paper_check | PMID {pmid} ({item} / {disease}) couldn't be checked: {e}")
        raise PaperCheckUnavailable(pmid) from e
    verdict = all(answer.get(k) is True for k in _CHECKLIST) and answer.get("benefit") == "yes"
    atlas_store.put_paper_verdict(pmid, item, disease_key, verdict, answer.get("reason"))
    logger.info(f"paper_check | PMID {pmid} | {item} / {disease} | {'kept' if verdict else 'rejected'}: {answer.get('reason')}")
    return verdict


BENEFIT_SYSTEM_PROMPT = """
You match the benefits an evidence review lists for an intervention to one disease.
Answer in JSON only: {"benefit": "<one benefit title exactly as listed, or null>", "reason": "<one short sentence>"}
Pick the benefit that is about treating, preventing or slowing this disease itself. Answer null
when none is: a benefit for a different condition, a general effect, or lifespan does not count.
When the disease is aging itself ("Aging"), a benefit about lifespan, healthspan or slowing aging
does count.
"""
_LEVEL = re.compile(r"^###\s+(High|Medium|Low|Speculative)\b")
_TITLE = re.compile(r"^####\s+(.+?)\s*$")


def evipedia_benefits(raw_text: str) -> list[tuple[str, str]]:
    """(level, title) of every benefit in the review's Expected Benefits section; risks are ignored."""
    section = raw_text.split("Expected Benefits:", 1)[1] if "Expected Benefits:" in raw_text else ""
    section = section.split("Potential Risks", 1)[0]
    benefits, level = [], None
    for line in section.splitlines():
        if m := _LEVEL.match(line.strip()):
            level = m.group(1)
        elif (m := _TITLE.match(line.strip())) and level:
            benefits.append((level, m.group(1)))
    return benefits


def evipedia_benefit(compound: str, review: dict, disease: str) -> dict | None:
    """The benefit the Evipedia review lists for this disease, as {"level", "title"}, or None."""
    benefits = evipedia_benefits(review.get("raw_text") or "")
    if not benefits:
        return None
    key = f"evipedia2:{compound.lower()}"
    stored = atlas_store.get_paper_record(key, compound, disease)
    if stored is None:
        listed = "\n".join(f"- [{level}] {title}" for level, title in benefits)
        try:
            answer = _parse(call_llm(f"Intervention: {compound}\nDisease: {disease}\n\nBenefits listed:\n{listed}",
                                     system_prompt=BENEFIT_SYSTEM_PROMPT))
        except Exception as e:
            logger.warning(f"paper_check | Evipedia benefits of {compound} / {disease} couldn't be matched: {e}")
            raise PaperCheckUnavailable(key) from e
        match = next(((lv, t) for lv, t in benefits if t == answer.get("benefit")), None)
        reason = f"{match[0]}|{match[1]}" if match else (answer.get("reason") or "no matching benefit")
        atlas_store.put_paper_verdict(key, compound, disease, bool(match), reason)
        stored = (bool(match), reason)
        logger.info(f"paper_check | Evipedia {compound} / {disease}: {reason}")
    if not stored[0]:
        return None
    level, title = stored[1].split("|", 1)
    return {"level": level, "title": title}
