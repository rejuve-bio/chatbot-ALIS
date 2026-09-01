import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from app.services.biology_service import (
    search_investigational_compounds, get_human_evidence_for_compound, search_human_evidence,
    get_target_genes_for_drug, get_gene_grounding,
)
from app.services.clinicaltrials_service import get_active_trials_for_compound
from app.services.llm_service import call_llm

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """
You are drafting a short clinical framing note for the "Investigational,
preclinical evidence" section of a longevity platform.

CRITICAL RULES:
- Use ONLY the compounds, genes, and facts given to you. Never introduce a
  compound, gene, or claim that isn't in the data provided.
- Never say a compound is "proven", "effective", or "safe" in humans if its
  evidence_tier is "animal_model" — always keep animal evidence framed as
  animal evidence.
- Write 2-3 plain sentences, no headers, no bullet points, no markdown.
"""

HUMAN_EVIDENCE_SYSTEM_PROMPT = """
You are summarizing the human-evidence picture for a list of compounds
that come from two different sources: some (source: DrugAge) were only
ever studied in animals and have no direct human-evidence review; others
(source: Evipedia) already have a direct, real human-evidence review.

CRITICAL RULES:
- Cover EVERY entry in the list given to you — not just one or a few.
- If has_human_review is true (source is Evipedia), say so plainly and
  summarize what its interactions_and_contraindications field says — do
  NOT claim "no human evidence exists" for these, that would be false.
- If has_human_review is false (source is DrugAge only), state plainly
  that it remains animal-only evidence with no human-evidence review.
- Use ONLY the findings given to you. Never invent a compound or a claim.
- Never upgrade animal-only evidence to "proven in humans."
- Write 2-4 plain sentences, no headers, no bullet points, no markdown.
"""

NOTE_SYSTEM_PROMPT = """
You are writing one short clinical note per compound, for a structured
JSON record in an "Investigational, preclinical evidence" list — NOT
prose for a reader, a single caveat sentence attached to that compound's
own data.

CRITICAL RULES:
- Use ONLY the facts given for that specific compound. Never invent
  anything, and never describe a different compound's evidence.
- Every compound's evidence_tier is "animal_model" — always say so plainly.
  If evipedia_found is true for that compound, also mention that a separate
  human-evidence review exists for it; if false, say no human-evidence
  review exists and it remains animal-only.
- One short sentence per compound. No headers, no markdown, no bullet points
  inside a note.
- Reply with ONLY a JSON object in this exact format, nothing else:
{"notes": [{"compound_name": "...", "note": "..."}, ...]}
- Include every compound you were given, in the same order, exactly once.
"""


def get_mechanisms_for_pc_group(pc_group: str) -> tuple[list[str], list[str]]:
    """"Indeterminate" is dropped — it means no real mechanism was assignable."""
    from data.pc_chunks import PC_CHUNKS
    mechanisms: list[str] = []
    diseases: list[str] = []
    for chunk in PC_CHUNKS:
        if chunk.get("pc_group") != pc_group:
            continue
        for m in chunk.get("mechanisms", []):
            if m not in mechanisms and m != "Indeterminate":
                mechanisms.append(m)
        for d in chunk.get("diseases", []):
            if d not in diseases:
                diseases.append(d)
    logger.info(f"get_mechanisms_for_pc_group | pc_group={pc_group} | mechanisms={mechanisms} | "
                f"diseases_count={len(diseases)}")
    return mechanisms, diseases


def _trim_trial(t: dict) -> dict:
    trimmed = {"nct_id": t.get("nct_id"), "title": t.get("title"), "status": t.get("status")}
    phase = t.get("phase")
    if phase:
        trimmed["phase"] = phase
    return trimmed


def _gene_grounding_for_compound(compound_name: str) -> list[dict]:
    target_genes = get_target_genes_for_drug(compound_name)
    grounding = []
    for g in target_genes:
        entry = get_gene_grounding(g["gene"])
        entry["targeted_by_drug"] = {"source": "ClinPGx", "pmids": g.get("pmids")}
        grounding.append(entry)
    return grounding


def _merge_evidence_lists(
    compounds: list[dict], human_evidence_direct: list[dict], disease_name: str | None
) -> list[dict]:
    by_name: dict[str, dict] = {}

    for c in compounds:
        by_name[c["compound_name"]] = {
            "name": c["compound_name"],
            "source": c.get("source", "DrugAge"),
            "evidence_tier": c.get("evidence_tier", "animal_model"),
            "species": c.get("species"),
            "note": c.get("note"),
            "interactions_and_contraindications": None,
            "url": None,
            "pmids": [],
            "gene_level_grounding": c.get("gene_level_grounding", []),
            "active_trials": c.get("active_trials", []),
        }

    new_evipedia = [h for h in human_evidence_direct if h["intervention_name"] not in by_name]
    with ThreadPoolExecutor(max_workers=max(2 * len(new_evipedia), 1)) as executor:
        gene_futures = {
            h["intervention_name"]: executor.submit(_gene_grounding_for_compound, h["intervention_name"])
            for h in new_evipedia
        }
        trial_futures = {
            h["intervention_name"]: executor.submit(
                get_active_trials_for_compound, h["intervention_name"], condition=disease_name
            )
            for h in new_evipedia
        }
        for h in new_evipedia:
            name = h["intervention_name"]
            by_name[name] = {
                "name": name,
                "source": "Evipedia",
                "evidence_tier": h.get("evidence_tier"),
                "species": None,
                "note": (
                    "Direct human-evidence review on Evipedia."
                    if h.get("evidence_tier") == "human" else
                    "Animal-model evidence only, from Evipedia."
                ),
                "interactions_and_contraindications": h.get("interactions_and_contraindications"),
                "url": h.get("url"),
                "pmids": h.get("pmids", []),
                "gene_level_grounding": gene_futures[name].result(),
                "active_trials": trial_futures[name].result(),
            }

    for h in human_evidence_direct:
        name = h["intervention_name"]
        if name in by_name and by_name[name]["source"] == "DrugAge":
            entry = by_name[name]
            entry["url"] = entry["url"] or h.get("url")
            entry["pmids"] = entry["pmids"] or h.get("pmids", [])
            entry["interactions_and_contraindications"] = (
                entry["interactions_and_contraindications"] or h.get("interactions_and_contraindications")
            )

    merged = list(by_name.values())
    for entry in merged:
        count = 1
        for gene in entry["gene_level_grounding"]:
            count += 1
            count += sum(
                1 for field in ("human_aging_link", "model_organism_link",
                                 "senescence_effect", "senescence_expression")
                if gene.get(field) is not None
            )
        if entry["evidence_tier"] == "human":
            count += 2
        count += len(entry["active_trials"])
        entry["evidence_count"] = count
        entry["active_trials"] = [_trim_trial(t) for t in entry["active_trials"][:2]]

    merged.sort(key=lambda e: e["evidence_count"], reverse=True)
    logger.info(f"_merge_evidence_lists | merged {len(merged)} entries | "
                f"order={[(e['name'], e['evidence_count']) for e in merged]}")
    return merged


def _explain_pc_groups(pc_group: str | None) -> list[dict]:
    if not pc_group:
        return []
    codes = [c.strip() for c in pc_group.split(",") if c.strip()]
    explained = []
    for code in codes:
        _, diseases = get_mechanisms_for_pc_group(code)
        label = (
            f"A biomarker pattern associated with: {', '.join(diseases[:3])}"
            if diseases else
            "A biomarker pattern combining several related lab/vital measurements."
        )
        explained.append({"code": code, "label": label})
    return explained


def build_atlas_response(
    mechanism_tags: list[str],
    pc_group: str | None = None,
    disease_name: str | None = None,
) -> dict:
    logger.info(f"build_atlas_response | pc_group={pc_group} | disease={disease_name} | "
                f"mechanisms={mechanism_tags}")

    _background_executors: list[ThreadPoolExecutor] = []
    try:
        return _build_atlas_response_inner(
            mechanism_tags, pc_group, disease_name, _background_executors
        )
    finally:
        for ex in _background_executors:
            ex.shutdown(wait=False)


def _build_atlas_response_inner(
    mechanism_tags: list[str],
    pc_group: str | None,
    disease_name: str | None,
    _background_executors: list,
) -> dict:
    background_executor = ThreadPoolExecutor(max_workers=1)
    _background_executors.append(background_executor)
    human_evidence_future = background_executor.submit(
        search_human_evidence, mechanism_tags, disease_name=disease_name
    )

    compounds = search_investigational_compounds(mechanism_tags, disease_name=disease_name)
    logger.info(f"build_atlas_response | assembled {len(compounds)} candidate compound(s) "
                f"before LLM synthesis")

    if not compounds:
        logger.warning(f"build_atlas_response | no investigational compounds found for "
                        f"mechanisms={mechanism_tags} — still checking Evipedia directly "
                        f"before giving up")
        human_evidence_direct = human_evidence_future.result()
        return {
            "pc_group": pc_group,
            "pc_groups_explained": _explain_pc_groups(pc_group),
            "disease_name": disease_name,
            "mechanisms": mechanism_tags,
            "evidence": _merge_evidence_lists([], human_evidence_direct, disease_name),
            "summary": "No investigational preclinical evidence was found for these mechanisms.",
        }

    compounds = [dict(c) for c in compounds]

    background_executor2 = ThreadPoolExecutor(max_workers=max(2 * len(compounds), 1))
    _background_executors.append(background_executor2)
    trial_futures = {
        c["compound_name"]: background_executor2.submit(
            get_active_trials_for_compound, c["compound_name"], condition=disease_name
        )
        for c in compounds
    }
    evipedia_futures = {
        c["compound_name"]: background_executor2.submit(get_human_evidence_for_compound, c["compound_name"])
        for c in compounds
    }
    for c in compounds:
        evipedia_hit = evipedia_futures[c["compound_name"]].result()
        c["evipedia_found"] = evipedia_hit is not None
        c["evipedia_evidence_tier"] = evipedia_hit.get("evidence_tier") if evipedia_hit else None
        c["evipedia_summary"] = evipedia_hit.get("raw_text", "")[:600] if evipedia_hit else None

    logger.info(f"build_atlas_response | evipedia_found: "
                f"{[c['compound_name'] for c in compounds if c['evipedia_found']]}")

    note_prompt = f"""
Write the note for each of these compounds, using ONLY the facts given:
{json.dumps([{"compound_name": c["compound_name"], "evidence_tier": c["evidence_tier"],
              "evipedia_found": c["evipedia_found"],
              "evipedia_evidence_tier": c["evipedia_evidence_tier"]} for c in compounds], indent=2)}
"""
    logger.info(f"build_atlas_response | calling LLM for per-compound notes")
    notes_by_compound = {}
    try:
        notes_response = call_llm(note_prompt, system_prompt=NOTE_SYSTEM_PROMPT)
        match = re.search(r"\{.*\}", notes_response, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {"notes": []}
        notes_by_compound = {n["compound_name"]: n["note"] for n in parsed.get("notes", [])}
        logger.info(f"build_atlas_response | notes received for: {list(notes_by_compound.keys())}")
    except Exception as e:
        logger.error(f"build_atlas_response | note generation failed, falling back to a plain "
                     f"fact-based note per compound: {e}")
    for c in compounds:
        c["note"] = notes_by_compound.get(c["compound_name"]) or (
            "Human evidence review exists separately for this compound (see evipedia_summary)."
            if c["evipedia_found"] else
            "Not validated in humans."
        )

    for c in compounds:
        c["active_trials"] = trial_futures[c["compound_name"]].result()
    logger.info(f"build_atlas_response | active_trials found for: "
                f"{[c['compound_name'] for c in compounds if c['active_trials']]}")

    human_evidence_direct = human_evidence_future.result()
    logger.info(f"build_atlas_response | direct Evipedia search found: "
                f"{[h['intervention_name'] for h in human_evidence_direct]}")

    evidence = _merge_evidence_lists(compounds, human_evidence_direct, disease_name)
    logger.info(f"build_atlas_response | merged evidence ranked: "
                f"{[(e['name'], e['evidence_count']) for e in evidence]}")

    prompt = f"""
Mechanism tags for this risk area: {', '.join(mechanism_tags)}
{f"Disease/risk area: {disease_name}" if disease_name else ""}

Real evidence already gathered, already ranked strongest-to-weakest by
evidence_count (do not add anything beyond this, and do not re-rank it):
{json.dumps(evidence, indent=2)}

Write the 2-3 sentence clinical framing note now.
"""
    human_prompt = f"""
All evidence identified for this risk area, ranked strongest-to-weakest by
evidence_count, tagged by source (DrugAge = animal-model only, Evipedia =
has a direct human-evidence review) and evidence_tier ("human" or
"animal_model"):
{json.dumps([{"name": e["name"], "source": e["source"], "evidence_tier": e["evidence_tier"],
              "has_human_review": e["source"] == "Evipedia",
              "interactions_and_contraindications": e["interactions_and_contraindications"]}
             for e in evidence], indent=2)}

Write the human-evidence summary now, covering ALL entries listed above —
name which ones have a direct human-evidence review (source: Evipedia,
evidence_tier: human) and which are animal-model only, don't claim no
human evidence exists if any entry above has has_human_review: true.
"""
    logger.info(f"build_atlas_response | calling LLM for synthesis + human-evidence summaries concurrently")
    with ThreadPoolExecutor(max_workers=2) as executor:
        summary_future = executor.submit(call_llm, prompt, system_prompt=SUMMARY_SYSTEM_PROMPT)
        human_summary_future = executor.submit(
            call_llm, human_prompt, system_prompt=HUMAN_EVIDENCE_SYSTEM_PROMPT, use_secondary=True
        )
        summary = summary_future.result()
        human_evidence_summary = human_summary_future.result()
    logger.info(f"build_atlas_response | LLM summary received: {summary}")
    logger.info(f"build_atlas_response | human_evidence_summary received: {human_evidence_summary}")

    response = {
        "pc_group": pc_group,
        "pc_groups_explained": _explain_pc_groups(pc_group),
        "disease_name": disease_name,
        "mechanisms": mechanism_tags,
        "evidence": evidence,
        "summary": summary,
        "human_evidence_summary": human_evidence_summary,
    }
    logger.info(f"build_atlas_response | done | evidence={[e['name'] for e in evidence]}")
    return response
