"""
One Qdrant collection per source, not per tier.
"""

import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from app.services.qdrant_service import (
    create_collection, upsert_chunks, search_chunks, filter_chunks, get_collection_count,
)
from app.services.llm_service import embed_batch, embed_text
from app.services.pathway_lookup import expand_mechanisms


logger = logging.getLogger(__name__)

DRUGAGE_COLLECTION = os.getenv("DRUGAGE_COLLECTION", "drugage_evidence")
GENAGE_HUMAN_COLLECTION = os.getenv("GENAGE_HUMAN_COLLECTION", "genage_human_evidence")
GENAGE_MODELS_COLLECTION = os.getenv("GENAGE_MODELS_COLLECTION", "genage_models_evidence")
CELLAGE_COLLECTION = os.getenv("CELLAGE_COLLECTION", "cellage_evidence")
CELLAGE_SIGNATURES_COLLECTION = os.getenv("CELLAGE_SIGNATURES_COLLECTION", "cellage_signatures_evidence")
CLINPGX_COLLECTION = os.getenv("CLINPGX_COLLECTION", "clinpgx_evidence")
EVIPEDIA_COLLECTION = os.getenv("EVIPEDIA_COLLECTION", "evipedia_evidence")

ALL_BIOLOGY_COLLECTIONS = [
    DRUGAGE_COLLECTION,
    GENAGE_HUMAN_COLLECTION,
    GENAGE_MODELS_COLLECTION,
    CELLAGE_COLLECTION,
    CELLAGE_SIGNATURES_COLLECTION,
    CLINPGX_COLLECTION,
    EVIPEDIA_COLLECTION,
]

ATLAS_RESOURCES_FOR_CHAT = [
    DRUGAGE_COLLECTION,
    GENAGE_HUMAN_COLLECTION,
    GENAGE_MODELS_COLLECTION,
    CELLAGE_COLLECTION,
    CELLAGE_SIGNATURES_COLLECTION,
    CLINPGX_COLLECTION,
    EVIPEDIA_COLLECTION,
]


def init_biology_collections():
    """
    Called once at app startup (see main.py's lifespan). Creates any
    missing collection, then populates only the ones that are actually
    empty — safe to run on every `docker compose up`, including on a
    server that already has data, without re-scraping/re-embedding
    everything or creating duplicate points (upsert_chunks uses random
    UUIDs per point, not content-keyed ids, so a blind re-populate would
    double up an already-populated collection rather than just refresh it).

    First-ever boot on a fresh server will take a while (Evipedia alone is
    a ~30 minute live scrape of ~600 reviews) — that's expected and only
    happens once. Every restart after that is fast, since every collection
    already has points and gets skipped.
    """
   
    populators = {
        DRUGAGE_COLLECTION: populate_drugage,
        GENAGE_HUMAN_COLLECTION: populate_genage_human,
        GENAGE_MODELS_COLLECTION: populate_genage_models,
        CELLAGE_COLLECTION: populate_cellage,
        CELLAGE_SIGNATURES_COLLECTION: populate_cellage_signatures,
        CLINPGX_COLLECTION: populate_clinpgx,
        EVIPEDIA_COLLECTION: populate_evipedia,
    }

    for name in ALL_BIOLOGY_COLLECTIONS:
        create_collection(name)

    for name in ALL_BIOLOGY_COLLECTIONS:
        count = get_collection_count(name)
        if count == 0:
            logger.info(f"{name} is empty — auto-populating")
            try:
                populators[name]()
            except Exception as e:
                logger.error(f"Failed to auto-populate {name}: {e}")
        else:
            logger.info(f"{name} already has {count} points — skipping population")


def _populate(collection_name: str, loader_fn, label: str, batch_size: int = 200) -> int:
    """
    Embeds and upserts in batches rather than all at once at the end — a
    long-running population (thousands of chunks) is one interruption away
    from losing everything if it only commits in a single final upsert.
    Batching means an interruption only costs the current batch, not the
    whole run.
    """
    chunks = loader_fn()
    if not chunks:
        logger.warning(f"No chunks produced by {label} — nothing to populate")
        return 0
    total = len(chunks)
    populated = 0
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]

        good_chunks, good_vectors = [], []
        for chunk in batch:
            try:
                good_vectors.append(embed_text(chunk["raw_text"]))
                good_chunks.append(chunk)
            except Exception as e:
                logger.error(f"{collection_name} | skipping chunk {chunk.get('name')!r} — "
                             f"embedding failed: {e}")
        if good_chunks:
            upsert_chunks(collection_name, good_chunks, good_vectors)
            populated += len(good_chunks)
        logger.info(f"{collection_name} | batch committed: {populated}/{total} "
                    f"chunks from {label} so far ({min(i + batch_size, total)}/{total} attempted)")
    logger.info(f"{collection_name} populated with {populated}/{total} chunks from {label} "
                f"(evidence_tier={chunks[0].get('evidence_tier')})")
    return populated


def populate_drugage() -> int:
    from app.services.ingestion.csv_cleaners.drugage_cleaner import load_drugage_chunks
    return _populate(DRUGAGE_COLLECTION, load_drugage_chunks, "DrugAge")


def populate_genage_human() -> int:
    from app.services.ingestion.csv_cleaners.genage_human_cleaner import load_genage_human_chunks
    return _populate(GENAGE_HUMAN_COLLECTION, load_genage_human_chunks, "GenAge (human)")


def populate_genage_models() -> int:
    from app.services.ingestion.csv_cleaners.genage_models_cleaner import load_genage_models_chunks
    return _populate(GENAGE_MODELS_COLLECTION, load_genage_models_chunks, "GenAge (models, ortholog-bridged)")


def populate_cellage() -> int:
    from app.services.ingestion.csv_cleaners.cellage_cleaner import load_cellage_chunks
    return _populate(CELLAGE_COLLECTION, load_cellage_chunks, "CellAge")


def populate_cellage_signatures() -> int:
    from app.services.ingestion.csv_cleaners.cellage_signatures_cleaner import load_cellage_signature_chunks
    return _populate(CELLAGE_SIGNATURES_COLLECTION, load_cellage_signature_chunks, "CellAge signatures")


def populate_clinpgx() -> int:
    from app.services.ingestion.csv_cleaners.clinpgx_cleaner import load_clinpgx_chunks
    return _populate(CLINPGX_COLLECTION, load_clinpgx_chunks, "ClinPGx")


def populate_evipedia(limit: int | None = None) -> int:
    """
    Live scrape, not a local file — safe and expected to be re-run
    periodically to pick up new/updated reviews (pass limit for a quick
    partial refresh instead of the full ~634-page scrape).
    """
    from app.services.ingestion.web_cleaners.evipedia_cleaner import load_evipedia_chunks
    return _populate(EVIPEDIA_COLLECTION, lambda: load_evipedia_chunks(limit=limit), "Evipedia")


def populate_all_biology_collections() -> dict[str, int]:
    """Populates all source collections. Safe to re-run to refresh any/all."""
    counts = {
        "drugage": populate_drugage(),
        "genage_human": populate_genage_human(),
        "genage_models": populate_genage_models(),
        "cellage": populate_cellage(),
        "cellage_signatures": populate_cellage_signatures(),
        "clinpgx": populate_clinpgx(),
        "evipedia": populate_evipedia(),
    }
    logger.info(f"All biology collections populated: {counts} (total {sum(counts.values())})")
    return counts


def search_biology_collection(
    collection_name: str,
    query_vector: list[float],
    name: str | None = None,
    evidence_tier: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    Search any single biology collection by name, optionally filtered by
    the chunk's own name field and/or evidence_tier (animal/human) — useful
    if a collection is ever queried alongside others despite living separately.
    """
    filters = {"name": name, "evidence_tier": evidence_tier}
    results = search_chunks(collection_name, query_vector, filters=filters, limit=limit)
    result_names = [r.get("name") for r in results]
    logger.info(f"search_biology_collection | collection={collection_name} | "
                f"filters={filters} | results={result_names}")
    return results


_drug_synonym_llm_lock = threading.Lock()


def _llm_disambiguate_drug_synonym(compound_name: str, candidates: list[str]) -> str | None:
    """
    Last-resort synonym check: is any candidate the EXACT SAME real-world
    drug as compound_name (e.g. Rapamycin/Sirolimus), not just a related
    one (e.g. Rapamycin vs Everolimus — both real rapalogs, different
    drugs)? Embedding similarity alone can't reliably tell these apart —
    tested directly: Sirolimus (true synonym) and Everolimus (different
    drug) scored 0.8365 vs 0.8133, too close to threshold safely. The LLM
    call is the actual identity check; the embedding search above it is
    only there to produce a short, cheap candidate list.
    """
    from app.services.llm_service import call_llm
    prompt = f"""
Query drug: {compound_name}

Candidate drug names found in a pharmacogenomics database (by embedding similarity, not verified):
{chr(10).join(f"{i + 1}. {c}" for i, c in enumerate(candidates))}

Which of these candidates, if any, refers to the EXACT SAME real-world drug as the query drug
(i.e. a synonym/alternate name for the identical compound, not just a related or similar drug)?
Answer with just the candidate name if there is an exact match, or "none" if none of them are
the same drug. One word or phrase only, no explanation.
"""
    answer = call_llm(prompt).strip()
    if answer.lower() == "none" or answer not in candidates:
        logger.info(f"_llm_disambiguate_drug_synonym | compound={compound_name} | "
                    f"candidates={candidates} | no confirmed synonym")
        return None
    logger.info(f"_llm_disambiguate_drug_synonym | compound={compound_name} | "
                f"candidates={candidates} | LLM confirmed synonym: {answer}")
    return answer


def get_target_genes_for_drug(compound_name: str) -> list[dict]:
    logger.info(f"get_target_genes_for_drug | compound={compound_name}")
    hits = filter_chunks(CLINPGX_COLLECTION, {"drug_name": compound_name}, limit=20)

    if not hits:
        for variant in (compound_name.lower(), compound_name.upper(), compound_name.capitalize()):
            if variant == compound_name:
                continue
            hits = filter_chunks(CLINPGX_COLLECTION, {"drug_name": variant}, limit=20)
            if hits:
                logger.info(f"get_target_genes_for_drug | compound={compound_name} | "
                            f"matched via case variant '{variant}'")
                break

    if not hits:
        
        query_vector = embed_text(compound_name)
        shortlist_hits = search_chunks(CLINPGX_COLLECTION, query_vector, limit=10, min_score=0.65)
        candidates = []
        for h in shortlist_hits:
            d = h.get("drug_name")
            if d and d not in candidates and d.lower() != compound_name.lower():
                candidates.append(d)
        if candidates:
            with _drug_synonym_llm_lock:
                confirmed = _llm_disambiguate_drug_synonym(compound_name, candidates)
            if confirmed:
                hits = filter_chunks(CLINPGX_COLLECTION, {"drug_name": confirmed}, limit=20)
                logger.info(f"get_target_genes_for_drug | compound={compound_name} | "
                            f"matched via LLM-confirmed synonym '{confirmed}'")

    genes = [
        {
            "gene": h.get("name"),
            "pk_relevant": h.get("pk_relevant"),
            "pd_relevant": h.get("pd_relevant"),
            "pmids": h.get("pmids"),
        }
        for h in hits
        if h.get("name")
    ]
    logger.info(f"get_target_genes_for_drug | compound={compound_name} | "
                f"genes_found={[g['gene'] for g in genes]}")
    return genes


def get_human_evidence_for_compound(compound_name: str) -> dict | None:
    """Exact-match lookup — does Evipedia have a human-evidence review for this compound?"""
    logger.info(f"get_human_evidence_for_compound | compound={compound_name}")
    hits = filter_chunks(EVIPEDIA_COLLECTION, {"name": compound_name}, limit=1)
    if hits:
        logger.info(f"get_human_evidence_for_compound | compound={compound_name} | "
                    f"found | evidence_tier={hits[0].get('evidence_tier')}")
    else:
        logger.info(f"get_human_evidence_for_compound | compound={compound_name} | none found")
    return hits[0] if hits else None


def get_gene_grounding(gene_symbol: str) -> dict:
    logger.info(f"get_gene_grounding | gene={gene_symbol}")
    grounding = {
        "gene": gene_symbol,
        "human_aging_link": None,
        "model_organism_link": None,
        "senescence_effect": None,
        "senescence_expression": None,
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        human_future = executor.submit(filter_chunks, GENAGE_HUMAN_COLLECTION, {"name": gene_symbol}, limit=1)
        model_future = executor.submit(filter_chunks, GENAGE_MODELS_COLLECTION, {"name": gene_symbol}, limit=1)
        cellage_future = executor.submit(filter_chunks, CELLAGE_COLLECTION, {"name": gene_symbol}, limit=1)
        sig_future = executor.submit(filter_chunks, CELLAGE_SIGNATURES_COLLECTION, {"name": gene_symbol}, limit=1)

        human_hits = human_future.result()
        model_hits = model_future.result()
        cellage_hits = cellage_future.result()
        sig_hits = sig_future.result()

    if human_hits:
        grounding["human_aging_link"] = {
            "source": "GenAge (human)",
            "evidence_type": human_hits[0].get("evidence_type"),
        }

    if model_hits:
        m = model_hits[0]
        grounding["model_organism_link"] = {
            "source": "GenAge (models)",
            "animal_gene": m.get("animal_gene_symbol"),
            "organism": m.get("organism"),
            "ortholog_confidence": m.get("ortholog_confidence"),
            "omim_diseases": m.get("omim_diseases"),
        }

    if cellage_hits:
        grounding["senescence_effect"] = {
            "source": "CellAge",
            "effect": cellage_hits[0].get("senescence_effect"),
        }

    if sig_hits:
        s = sig_hits[0]
        grounding["senescence_expression"] = {
            "source": "CellAge signatures",
            "direction": s.get("expression_direction"),
            "p_value": s.get("p_value"),
        }

    found_in = [k for k, v in grounding.items() if k != "gene" and v is not None]
    logger.info(f"get_gene_grounding | gene={gene_symbol} | found_in={found_in or 'none'}")
    return grounding


_compound_search_cache: dict[tuple, list[dict]] = {}


def _build_combined_query_terms(mechanism_tags: list[str], disease_name: str | None) -> list[str]:

    pathway_terms = expand_mechanisms(mechanism_tags)
    combined_terms = list(mechanism_tags)
    for term in pathway_terms:
        if term not in combined_terms:
            combined_terms.append(term)
    if disease_name and disease_name not in combined_terms:
        combined_terms.append(disease_name)
    return combined_terms


def _search_disease_specific(collection: str, disease_name: str | None, limit: int = 5) -> list[dict]:
    if not disease_name:
        return []
    query_vector = embed_text(disease_name)
    return search_chunks(collection, query_vector, limit=limit)


def search_investigational_compounds(
    mechanism_tags: list[str], disease_name: str | None = None, limit: int = 5
) -> list[dict]:

    cache_key = (tuple(sorted(mechanism_tags)), disease_name)
    if cache_key in _compound_search_cache:
        logger.info(f"search_investigational_compounds | cache hit for {cache_key}")
        return list(_compound_search_cache[cache_key])

    combined_terms = _build_combined_query_terms(mechanism_tags, disease_name)
    logger.info(f"search_investigational_compounds | mechanism_tags={mechanism_tags} | "
                f"disease_name={disease_name} | combined_terms={combined_terms}")
    query_text = ", ".join(combined_terms)
    query_vector = embed_text(query_text)

    with ThreadPoolExecutor(max_workers=2) as executor:
        mechanism_future = executor.submit(search_chunks, DRUGAGE_COLLECTION, query_vector, limit=limit)
        disease_future = executor.submit(_search_disease_specific, DRUGAGE_COLLECTION, disease_name, limit)
        mechanism_hits = mechanism_future.result()
        disease_hits = disease_future.result()
    logger.info(f"search_investigational_compounds | mechanism_hits={[h.get('name') for h in mechanism_hits]} | "
                f"disease_hits={[h.get('name') for h in disease_hits]}")

    
    seen_compounds = set()
    selected_hits = []
    for hit in disease_hits + mechanism_hits:
        if len(selected_hits) >= limit:
            break
        compound = hit.get("name")
        if not compound or compound in seen_compounds:
            continue
        seen_compounds.add(compound)
        selected_hits.append(hit)


    def _grounding_for_hit(hit: dict) -> list[dict]:
        target_genes = get_target_genes_for_drug(hit.get("name"))
        grounding = []
        for g in target_genes:
            g_data = get_gene_grounding(g["gene"])
            g_data["targeted_by_drug"] = {"source": "ClinPGx", "pmids": g.get("pmids")}
            grounding.append(g_data)
        return grounding

    with ThreadPoolExecutor(max_workers=max(len(selected_hits), 1)) as executor:
        grounding_futures = [executor.submit(_grounding_for_hit, hit) for hit in selected_hits]
        gene_level_groundings = [f.result() for f in grounding_futures]

    results = [
        {
            "compound_name": hit.get("name"),
            "source": hit.get("source", "DrugAge"),
            "evidence_tier": hit.get("evidence_tier", "animal_model"),
            "species": hit.get("organism"),

            "note": None,
            "gene_level_grounding": gene_level_grounding,
        }
        for hit, gene_level_grounding in zip(selected_hits, gene_level_groundings)
    ]

    logger.info(f"search_investigational_compounds | mechanism_tags={mechanism_tags} | "
                f"compounds_found={[r['compound_name'] for r in results]}")
    _compound_search_cache[cache_key] = results
    return list(results)


_human_evidence_search_cache: dict[tuple, list[dict]] = {}


def search_human_evidence(
    mechanism_tags: list[str], disease_name: str | None = None, limit: int = 5
) -> list[dict]:
    cache_key = (tuple(sorted(mechanism_tags)), disease_name)
    if cache_key in _human_evidence_search_cache:
        logger.info(f"search_human_evidence | cache hit for {cache_key}")
        return list(_human_evidence_search_cache[cache_key])

    combined_terms = _build_combined_query_terms(mechanism_tags, disease_name)
    logger.info(f"search_human_evidence | mechanism_tags={mechanism_tags} | "
                f"disease_name={disease_name} | combined_terms={combined_terms}")
    query_text = ", ".join(combined_terms)
    query_vector = embed_text(query_text)

    with ThreadPoolExecutor(max_workers=2) as executor:
        mechanism_future = executor.submit(search_chunks, EVIPEDIA_COLLECTION, query_vector, limit=limit)
        disease_future = executor.submit(_search_disease_specific, EVIPEDIA_COLLECTION, disease_name, limit)
        mechanism_hits = mechanism_future.result()
        disease_hits = disease_future.result()
    logger.info(f"search_human_evidence | mechanism_hits={[h.get('name') for h in mechanism_hits]} | "
                f"disease_hits={[h.get('name') for h in disease_hits]}")

    seen_names = set()
    results = []
    for hit in disease_hits + mechanism_hits:
        if len(results) >= limit:
            break
        name = hit.get("name")
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        results.append({
            "intervention_name": name,
            "source": "Evipedia",
            "evidence_tier": hit.get("evidence_tier"),
            "category": hit.get("category"),
            "url": hit.get("url"),
            "interactions_and_contraindications": hit.get("interactions_and_contraindications"),
            "pmids": hit.get("pmids", []),
            "nct_ids": hit.get("nct_ids", []),
        })
    _human_evidence_search_cache[cache_key] = results
    return list(results)
