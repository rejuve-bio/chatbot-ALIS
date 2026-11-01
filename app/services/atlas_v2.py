"""
Intervention Atlas: every value traceable to its source; one fixed response shape.

- lifestyle: the top interventions of the patient's PC groups (LinAge2 supplementary table).
- treatments: the top drug classes ChEMBL lists as approved for the disease, confirmed by the FDA
  label and grouped by NLM RxClass.
- investigational_evidence: Evipedia, NCBI (PubMed) and DrugAge are each searched directly with
  the risk's pathways (GO / Reactome, data/pathway_mapping.py); findings are merged per
  intervention, filtered by fixed rules, and ordered Evipedia / NCBI first, DrugAge-only last.
- systems: the avatar body systems, from the disease's place in the MeSH tree (mesh_service.py).

No LLM decides what is shown; the LLM only phrases the short summary (what drives the risk, the
best option, the main caution) from the final sections. Results are stored per disease + PC
groups (atlas_store.py), so any patient with the same risk and PCs is served from the store.
"""

import json
import logging
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from app.services import atlas_store
from app.services.biology_service import (
    DRUGAGE_COLLECTION, EVIPEDIA_COLLECTION,
    get_gene_grounding, get_human_evidence_for_compound,
)
from app.services.chembl_service import (
    ChemblUnavailable, base_drug_name, get_approved_drugs, get_target_genes, is_withdrawn,
)
from app.services.clinicaltrials_service import search_active_trials
from app.services.disease_terms import DISEASE_TERMS, get_disease_terms
from app.services.llm_service import call_llm, embed_text
from app.services.risk_systems import get_risk_systems, listed_systems
from app.services.ncbi_service import get_abstracts, get_citations, get_pathway_studies, get_recent_human_studies
from app.services.openfda_service import (
    LOOKUP_FAILED, boxed_warning_title, get_drug_label, is_indicated_for, is_marketed,
    rxnorm_ingredient_name,
)
from app.services.explain import label_use, paper_finding
from app.services.paper_check import PaperCheckUnavailable, evipedia_benefit, treats_disease
from app.services.pathway_service import get_risk_pathways, pathways_acted_on
from app.services.qdrant_service import search_chunks
from app.services.rxclass_service import get_contraindicated_drugs, get_drug_classes
from data.pc_chunks import PC_CHUNKS

logger = logging.getLogger(__name__)

_EXECUTOR_WORKERS = 16
_MAX_TREATMENT_CLASSES = 3
_MAX_LIFESTYLE = 3
_RANKING_POOL = 8
_WARMUP_PAUSE_SECONDS = 5
_PER_SOURCE = 5
_HITS_PER_PATHWAY = 3
_MAX_INVESTIGATIONAL = 10
_MAX_ANIMAL_ONLY = 3
_MIN_MEDIAN_GAIN = 5.0
_AGING_FIELDS = ("human_aging_link", "model_organism_link", "senescence_effect", "senescence_expression")

SUMMARY_SYSTEM_PROMPT = """
You write the summary line of one risk area in a doctor-facing longevity report.
The report's lists are already shown on screen; the summary must not repeat them.
You are given at most three facts:
- drivers: the biological processes driving this risk for this patient
- best_option: the single top-ranked option and its evidence level
- caution: a boxed warning carried by one of the listed drugs (may be absent)

CRITICAL RULES:
- Use only these facts. Never add a drug, gene, dose, number or claim.
- Say what drives the risk, then the best option with its evidence level, then
  the caution if one is given. Skip any fact that is absent.
- Evidence levels: approved = FDA-approved for this disease; human_and_animal =
  human studies plus animal lifespan data; human = human studies; animal =
  animal studies only (early research, not proven in humans).
- Be brief: at most 2 short sentences and 25 words in total. Shorten the caution
  to its key words (e.g. "boxed warning: lactic acidosis"). Plain text, no
  markdown, no source names.

- When the caution's name differs from the best option's, name it so the warning
  isn't read as belonging to the best option.

Examples:
"Driven by vascular aging and inflammation. Best option: Biguanide
(FDA-approved); boxed warning: lactic acidosis."
"Driven by neurodegeneration. Best option: Cholinesterase Inhibitor
(FDA-approved). Amyloid Beta-directed Antibody: boxed warning for brain swelling."
"""


def _variants(name: str) -> list[str]:
    return list(dict.fromkeys([name, name.lower(), name.capitalize(), name.title(), name.upper()]))


def _drug_genes(drug: str) -> set[str]:
    """Genes a drug acts on: ChEMBL's curated mechanism-of-action targets only. ClinPGx genes are
    left out on purpose — they change how a patient responds to a drug, not what the drug acts on."""
    return get_target_genes(drug)


def _disease_search_terms(terms: dict) -> list[str]:
    return terms.get("literature_headings") or terms.get("mesh_headings") or [terms.get("trial_condition")]


def _uses_mesh(terms: dict) -> bool:
    return bool(terms.get("literature_headings") or terms.get("mesh_headings"))


_PACKAGING = re.compile(r"pharmacy bulk|bulk package|not for direct infusion|for intravenous|dilut|package insert"
                        r"|^boxed warning|^warning section|^see full prescribing", re.I)


def _class_warning(drugs: list[str], labels: dict) -> str | None:
    titles: dict[str, list[str]] = {}
    for d in drugs:
        title = boxed_warning_title(labels.get(d))
        if title and not _PACKAGING.search(title):
            titles.setdefault(title, []).append(d)
    if not titles:
        return None
    if len(titles) == 1 and len(next(iter(titles.values()))) == len(drugs):
        return next(iter(titles))
    return "; ".join(f"{t} ({', '.join(ds)})" for t, ds in titles.items())


class TreatmentSourcesDown(Exception):
    """ChEMBL or openFDA couldn't be reached while building treatments."""


def _build_treatments(terms: dict, executor: ThreadPoolExecutor, sex: str | None = None) -> tuple[list[dict], set[str]]:
    candidates = sorted({d for heading in terms["mesh_headings"] for d in get_approved_drugs(heading)})
    labels = dict(zip(candidates, executor.map(get_drug_label, candidates)))
    if any(label is LOOKUP_FAILED for label in labels.values()):
        raise TreatmentSourcesDown("openFDA")
    approved = [d for d in candidates
                if is_indicated_for(labels[d], terms["label_terms"], terms.get("label_exclude_terms", []))]
    approved = [d for d in approved if not is_withdrawn(d) and is_marketed(d) is not False]
    uses = dict(zip(approved, executor.map(lambda d: label_use(d, labels[d], terms["trial_condition"]), approved)))
    approved = [d for d in approved if (uses[d] or {}).get("scope") == "general"]
    logger.info(f"atlas_v2 | ChEMBL candidates={len(candidates)} | confirmed by FDA label, marketed={approved}")
    if not approved:
        return [], set()

    drug_classes = dict(zip(approved, executor.map(get_drug_classes, approved)))
    class_sizes: dict[str, int] = {}
    for classes in drug_classes.values():
        for c in classes:
            class_sizes[c] = class_sizes.get(c, 0) + 1
    groups: dict[str, list[str]] = {}
    for d in approved:
        cls = max(drug_classes[d], key=lambda c: (class_sizes[c], c)) if drug_classes[d] else d.capitalize()
        if any(w in cls.lower() for w in ("diagnostic", "contrast")):
            continue
        groups.setdefault(cls, []).append(d)

    def _trial_count(item: tuple[str, list[str]]) -> tuple[list[dict], int]:
        return search_active_trials(item[1][:10], terms["trial_condition"], limit=5)

    def _label_count(drugs: list[str]) -> int:
        return sum((labels.get(d) or {}).get("label_count", 0) for d in drugs)

    class_items = sorted(groups.items(), key=lambda kv: -_label_count(kv[1]))[:_RANKING_POOL]
    with ThreadPoolExecutor(max_workers=max(len(class_items), 1)) as group_executor:
        trial_results = list(group_executor.map(_trial_count, class_items))
    label_counts = [_label_count(drugs) for _, drugs in class_items]
    trial_counts = [total for _, total in trial_results]

    def _ranks(values: list[int]) -> list[int]:
        order = sorted(range(len(values)), key=lambda i: -values[i])
        ranks = [0] * len(values)
        for rank, i in enumerate(order):
            ranks[i] = rank
        return ranks

    label_rank, trial_rank = _ranks(label_counts), _ranks(trial_counts)
    order = sorted(range(len(class_items)), key=lambda i: (trial_rank[i], label_rank[i]))
    top = order[:_MAX_TREATMENT_CLASSES]
    logger.info(f"atlas_v2 | class ranking: {[(class_items[i][0], label_counts[i], trial_counts[i]) for i in order]}")

    def _item(i: int) -> dict:
        cls, drugs = class_items[i]
        names = [cls] + drugs
        review = next((r for r in map(_evipedia_review, [cls] + drugs) if r), None)
        recent, _ = get_recent_human_studies(drugs, _disease_search_terms(terms), limit=5,
                                             sex=sex, mesh=_uses_mesh(terms))
        publications = _first_treating(recent, names, cls, terms["trial_condition"], sex)
        condition = terms["trial_condition"]
        listed = [f"{d} – {uses[d]['use']}" for d in drugs[:4] if uses.get(d)]
        more = f"; and {len(drugs) - 4} more" if len(drugs) > 4 else ""
        key_evidence = f"FDA-labelled uses for {condition.lower()}: " + "; ".join(listed) + more + "."
        if publications:
            finding = paper_finding(publications[0], cls, condition, sex)
            key_evidence += (f" Latest {_study_phrase(publications[0])} ({publications[0]['year']})"
                             + (f": {finding}." if finding else "."))
        key_evidence += " FDA labelling, not a treatment recommendation."
        return _flat_item(
            name=cls, drugs=drugs, evidence="approved",
            sources=["ChEMBL", "FDA drug label"] + (["Evipedia"] if review else []) + (["NCBI"] if recent else []),
            key_evidence=key_evidence, warning=_class_warning(drugs, labels), review=review, publications=publications,
            trials=_pick_trial(trial_results[i][0], names), genes=[],
        )

    with ThreadPoolExecutor(max_workers=max(len(top), 1)) as group_executor:
        treatments = list(group_executor.map(_item, top))
    return treatments, {base_drug_name(d) for t in treatments for d in t["drugs"]}


_SPECIES = {"Mus musculus": "Mouse", "Rattus norvegicus": "Rat", "Drosophila melanogaster": "Fly",
            "Caenorhabditis elegans": "Worm", "Saccharomyces cerevisiae": "Yeast",
            "Acheta domesticus": "Cricket", "Brachionus manjavacas": "Rotifer", "Brachionus plicatilis": "Rotifer",
            "Podospora anserina": "Fungus (Podospora)", "Nothobranchius furzeri": "Killifish", "Musca domestica": "Housefly",
            "Daphnia pulex": "Water flea", "Bombyx mori": "Silkworm", "Apis mellifera": "Honeybee"}
_SPECIES_TIER = {"Mus musculus": 0, "Rattus norvegicus": 0, "Drosophila melanogaster": 1, "Caenorhabditis elegans": 2}
_EVIDENCE_ORDER = {"human_and_animal": 0, "human": 1, "animal": 2}


def _evipedia_review(name: str) -> dict | None:
    for v in _variants(name):
        hit = get_human_evidence_for_compound(v)
        if hit:
            return hit
    return None


def _mentions(text: str, names: list[str]) -> bool:
    text = (text or "").lower()
    return any(n and len(n) > 2 and n.lower() in text for n in names)


_COMPARATOR = re.compile(r"\b(versus|vs\.?|compared (with|to)|in combination with|added to|add-on to|adjunct to)\b")


def _is_subject(title: str, names: list[str]) -> bool:
    """True when the title names the intervention and it isn't only the comparator or the
    background therapy ("Apixaban versus aspirin…", "X in combination with donepezil…")."""
    title = (title or "").lower()
    positions = [title.find(n.lower()) for n in names if n and len(n) > 2 and n.lower() in title]
    if not positions:
        return False
    first = min(positions)
    comparator = _COMPARATOR.search(title)
    return not (comparator and comparator.start() < first)


def _pick_publication(citations: list[dict], names: list[str]) -> list[dict]:
    """The single most recent paper whose title names the intervention (or [] if none does)."""
    relevant = {c["pmid"]: c for c in citations if c.get("pmid") and _mentions(c.get("title"), names)}
    return sorted(relevant.values(), key=lambda c: c.get("year") or "", reverse=True)[:1]


def _first_treating(papers: list[dict], names: list[str], item: str, condition: str,
                    sex: str | None = None) -> list[dict]:
    """The newest paper that names the item as its subject and tests it as a treatment for this
    disease, as a one-item list ([] if none). Checked newest first, stopping at the first that passes."""
    by_pmid = {p["pmid"]: p for p in papers if p.get("pmid") and _is_subject(p.get("title"), names)}
    ordered = sorted(by_pmid.values(), key=lambda p: p.get("year") or "", reverse=True)
    try:
        get_abstracts([p["pmid"] for p in ordered])
    except Exception:
        pass
    return next(([p] for p in ordered if treats_disease(p, item, condition, sex)), [])


def _pick_trial(trials: list[dict], names: list[str]) -> list[dict]:
    """The first active trial whose title names the intervention (or [] if none does)."""
    return [t for t in trials if _mentions(t.get("title"), names)][:1]


def _study_phrase(citation: dict) -> str:
    kind = (citation.get("type") or "").lower()
    if "meta-analysis" in kind:
        return "human meta-analysis"
    if "systematic review" in kind:
        return "human systematic review"
    if "randomized" in kind:
        return "human randomized trial"
    return "human study"


def _merge_key(name: str) -> str:
    """One key per real-world intervention, so "Rapamycin" (DrugAge) and "Sirolimus" (PubMed) merge."""
    return rxnorm_ingredient_name(name) or base_drug_name(name)


def _evipedia_by_pathway(terms_by_pathway: dict[str, str], disease_name: str) -> dict[str, dict]:
    """Evipedia reviews closest to each pathway in this disease; ranked by how many pathways they match."""
    found: dict[str, dict] = {}
    for pathway_name, term in terms_by_pathway.items():
        for hit in search_chunks(EVIPEDIA_COLLECTION, embed_text(f"{term} in {disease_name}"), limit=_HITS_PER_PATHWAY):
            entry = found.setdefault(hit["name"], {"hit": hit, "pathways": set()})
            entry["pathways"].add(pathway_name)
    return dict(sorted(found.items(), key=lambda kv: -len(kv[1]["pathways"]))[:_PER_SOURCE])


def _ncbi_by_pathway(terms_by_pathway: dict[str, str], disease_name: str, disease_terms: list[str],
                     executor: ThreadPoolExecutor, sex: str | None = None, mesh: bool = False) -> dict[str, dict]:
    """Interventions studied in recent human trials / meta-analyses on each pathway in this disease.
    A substance counts when the paper's title names it (the paper is about it), it is a specific
    compound (a PubMed supplementary concept, written lowercase, or an RxNorm ingredient), and it is
    not the pathway's own molecule ("glucose" for glucose regulation)."""
    found: dict[str, dict] = {}
    items = list(terms_by_pathway.items())
    for (pathway_name, term), studies in zip(items, executor.map(lambda kv: get_pathway_studies(kv[1], disease_terms, sex=sex, mesh=mesh), items)):
        context = f"{term} {pathway_name} {disease_name}".lower()
        for study in studies:
            title = study["title"].lower()
            for substance in study["substances"]:
                name = substance.lower()
                if name not in title or name in context:
                    continue
                if not (substance[:1].islower() or rxnorm_ingredient_name(substance)):
                    continue
                entry = found.setdefault(name, {"name": substance, "papers": {}})
                entry["papers"][study["pmid"]] = {k: study[k] for k in ("pmid", "title", "journal", "year", "type", "url")}
    return dict(sorted(found.items(), key=lambda kv: -len(kv[1]["papers"]))[:_PER_SOURCE])


def _lifespan_findings(chunk: dict) -> list[dict]:
    """DrugAge lifespan results per species. A species counts when its positive results outnumber
    its negative ones and their median gain is at least _MIN_MEDIAN_GAIN. Strongest first: species
    tier, then ITP-validated (mouse, replicated in three labs), more studies, both sexes, larger gain."""
    by_species: dict[str, list[tuple[float, dict]]] = {}
    for e in chunk.get("evidence") or []:
        try:
            pct = float(e.get("avg_lifespan_change_percent"))
        except (TypeError, ValueError):
            continue
        by_species.setdefault(e.get("species") or "Unknown organism", []).append((pct, e))
    itp = str(chunk.get("itp_validated")) == "True"
    findings = []
    for species, results in by_species.items():
        positive = [(pct, e) for pct, e in results if pct > 0]
        negative = [pct for pct, _ in results if pct < 0]
        if len(positive) <= len(negative):
            continue
        median = statistics.median(pct for pct, _ in positive)
        if median < _MIN_MEDIAN_GAIN:
            continue
        studies = sorted({e.get("pubmed_id") for _, e in positive if e.get("pubmed_id")})
        sexes = {e.get("gender") for _, e in positive} & {"Male", "Female"}
        findings.append({"species": species, "tier": _SPECIES_TIER.get(species, 3), "median": median,
                         "studies": len(studies), "pmids": studies, "both_sexes": len(sexes) == 2,
                         "itp": itp and species == "Mus musculus"})
    return sorted(findings, key=lambda f: (f["tier"], not f["itp"], -f["studies"], not f["both_sexes"], -f["median"]))


def _lifespan_phrase(findings: list[dict]) -> str:
    """"Animal model: Mouse lifespan +13% (ITP-validated, 7 studies); Worm lifespan +21%"."""
    parts = []
    for f in findings[:2]:
        notes = (["ITP-validated"] if f["itp"] else []) + ([f"{f['studies']} studies"] if f["studies"] > 1 else [])
        parts.append(f"{_SPECIES.get(f['species'], f['species'])} lifespan +{f['median']:.0f}%"
                     + (f" ({', '.join(notes)})" if notes else ""))
    return "Animal model: " + "; ".join(parts)


def _aging_targets(genes: set[str]) -> set[str]:
    """The genes with a GenAge (human or model organism) or CellAge (senescence) link."""
    return {g for g in genes if any(get_gene_grounding(g).get(f) for f in _AGING_FIELDS)}


def _grounding_genes(drug: str, aging_targets: set[str]) -> list[dict]:
    """Aging and senescence genes the drug acts on: its ChEMBL targets in this risk's pathways."""
    genes = []
    for g in sorted(aging_targets):
        grounding = get_gene_grounding(g)
        grounding["targeted_by_drug"] = {"source": "ChEMBL", "drug": drug, "pmids": []}
        genes.append(grounding)
    return genes


def _drugage_candidates(terms_by_pathway: dict[str, str], disease_name: str) -> dict[str, dict]:
    """DrugAge compounds with a consistent lifespan gain, closest to each of the risk's pathways
    or to the disease itself."""
    found: dict[str, dict] = {}
    def _add(hit: dict, searched: bool):
        findings = _lifespan_findings(hit)
        if not findings:
            return
        entry = found.setdefault(hit["name"], {"chunk": hit, "species": findings, "searched": False})
        entry["searched"] |= searched
    for query in list(terms_by_pathway.values()) + [disease_name]:
        for hit in search_chunks(DRUGAGE_COLLECTION, embed_text(query), limit=_HITS_PER_PATHWAY * 2):
            _add(hit, True)
    return found


def _build_investigational(
    disease_name: str, terms: dict, pathways: list[dict], executor: ThreadPoolExecutor, sex: str | None = None,
) -> list[dict]:
    """Evipedia, NCBI and DrugAge are each searched directly with the risk's pathways (plus DrugAge's
    disease match and ITP compounds); findings are merged per intervention, filtered by fixed rules
    and ordered: human + animal evidence, then human only, then animal only."""
    terms_by_pathway = {p["name"]: p["pathway_term"] for p in pathways} or {disease_name: disease_name}
    disease_terms = _disease_search_terms(terms)
    mesh = _uses_mesh(terms)
    evipedia = _evipedia_by_pathway(terms_by_pathway, disease_name)
    ncbi = _ncbi_by_pathway(terms_by_pathway, disease_name, disease_terms, executor, sex, mesh)
    drugage = _drugage_candidates(terms_by_pathway, disease_name)
    logger.info(f"atlas_v2 | candidates | evipedia={list(evipedia)} | ncbi={[v['name'] for v in ncbi.values()]} | "
                f"drugage={list(drugage)}")

    candidates: dict[str, dict] = {}
    def _candidate(name: str) -> dict:
        key = _merge_key(name)
        return candidates.setdefault(key, {"key": key, "names": [], "evipedia": None, "papers": {}, "drugage": None})
    for name, e in evipedia.items():
        c = _candidate(name); c["names"].append(name); c["evipedia"] = e["hit"]
    for e in ncbi.values():
        c = _candidate(e["name"]); c["names"].append(e["name"]); c["papers"].update(e["papers"])
    for name, e in drugage.items():
        c = _candidate(name); c["names"].append(name); c["drugage"] = e

    contraindicated = {d for mesh in terms.get("mesh_ids", []) for d in get_contraindicated_drugs(mesh)}
    condition = terms.get("trial_condition", disease_name)

    def _item(c: dict) -> dict | None:
        key = c["key"]
        if key in contraindicated or is_withdrawn(key):
            return None
        label = get_drug_label(key)
        if label is LOOKUP_FAILED:
            return None
        if label and is_marketed(key) is False:
            return None
        names = list(dict.fromkeys(c["names"] + [key]))
        approved = bool(label) and is_indicated_for(label, terms.get("label_terms", []), terms.get("label_exclude_terms", []))
        use = label_use(key, label, condition) if approved else None
        approved = approved and (use or {}).get("scope") == "general"
        recent, _ = get_recent_human_studies(names, disease_terms, limit=5, sex=sex, mesh=mesh)
        human_papers = _first_treating(list(c["papers"].values()) + recent, names, c["names"][0], condition, sex)
        pathway_hits = pathways_acted_on(_drug_genes(key), pathways)
        aging_risk = condition == "Aging"
        if not (approved or human_papers or pathway_hits or (aging_risk and (c["drugage"] or c["evipedia"]))):
            return None
        aging_targets = _aging_targets({g for p in pathway_hits for g in p["genes"]})
        review = c["evipedia"] or _evipedia_review(c["names"][0]) or _evipedia_review(key)
        benefit = evipedia_benefit(c["names"][0], review, condition) if review else None
        drugage = c["drugage"]
        if c["evipedia"] and not (drugage or c["papers"] or approved or human_papers or benefit):
            return None
        review_human = bool(benefit and benefit["level"] in ("High", "Medium"))

        if not (approved or human_papers or drugage or benefit):
            return None
        human = approved or bool(human_papers) or review_human
        evidence = "human_and_animal" if human and drugage else "human" if human else "animal"
        key_evidence = []
        if approved:
            key_evidence.append(f"FDA-labelled for {condition.lower()}: {use['use']}.")
        if human_papers:
            finding = paper_finding(human_papers[0], c["names"][0], condition, sex)
            phrase = _study_phrase(human_papers[0])
            key_evidence.append(f"{phrase[:1].upper() + phrase[1:]} ({human_papers[0]['year']})"
                                + (f": {finding}." if finding else "."))
        if benefit:
            key_evidence.append(f"Evipedia ({benefit['level'].lower()} evidence): {benefit['title']}.")
        if drugage:
            only = "" if (approved or human_papers or review_human) else f" Animal studies only; not tested for {condition.lower()} in people."
            key_evidence.append(f"{_lifespan_phrase(drugage['species'])}.{only}")

        cited = get_citations((review or {}).get("pmids", []), limit=30) + get_citations(
            [pmid for f in (drugage or {}).get("species", []) for pmid in f["pmids"]], limit=10)
        trials, _ = search_active_trials([key], condition, limit=5)
        trial = [t for t in trials if _is_subject(t.get("title"), names)][:1]
        sources = ((["FDA drug label"] if approved else []) + (["DrugAge"] if drugage else [])
                   + (["Evipedia"] if review else []) + (["NCBI"] if human_papers else []))
        item = _flat_item(
            name=c["names"][0], drugs=[key] if rxnorm_ingredient_name(key) or label else [],
            evidence=evidence, sources=sources, key_evidence=" ".join(key_evidence) or None,
            warning=boxed_warning_title(label), review=review,
            publications=human_papers or _pick_publication(cited, names),
            trials=trial,
            genes=_grounding_genes(key, aging_targets) if drugage else [],
        )
        kinds = sum([approved, bool(human_papers), bool(drugage), bool(trial)])
        best = drugage["species"][0] if drugage else None
        item["_rank"] = (-kinds, _EVIDENCE_ORDER[evidence], -len(aging_targets), -len(pathway_hits),
                         best["tier"] if best else 9, not (best and best["itp"]), -(best["studies"] if best else 0),
                         -int((human_papers[0]["year"] if human_papers else "0") or 0),
                         -(best["median"] if best else 0))
        item["_animal_only"] = evidence == "animal"
        return item

    items = sorted((i for i in executor.map(_item, candidates.values()) if i), key=lambda i: i.pop("_rank"))
    kept, animal_only = [], 0
    for item in items:
        if item.pop("_animal_only"):
            animal_only += 1
            if animal_only > _MAX_ANIMAL_ONLY:
                continue
        kept.append(item)
    return kept[:_MAX_INVESTIGATIONAL]


def _lifestyle(pc_groups: list[str]) -> list[str]:
    """The first _MAX_LIFESTYLE interventions, taking PC groups in the order the backend lists them
    (strongest contributor first), so the top items come from the most relevant PC."""
    out: list[str] = []
    for pc_group in pc_groups:
        for chunk in PC_CHUNKS:
            if chunk.get("pc_group") == pc_group:
                for item in chunk.get("interventions", []):
                    if item not in out:
                        out.append(item)
    return out[:_MAX_LIFESTYLE]


_MAX_DRIVERS = 3


def _summary(disease_name: str, mechanisms: list[str], treatments: list[dict], investigational: list[dict]) -> str:
    """What the lists don't say: what drives the risk, the one best option, and the main caution
    (the boxed warning of the highest-ranked listed item that has one, treatments first)."""
    items = treatments + investigational
    best = items[0] if items else None
    cautioned = next((i for i in items if i["warning"]), None)
    facts = {
        "drivers": mechanisms[:_MAX_DRIVERS],
        "best_option": {"name": best["name"], "drugs": best["drugs"][:3], "evidence": best["evidence"],
                        "section": "treatments" if treatments else "investigational"} if best else None,
        "caution": {"name": cautioned["name"], "boxed_warning": cautioned["warning"]} if cautioned else None,
    }
    try:
        return call_llm(f"Risk area: {disease_name}\n\n{json.dumps(facts, indent=2)}",
                        system_prompt=SUMMARY_SYSTEM_PROMPT).strip()
    except Exception as e:
        logger.error(f"atlas_v2 | summary failed, using fallback text: {e}")
        parts = [f"Driven mainly by {', '.join(m.lower() for m in facts['drivers'])}." if facts["drivers"] else ""]
        if best:
            parts.append(f"Best-supported option: {best['name']} ({best['evidence'].replace('_', ' ')}).")
        if cautioned:
            parts.append(f"{cautioned['name']} carries a boxed warning: {cautioned['warning'].lower()}.")
        return " ".join(p for p in parts if p)


_TREATMENT_CACHE: dict[str, tuple[float, tuple[list[dict], set[str]]]] = {}
_CACHE_TTL_SECONDS = 86400


def get_treatments(disease_name: str, sex: str | None = None) -> tuple[list[dict], set[str]]:
    """Approved treatment classes for a disease; cached per disease and sex (the cited paper's
    population has to fit the patient)."""
    cached = _TREATMENT_CACHE.get((disease_name, sex))
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]
    terms = get_disease_terms(disease_name)
    with ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS) as executor:
        result = _build_treatments(terms, executor, sex)
    _TREATMENT_CACHE[(disease_name, sex)] = (time.time(), result)
    return result


def warm_treatment_cache() -> None:
    """Builds every known disease's treatments one at a time (gentle on the public APIs) so that
    first requests don't wait on them."""
    if os.getenv("ATLAS_WARMUP", "1") == "0":
        logger.info("atlas_v2 | warm-up disabled (ATLAS_WARMUP=0)")
        return
    for disease_name, sex in ((d, s) for d in DISEASE_TERMS for s in ("M", "F")):
        time.sleep(_WARMUP_PAUSE_SECONDS)
        try:
            started = time.time()
            treatments, _ = get_treatments(disease_name, sex)
            logger.info(f"atlas_v2 | warmed {disease_name}: {len(treatments)} class(es) in {time.time() - started:.0f}s")
        except Exception as e:
            logger.warning(f"atlas_v2 | warm-up failed for {disease_name}: {e}")



def _trial(t: dict) -> dict:
    return {"nct_id": t.get("nct_id"), "title": t.get("title"), "status": t.get("status"),
            "phase": t.get("phase") or [], "url": t.get("url")}


def _citation(c: dict) -> dict:
    return {k: c.get(k) for k in ("pmid", "title", "journal", "year", "type", "url")}


def _gene(g: dict) -> dict:
    model = g.get("model_organism_link")
    target = g.get("targeted_by_drug") or {}
    return {
        "gene": g.get("gene"),
        "human_aging_link": g.get("human_aging_link"),
        "model_organism_link": (
            {k: model.get(k) for k in ("source", "animal_gene", "organism", "ortholog_confidence")}
            if model else None
        ),
        "senescence_effect": g.get("senescence_effect"),
        "senescence_expression": g.get("senescence_expression"),
        "targeted_by_drug": {"source": target.get("source"), "drug": target.get("drug"),
                             "pmids": target.get("pmids") or []},
    }


def _flat_item(*, name, drugs, evidence, sources, key_evidence, warning, review, publications, trials, genes) -> dict:
    return {
        "name": name,
        "drugs": drugs or [],
        "evidence": evidence,
        "sources": sources,
        "key_evidence": key_evidence,
        "warning": warning,
        "evipedia_url": (review or {}).get("url"),
        "interactions": (review or {}).get("interactions_and_contraindications"),
        "publications": [_citation(c) for c in publications],
        "active_trials": [_trial(t) for t in trials],
        "gene_level_grounding": [_gene(g) for g in genes],
    }


_LINAGE2_NAMES = {"Insulin use": "Diabetes mellitus and insulin use"}


def _disease_mechanisms(disease_name: str) -> list[str]:
    """Mechanisms of every LinAge2 PC group that lists this disease."""
    norm = lambda s: " ".join("".join(ch if ch.isalnum() else " " for ch in s.lower()).split())
    target, mechanisms = norm(_LINAGE2_NAMES.get(disease_name, disease_name)), []
    for chunk in PC_CHUNKS:
        if any(norm(d) == target for d in chunk.get("diseases", [])):
            mechanisms += [m for m in chunk.get("mechanisms", []) if m != "Indeterminate" and m not in mechanisms]
    return mechanisms


def build_risk_area(disease_name: str, pc_groups: list[str], mechanisms: list[str] | None = None) -> dict:
    stored = atlas_store.get(disease_name, pc_groups)
    if stored:
        stored["systems"] = listed_systems(disease_name) or stored.get("systems")
        return stored
    if mechanisms is None:
        from app.services.atlas_service import combined_mechanisms
        mechanisms, _ = combined_mechanisms(pc_groups)

    started = time.time()
    terms = get_disease_terms(disease_name)
    pathways = get_risk_pathways(mechanisms) or get_risk_pathways(_disease_mechanisms(disease_name))
    sex = pc_groups[0][-1] if pc_groups and pc_groups[0][-1] in ("M", "F") else None
    try:
        treatments, _ = get_treatments(disease_name, sex)
        sources_down = False
    except (ChemblUnavailable, TreatmentSourcesDown, PaperCheckUnavailable) as e:
        logger.warning(f"atlas_v2 | {disease_name} | treatment source unreachable ({e!r}); returning no treatments")
        treatments, sources_down = [], True
    try:
        with ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS) as executor:
            investigational = _build_investigational(disease_name, terms, pathways, executor, sex)
    except (PaperCheckUnavailable, ChemblUnavailable) as e:
        logger.warning(f"atlas_v2 | {disease_name} | a source couldn't be checked ({e!r}); not caching this result")
        investigational, sources_down = [], True
    lifestyle = _lifestyle(pc_groups)

    result = {
        "disease_name": disease_name,
        "systems": get_risk_systems(disease_name, terms.get("mesh_headings")),
        "summary": _summary(disease_name, mechanisms, treatments, investigational),
        "lifestyle": lifestyle,
        "treatments": treatments,
        "investigational_evidence": investigational,
    }
    if sources_down:
        logger.warning(f"atlas_v2 | {disease_name} | not caching: treatments are missing because a source was down")
        return dict(result)
    logger.info(f"atlas_v2 | {disease_name} | treatments={[t['name'] for t in treatments]} | "
                f"investigational={[i['name'] for i in investigational]} | {time.time() - started:.1f}s")
    atlas_store.put(disease_name, pc_groups, result)
    return dict(result)
