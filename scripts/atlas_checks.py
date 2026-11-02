"""
Checks every stored Intervention Atlas result against the rules the pipeline is meant to enforce,
plus known traps found during review. Run after each rebuild:

    docker compose run --rm --no-deps -T -e PYTHONPATH=/app api python scripts/atlas_checks.py

Exits 1 when any check fails, printing each failing item.
"""

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict

from app.services.atlas_store import FORMAT_VERSION, STORE_PATH
from app.services.chembl_service import ChemblUnavailable, get_target_genes, is_withdrawn
from app.services.openfda_service import is_marketed
from app.services.paper_check import _DEVICE, _NOT_EFFICACY

AREA_KEYS = ["disease_name", "systems", "summary", "lifestyle", "treatments", "investigational_evidence"]
ITEM_KEYS = ["name", "drugs", "evidence", "sources", "key_evidence", "warning", "evipedia_url", "interactions",
             "publications", "active_trials", "gene_level_grounding"]
SYSTEMS = {"cardiovascular", "respiratory", "neurological", "metabolic", "hepatic", "digestive", "renal",
           "musculoskeletal", "reproductive", "immune", "vision", "hearing", "skin", "unlocalized"}
WRONG_MEDICINES = {"nicotine", "bisphenol a", "cholesterol", "carboplatin", "paclitaxel", "factor viii", "levodopa"}
ANY_POPULATION = re.compile(r"pregnan|gestation|child|pediatric|paediatric|infant|maternal|healthy (subjects|volunteers)", re.I)
OTHER_SEX = {"M": re.compile(r"\bwomen\b|postmenopausal|ovarian|cervical|breast cancer", re.I),
             "F": re.compile(r"\bmen\b|prostat|testosterone deficiency", re.I)}
INHERITED = re.compile(r"lynch syndrome|brca|familial adenomatous", re.I)
HUMAN_BASIS = re.compile(r"FDA-labelled|Human (randomized trial|meta-analysis|systematic review|study)|"
                         r"Evipedia \((high|medium) evidence\)")
PACKAGING = re.compile(r"pharmacy bulk|bulk package|direct infusion|^boxed warning|warning section", re.I)
CAPTOPRIL_HUMAN_OK = re.compile(r"hypertension|cardiovascular|cardiac failure|heart|kidney|diabetes|diabetic|stroke|angina|infarction", re.I)


def sex_of(pc_key: str) -> str | None:
    groups = pc_key.split(":", 1)[-1].split(",")
    return groups[0][-1] if groups and groups[0][-1:] in ("M", "F") else None


db = sqlite3.connect(STORE_PATH)
rows = db.execute("SELECT disease, pc_groups, result FROM risk_area WHERE pc_groups LIKE ?", (f"v{FORMAT_VERSION}:%",)).fetchall()
failures = defaultdict(list)
chembl_down = False
spread = defaultdict(Counter)
risks_per_sex = Counter()

for disease, pcs, res in rows:
    a, sex, where = json.loads(res), sex_of(pcs), f"{disease} [{pcs.split(':', 1)[-1]}]"
    risks_per_sex[sex] += 1
    items = [("T", i) for i in a["treatments"]] + [("I", i) for i in a["investigational_evidence"]]
    if list(a) != AREA_KEYS:
        failures["format: risk keys"].append(where)
    if set(a["systems"]) - SYSTEMS or ("unlocalized" in a["systems"] and len(a["systems"]) > 1):
        failures["body systems"].append(f"{where}: {a['systems']}")
    if sum(i["evidence"] == "animal" for _, i in items) > 3:
        failures["at most 3 animal-only items"].append(where)
    for kind, i in items:
        name, ke = i["name"], i["key_evidence"] or ""
        spread[sex][name.lower()] += kind == "I"
        tag = f"{where} / {name}"
        if list(i) != ITEM_KEYS:
            failures["format: item keys"].append(tag)
        if not ke or (not i["sources"] and kind == "I"):
            failures["every item explains itself (text and sources)"].append(tag)
        if name.lower() in WRONG_MEDICINES:
            failures["known wrong medicines"].append(tag)
        for p in i["publications"]:
            t = p.get("title") or ""
            if ANY_POPULATION.search(t) or (sex and OTHER_SEX[sex].search(t)):
                failures["paper population fits the patient"].append(f"{tag}: {t[:90]}")
            if _NOT_EFFICACY.search(t) or _DEVICE.search(t):
                failures["papers report treatment results (no cost model, protocol, preclinical, device)"].append(f"{tag}: {t[:90]}")
            if INHERITED.search(t) and not INHERITED.search(disease):
                failures["no rare inherited high-risk population for a general risk"].append(f"{tag}: {t[:90]}")
        if i["evidence"] in ("human", "human_and_animal") and not HUMAN_BASIS.search(ke):
            failures["'human' evidence has a stated basis"].append(f"{tag}: {ke[:90]}")
        if "DrugAge" in i["sources"] and "Animal model:" not in ke:
            failures["animal evidence labelled as animal model"].append(tag)
        if "Evipedia" in i["sources"] and not i["evipedia_url"]:
            failures["Evipedia items link to Evipedia"].append(tag)
        if i["warning"] and PACKAGING.search(i["warning"]):
            failures["warnings are safety warnings, not packaging notices"].append(f"{tag}: {i['warning'][:70]}")
        if kind == "T" and not ke.startswith("FDA-labelled uses for"):
            failures["treatments explain each drug's FDA-labelled use"].append(f"{tag}: {ke[:80]}")
        targets = None
        if i["drugs"] and i["gene_level_grounding"] and not chembl_down:
            try:
                targets = set().union(*(get_target_genes(d) for d in i["drugs"]))
            except ChemblUnavailable:
                chembl_down = True
        if targets is not None and any(g["gene"] not in targets for g in i["gene_level_grounding"]):
            failures["gene grounding = the drug's own ChEMBL targets"].append(tag)
        if name == "Captopril" and i["evidence"] != "animal" and not CAPTOPRIL_HUMAN_OK.search(disease):
            failures["trap: captopril human only where Evipedia/FDA cover the disease"].append(f"{tag}: {i['evidence']}")
    treat_drugs = {d for _, i in items if _ == "T" for d in i["drugs"]}
    if disease == "Hypercholesterolemia" and "amlodipine" in treat_drugs:
        failures["trap: no amlodipine for high cholesterol"].append(where)
    if disease == "Pneumonia" and treat_drugs & {"prednisolone", "prednisone", "methylprednisolone", "dexamethasone"}:
        failures["trap: no corticosteroid for general pneumonia"].append(where)
    if disease == "Pneumonia" and "amoxicillin" in treat_drugs:
        failures["trap: amoxicillin's label says lower respiratory infections, not pneumonia"].append(where)

for drug in sorted({d for _, _, res in rows for t in json.loads(res)["treatments"] for d in t["drugs"]}):
    if chembl_down:
        break
    try:
        if is_withdrawn(drug) or is_marketed(drug) is False:
            failures["no withdrawn or discontinued drug among treatments"].append(drug)
    except ChemblUnavailable:
        chembl_down = True
for sex, counter in spread.items():
    for name, n in counter.items():
        if n > max(10, risks_per_sex[sex] / 2):
            failures["no compound in more than half the risks"].append(f"{name}: {n} of {risks_per_sex[sex]} ({sex})")

checks = ["format: risk keys", "format: item keys", "body systems", "every item explains itself (text and sources)",
          "known wrong medicines", "paper population fits the patient",
          "papers report treatment results (no cost model, protocol, preclinical, device)",
          "no rare inherited high-risk population for a general risk", "'human' evidence has a stated basis",
          "animal evidence labelled as animal model", "at most 3 animal-only items", "Evipedia items link to Evipedia",
          "warnings are safety warnings, not packaging notices", "treatments explain each drug's FDA-labelled use",
          "gene grounding = the drug's own ChEMBL targets", "no withdrawn or discontinued drug among treatments",
          "no compound in more than half the risks", "trap: captopril human only where Evipedia/FDA cover the disease",
          "trap: no amlodipine for high cholesterol", "trap: no corticosteroid for general pneumonia",
          "trap: amoxicillin's label says lower respiratory infections, not pneumonia"]
print(f"checked {len(rows)} stored results (format v{FORMAT_VERSION})\n")
if chembl_down:
    print("NOTE  ChEMBL unreachable: 'gene grounding' and 'withdrawn/discontinued' were SKIPPED, not passed\n")
for c in checks:
    n = len(failures.get(c, []))
    print(f"{'PASS' if not n else 'FAIL'}  {c}" + (f"  ({n})" if n else ""))
for c in checks:
    for f in failures.get(c, [])[:8]:
        print(f"   - {c}: {f}")
sys.exit(1 if failures else 0)
