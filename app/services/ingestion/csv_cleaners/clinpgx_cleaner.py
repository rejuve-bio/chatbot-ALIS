"""
Cleans ClinPGx's relationships.tsv (datas/clinpgx_relationships.tsv) into
biology-evidence chunks. Gene <-> Chemical (drug) relationships only — this is
the "gene & pathway level drug targeting data" Mike flagged this source for
(grounding which gene a drug actually targets), not personalized
pharmacogenomics (we don't have patient genotype data to use that part).

CC BY-SA 4.0 licensed — commercially usable, confirmed from the real
downloaded LICENSE.txt, unlike DrugBank.
"""

import csv
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../../datas/clinpgx_relationships.tsv")


def load_clinpgx_chunks(tsv_path: str = None) -> list[dict]:
    path = tsv_path or DATA_PATH
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    chunks = []
    seen_pairs = set()

    for row in rows:
        e1_type, e2_type = row.get("Entity1_type"), row.get("Entity2_type")
        if not (e1_type == "Gene" and e2_type == "Chemical"):
            continue  # only Gene->Chemical direction — the reverse row is a duplicate
        if row.get("Association") != "associated":
            continue  # drop "not associated" and "ambiguous" for clean signal

        gene = row.get("Entity1_name", "").strip()
        drug = row.get("Entity2_name", "").strip()
        if not gene or not drug:
            continue

        pair_key = (gene, drug)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        pk = "yes" if row.get("PK") else "no"
        pd = "yes" if row.get("PD") else "no"
        pmids = row.get("PMIDs", "").strip()

        raw_text = (
            f"Gene: {gene}. Targeted by drug: {drug}. Evidence tier: human. "
            f"Pharmacokinetic relevance: {pk}. Pharmacodynamic relevance: {pd}. "
            f"Source: ClinPGx. PubMed references: {pmids if pmids else 'none'}."
        )

        chunks.append({
            "type": "gene_drug_relationship",
            "name": gene,
            "drug_name": drug,
            "category": "gene",
            "source": "ClinPGx",
            "evidence_tier": "human",
            "pk_relevant": row.get("PK", "").strip() == "PK",
            "pd_relevant": row.get("PD", "").strip() == "PD",
            "pmids": pmids,
            "target_pc_groups": [],
            "target_conditions": [],
            "raw_text": raw_text,
        })

    return chunks


if __name__ == "__main__":
    result = load_clinpgx_chunks()
    print(f"Loaded {len(result)} ClinPGx gene-drug chunks")
    print(result[0])
