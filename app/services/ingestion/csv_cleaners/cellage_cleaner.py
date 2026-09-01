"""
Cleans CellAge (data/cellage3.tsv) into biology-evidence chunks. Human
cellular-senescence gene data — whether a gene induces or inhibits cells
becoming senescent, and in what context. Mechanism-level, no disease/PC link.
"""

import csv
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../../datas/cellage3.tsv")


def load_cellage_chunks(tsv_path: str = None) -> list[dict]:
    path = tsv_path or DATA_PATH
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    chunks = []
    for row in rows:
        symbol = row.get("Gene symbol", "").strip()
        if not symbol:
            continue

        effect = row.get("Senescence Effect", "").strip() or "Unclear"
        sen_type = row.get("Type of senescence", "").strip() or "Unclear"
        cancer_cell = row.get("Cancer Cell", "").strip() or "Unknown"

        if effect == "Unclear":
            continue  # drop indefinite results — keeps only definitive Induces/Inhibits signal

        raw_text = (
            f"Gene: {symbol} ({row.get('Gene name', '').strip()}). Evidence tier: human. "
            f"Senescence effect: {effect} cellular senescence. "
            f"Type of senescence studied: {sen_type}. "
            f"Studied in a cancer cell line: {cancer_cell}. "
            f"Source: CellAge database. Reference: {row.get('Reference', '').strip()}."
        )

        chunks.append({
            "type": "gene",
            "name": symbol,
            "gene_name": row.get("Gene name", "").strip(),
            "category": "gene",
            "source": "CellAge",
            "evidence_tier": "human",
            "senescence_effect": effect,
            "senescence_type": sen_type,
            "cancer_cell_studied": cancer_cell,
            "target_pc_groups": [],
            "target_conditions": [],
            "raw_text": raw_text,
        })

    return chunks


if __name__ == "__main__":
    result = load_cellage_chunks()
    print(f"Loaded {len(result)} CellAge gene chunks")
    print(result[0])
