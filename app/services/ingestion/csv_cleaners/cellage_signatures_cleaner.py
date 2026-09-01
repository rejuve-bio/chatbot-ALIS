"""
Cleans CellAge's gene expression signatures (data/cell_signatures.csv) into
biology-evidence chunks. Which genes are statistically over/under-expressed
during cellular senescence. Mechanism-level, no disease/PC link.
"""

import csv
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../../datas/cell_signatures.csv")


def load_cellage_signature_chunks(csv_path: str = None) -> list[dict]:
    path = csv_path or DATA_PATH
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    chunks = []
    for row in rows:
        symbol = row.get("gene_symbol", "").strip()
        if not symbol:
            continue

        try:
            overexp = int(row.get("ovevrexp", 0) or 0)
            underexp = int(row.get("underexp", 0) or 0)
        except ValueError:
            overexp, underexp = 0, 0

        direction = "overexpressed" if overexp > underexp else "underexpressed" if underexp > overexp else "mixed"

        raw_text = (
            f"Gene: {symbol} ({row.get('gene_name', '').strip()}). Evidence tier: human. "
            f"Consistently {direction} during cellular senescence "
            f"(studies overexpressed: {overexp}, underexpressed: {underexp}). "
            f"p-value: {row.get('p_value', '').strip()}. "
            f"Source: CellAge expression signatures."
        )

        chunks.append({
            "type": "gene",
            "name": symbol,
            "gene_name": row.get("gene_name", "").strip(),
            "category": "gene",
            "source": "CellAge expression signatures",
            "evidence_tier": "human",
            "expression_direction": direction,
            "p_value": row.get("p_value", "").strip(),
            "target_pc_groups": [],
            "target_conditions": [],
            "raw_text": raw_text,
        })

    return chunks


if __name__ == "__main__":
    result = load_cellage_signature_chunks()
    print(f"Loaded {len(result)} CellAge signature chunks")
    print(result[0])
