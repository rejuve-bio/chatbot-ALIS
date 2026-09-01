"""
Cleans GenAge's human gene list (data/genage_human.csv) into biology-evidence
chunks. Human gene-level evidence, tagged with GenAge's own confidence level
(the "why" field) — no disease or PC-group link, mechanism-level only.
"""

import csv
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../../datas/genage_human.csv")

_WHY_LABELS = {
    "functional": "direct experimental evidence in humans",
    "putative": "suspected but not experimentally confirmed",
    "mammal": "evidence from mammal studies",
    "model": "inferred from a model-organism homolog",
    "downstream": "downstream of a known ageing pathway",
    "upstream": "upstream of a known ageing pathway",
    "human_link": "linked to human disease/trait via genetic association",
    "cell": "evidence from cell-based studies",
}


def _describe_why(why_field: str) -> str:
    tags = [t.strip() for t in why_field.split(",") if t.strip()]
    return "; ".join(_WHY_LABELS.get(t, t) for t in tags) or "unspecified evidence type"


def load_genage_human_chunks(csv_path: str = None) -> list[dict]:
    path = csv_path or DATA_PATH
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    chunks = []
    for row in rows:
        symbol = row.get("symbol", "").strip()
        if not symbol:
            continue

        why = row.get("why", "").strip()
        why_desc = _describe_why(why)

        raw_text = (
            f"Gene: {symbol} ({row.get('name', '').strip()}). Evidence tier: human. "
            f"GenAge evidence type: {why_desc}. "
            f"Source: GenAge human gene database."
        )

        chunks.append({
            "type": "gene",
            "name": symbol,
            "gene_name": row.get("name", "").strip(),
            "category": "gene",
            "source": "GenAge (human)",
            "evidence_tier": "human",
            "evidence_type": why,
            "target_pc_groups": [],
            "target_conditions": [],
            "raw_text": raw_text,
        })

    return chunks


if __name__ == "__main__":
    result = load_genage_human_chunks()
    print(f"Loaded {len(result)} GenAge human gene chunks")
    print(result[0])
