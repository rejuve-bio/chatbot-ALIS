"""Cleans the raw DrugAge CSV into intervention chunks. Animal-model only, so every chunk is tagged evidence_tier="animal_model"."""

import csv
import os
from collections import defaultdict

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../../datas/drugage.csv")


def _load_raw_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_significant(row: dict) -> bool:
    return row.get("avg_lifespan_significance") == "S"


def load_drugage_chunks(csv_path: str = None) -> list[dict]:
    """One chunk per compound, statistically significant entries only."""
    path = csv_path or DATA_PATH
    rows = _load_raw_rows(path)
    significant = [r for r in rows if _is_significant(r)]

    by_compound: dict[str, list[dict]] = defaultdict(list)
    for row in significant:
        by_compound[row["compound_name"]].append(row)

    chunks = []
    for compound, entries in by_compound.items():
        species = sorted({e["species"] for e in entries if e["species"]})
        itp_validated = any(e["ITP"] == "Yes" for e in entries)
        pubmed_ids = sorted({e["pubmed_id"] for e in entries if e["pubmed_id"]})

        evidence = [
            {
                "species": e["species"] or None,
                "dosage": e["dosage"] or None,
                "avg_lifespan_change_percent": e["avg_lifespan_change_percent"] or None,
                "max_lifespan_change_percent": e["max_lifespan_change_percent"] or None,
                "gender": e["gender"] or None,
                "itp_validated": e["ITP"] == "Yes",
                "pubmed_id": e["pubmed_id"] or None,
            }
            for e in entries
        ]

        raw_text = (
            f"Compound: {compound}. Evidence tier: animal model (non-human). "
            f"Tested in: {', '.join(species) if species else 'unspecified species'}. "
            f"ITP cross-validated: {'yes' if itp_validated else 'no'}. "
            f"Statistically significant lifespan effect reported in {len(entries)} "
            f"study result(s) (DrugAge). "
            f"PubMed references: {', '.join(pubmed_ids) if pubmed_ids else 'none'}."
        )

        chunks.append({
            "type": "intervention",
            "name": compound,
            "category": "drug",
            "source": "DrugAge",
            "evidence_tier": "animal_model",
            "organism": species,
            "itp_validated": itp_validated,
            "target_pc_groups": [],
            "target_conditions": [],
            "evidence": evidence,
            "raw_text": raw_text,
        })

    return chunks


if __name__ == "__main__":
    result = load_drugage_chunks()
    print(f"Loaded {len(result)} significant, per-compound chunks from DrugAge "
          f"(filtered from raw rows, animal-model evidence only)")
    print()
    print("Sample chunk:")
    print(result[0])
