"""
Cleans GenAge's model-organism gene list (datas/genage_models.csv) into
biology-evidence chunks. Every row here is an animal gene (yeast/worm/fly/
mouse/etc.) with no human relevance on its own — so each row is bridged to
its human ortholog using the real Alliance + FlyBase ortholog data (Saulo's
resources, see ortholog_lookup.py) before it becomes a chunk.

Rows whose gene has no human ortholog match in either resource are dropped —
there's nothing to ground them in human biology, so they're not useful here
(same reasoning that got DrugAge dropped: unlinkable animal data isn't kept).
"""

import csv
import os

from app.services.ortholog_lookup import lookup_ortholog, lookup_fly_human_disease

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../../datas/genage_models.csv")


def _find_human_ortholog(symbol: str, organism: str) -> tuple[dict | None, str | None]:
    """Best human ortholog match + any linked OMIM disease names, or (None, None) if unmatched."""
    human_matches = [m for m in lookup_ortholog(symbol) if m["ortholog_species"] == "Homo sapiens"]
    best = next((m for m in human_matches if m["is_best_score"]), human_matches[0] if human_matches else None)

    omim_diseases = None
    if organism == "Drosophila melanogaster":
        for fly_match in lookup_fly_human_disease(symbol):
            if fly_match.get("omim_diseases"):
                omim_diseases = fly_match["omim_diseases"]
                if best is None:
                    best = {
                        "ortholog_symbol": fly_match["human_gene_symbol"],
                        "confidence": f"DIOPT score {fly_match['diopt_score']}",
                    }
                break

    return best, omim_diseases


def load_genage_models_chunks(csv_path: str = None) -> list[dict]:
    path = csv_path or DATA_PATH
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    chunks = []
    for row in rows:
        symbol = row.get("symbol", "").strip()
        organism = row.get("organism", "").strip()
        if not symbol or not organism:
            continue

        best, omim_diseases = _find_human_ortholog(symbol, organism)
        if best is None:
            continue  # no human ortholog found — not usable here

        lifespan_effect = row.get("lifespan effect", "").strip() or "unspecified"
        longevity_influence = row.get("longevity influence", "").strip() or "unspecified"

        raw_text = (
            f"Animal gene: {symbol} ({row.get('name', '').strip()}) in {organism}. "
            f"Evidence tier: animal_model. Human ortholog: {best['ortholog_symbol']} "
            f"(confidence: {best['confidence']}). "
            f"Lifespan effect when manipulated: {lifespan_effect}. "
            f"Longevity influence: {longevity_influence}. "
            f"Linked human diseases (via fly ortholog): {omim_diseases or 'none known'}. "
            f"Source: GenAge (model organisms), bridged to human via Alliance/FlyBase ortholog data."
        )

        chunks.append({
            "type": "gene",
            "name": best["ortholog_symbol"],
            "animal_gene_symbol": symbol,
            "animal_gene_name": row.get("name", "").strip(),
            "organism": organism,
            "category": "gene",
            "source": "GenAge (models, ortholog-bridged)",
            "evidence_tier": "animal_model",
            "ortholog_confidence": best["confidence"],
            "lifespan_effect": lifespan_effect,
            "longevity_influence": longevity_influence,
            "omim_diseases": omim_diseases,
            "target_pc_groups": [],
            "target_conditions": [],
            "raw_text": raw_text,
        })

    return chunks


if __name__ == "__main__":
    result = load_genage_models_chunks()
    print(f"Loaded {len(result)} ortholog-bridged GenAge-models chunks")
    print()
    print("Sample chunk:")
    print(result[0])
