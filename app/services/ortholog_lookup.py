"""
Gene-ortholog lookup 
- Alliance ortholog data: gene match across 5 species, with an algorithm
  agreement count as a confidence score.
- FlyBase dmel->human file: fly gene -> human ortholog, with a DIOPT
  confidence score and, where known, linked OMIM human diseases.
"""

import csv
import gzip
import os
from functools import lru_cache

ALLIANCE_PATH = os.path.join(os.path.dirname(__file__), "../../datas/alliance_ortholog_5species.tsv.gz")
FLYBASE_PATH = os.path.join(os.path.dirname(__file__), "../../datas/flybase_dmel_human_orthologs_disease.tsv.gz")


@lru_cache(maxsize=1)
def _load_alliance_index() -> dict[str, list[dict]]:
    """Gene symbol -> list of ortholog matches across the 5 Alliance species."""
    index: dict[str, list[dict]] = {}
    with gzip.open(ALLIANCE_PATH, "rt") as f:
        reader = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        for row in reader:
            for gene_key, other_key in (("Gene1Symbol", "Gene2"), ("Gene2Symbol", "Gene1")):
                symbol = row.get(gene_key, "").strip()
                if not symbol:
                    continue
                index.setdefault(symbol, []).append({
                    "ortholog_symbol": row.get(f"{other_key}Symbol"),
                    "ortholog_species": row.get(f"{other_key}SpeciesName"),
                    "confidence": f"{row.get('AlgorithmsMatch')}/{row.get('OutOfAlgorithms')} algorithms agree",
                    "is_best_score": row.get("IsBestScore") == "Yes",
                })
    return index


@lru_cache(maxsize=1)
def _load_flybase_index() -> dict[str, list[dict]]:
    """Fly gene symbol -> list of human ortholog + disease matches."""
    index: dict[str, list[dict]] = {}
    with gzip.open(FLYBASE_PATH, "rt") as f:
        reader = csv.DictReader((line for line in f if not line.startswith("#") and line.strip()), delimiter="\t",
                                 fieldnames=["Dmel_gene_ID", "Dmel_gene_symbol", "Human_gene_HGNC_ID",
                                             "Human_gene_OMIM_ID", "Human_gene_symbol", "DIOPT_score",
                                             "OMIM_Phenotype_IDs", "OMIM_Phenotype_names"])
        for row in reader:
            symbol = row.get("Dmel_gene_symbol", "").strip()
            if not symbol:
                continue
            index.setdefault(symbol, []).append({
                "human_gene_symbol": row.get("Human_gene_symbol"),
                "diopt_score": row.get("DIOPT_score"),
                "omim_diseases": row.get("OMIM_Phenotype_names", "").strip() or None,
            })
    return index


def lookup_ortholog(gene_symbol: str) -> list[dict]:
    """Cross-species matches from the Alliance data for any gene symbol."""
    return _load_alliance_index().get(gene_symbol, [])


def lookup_fly_human_disease(fly_gene_symbol: str) -> list[dict]:
    """Human ortholog + linked OMIM diseases for a specific fly gene symbol."""
    return _load_flybase_index().get(fly_gene_symbol, [])


if __name__ == "__main__":
    print("Alliance lookup for 'Cyp4f18':")
    print(lookup_ortholog("Cyp4f18"))
    print()
    print("FlyBase disease lookup for 'Egfr':")
    print(lookup_fly_human_disease("Egfr"))
