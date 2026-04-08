# app/services/codebook.py

import pandas as pd
import os

_codebook = {}

def load_codebook(csv_path: str = None):
    global _codebook
    if not csv_path:
        csv_path = os.path.join(os.path.dirname(__file__), "../../data/codebook_linAge2.csv")
    
    df = pd.read_csv(csv_path)
    # codebook has Var and Human columns
    for _, row in df.iterrows():
        code = str(row.get("Var", "")).strip()
        human = str(row.get("Human", "")).strip()
        if code and human and code != "nan" and human != "nan":
            _codebook[code] = human


def get_label(code: str) -> str:
    if not _codebook:
        load_codebook()
    return _codebook.get(code, code)