# app/services/parsers/excel_parser.py

import pandas as pd
import io
from typing import Optional


def parse_excel(file_bytes: bytes, patient_id: str, pc_group: Optional[str] = None) -> list[dict]:
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    fixed_cols = ["parName", "Variable Name (SI units per NHANES IV)", "Reference ranges (Adult, male)"]
    timepoint_cols = [c for c in df.columns if c not in fixed_cols]

    chunks = []

    for _, row in df.iterrows():
        param_code = str(row.get("parName", "")).strip()
        param_name = str(row.get("Variable Name (SI units per NHANES IV)", "")).strip()
        ref_range = str(row.get("Reference ranges (Adult, male)", "")).strip()

        if not param_code or param_code == "nan":
            continue

        readings = {}
        for col in timepoint_cols:
            val = row.get(col)
            try:
                readings[col] = float(val)
            except:
                continue

        text_summary = (
            f"Patient {patient_id} | {param_name} ({param_code}) | "
            f"Reference range: {ref_range} | "
            f"Readings: {', '.join([f'{k}: {v}' for k, v in readings.items()])} | "
            f"PC group: {pc_group or 'unknown'}."
        )

        chunks.append({
            "patient_id": patient_id,
            "param_code": param_code,
            "param_name": param_name,
            "reference_range": ref_range,
            "readings": readings,
            "pc_group": pc_group,
            "text_summary": text_summary
        })

    return chunks