"""
Avatar body systems for each patient risk (the "systems" field), per body-systems.md.

Reviewed by hand for every risk name ALIS sends, because MeSH files diseases in several branches
for bookkeeping (Asthma under immune, Osteoporosis under metabolic) and doesn't know names like
"Hip fracture". A risk can list one to three systems, main one first; "unlocalized" only when no
organ system fits and never together with one. A risk name not listed here falls back to MeSH.
"""

import logging

from app.services.mesh_service import get_body_systems

logger = logging.getLogger(__name__)

RISK_SYSTEMS: dict[str, list[str]] = {
    "Acute myocardial infarction": ["cardiovascular"],
    "Alzheimer's disease": ["neurological"],
    "Anemia": ["metabolic"],
    "Angina": ["cardiovascular"],
    "Arthritis": ["musculoskeletal"],
    "Asthma": ["respiratory"],
    "Cancer": ["unlocalized"],
    "Cardiovascular disease": ["cardiovascular"],
    "Cardiovascular disease, including congestive cardiac failure": ["cardiovascular"],
    "Chronic bronchitis": ["respiratory"],
    "Chronic kidney disease": ["renal"],
    "Chronic liver disease": ["hepatic"],
    "Chronic lung disease": ["respiratory"],
    "Chronic lung diseases (asthma, chronic bronchitis, emphysema)": ["respiratory"],
    "Cognitive impairment": ["neurological"],
    "Congestive cardiac failure": ["cardiovascular"],
    "Diabetes mellitus": ["metabolic", "cardiovascular", "renal"],
    "Diabetes mellitus and insulin use": ["metabolic"],
    "Diabetic complications": ["metabolic", "cardiovascular", "renal"],
    "Diabetic foot ulcers": ["skin", "metabolic"],
    "Diabetic peripheral neuropathy": ["neurological", "metabolic"],
    "Diabetic retinopathy": ["vision", "metabolic"],
    "Emphysema": ["respiratory"],
    "General Age-Related Decline": ["unlocalized"],
    "Hip fracture": ["musculoskeletal"],
    "Hypercholesterolemia": ["metabolic"],
    "Hypertension": ["cardiovascular"],
    "Insulin use": ["metabolic"],
    "Ischemic heart disease": ["cardiovascular"],
    "Obesity": ["metabolic"],
    "Osteoporosis": ["musculoskeletal"],
    "Peripheral arterial disease": ["cardiovascular"],
    "Pneumonia": ["respiratory"],
    "Spine fracture": ["musculoskeletal"],
    "Stroke": ["cardiovascular", "neurological"],
    "Thyroid disease": ["metabolic"],
    "Visual impairment": ["vision"],
    "Wrist fracture": ["musculoskeletal"],
}


def _norm(name: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in name.lower()).split())


_BY_NORM = {_norm(k): v for k, v in RISK_SYSTEMS.items()}


def listed_systems(disease_name: str) -> list[str] | None:
    """The reviewed systems for this risk name, or None when it isn't listed."""
    systems = _BY_NORM.get(_norm(disease_name))
    return list(systems) if systems else None


def get_risk_systems(disease_name: str, mesh_headings: list[str]) -> list[str]:
    systems = listed_systems(disease_name)
    if systems:
        return systems
    logger.warning(f"risk_systems | {disease_name!r} is not in RISK_SYSTEMS; using MeSH (add it to the table)")
    return get_body_systems(mesh_headings or [disease_name])
