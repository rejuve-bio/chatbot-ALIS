"""
★ SYNTHETIC DRAFT — NOT clinician-reviewed ★
"""

import logging

logger = logging.getLogger(__name__)

SYNTHETIC_PATHWAY_TABLE: dict[str, list[str]] = {
    "Metabolic aging": [
        "insulin signaling", "mTOR signaling", "AMPK activation",
        "glucose regulation", "lipid metabolism",
    ],
    "Vascular aging": [
        "endothelial function", "arterial stiffness", "angiogenesis",
        "nitric oxide signaling",
    ],
    "Vascular aging (smoking-related)": [
        "endothelial function", "oxidative stress", "arterial stiffness",
        "platelet aggregation",
    ],
    "Smoking-related": [
        "oxidative stress", "xenobiotic metabolism", "airway inflammation",
    ],
    "Inflammation": [
        "NF-kB signaling", "cytokine regulation",
        "senescence-associated secretory phenotype", "innate immune activation",
    ],
    "Neurodegeneration": [
        "amyloid clearance", "synaptic plasticity", "neuroinflammation",
        "tau pathology", "mitochondrial dysfunction",
    ],
    "Cancer-related": [
        "DNA damage response", "cell cycle regulation", "apoptosis",
        "tumor suppressor signaling",
    ],
    "Cardiac disease-related": [
        "cardiac remodeling", "lipid metabolism", "arterial stiffness",
        "renin-angiotensin system",
    ],
    "Lung disease-related": [
        "oxidative stress", "airway inflammation", "tissue remodeling",
    ],
}


def expand_mechanism(mechanism_tag: str) -> list[str]:

    pathways = SYNTHETIC_PATHWAY_TABLE.get(mechanism_tag)
    if pathways is None:
        logger.info(f"expand_mechanism | '{mechanism_tag}' not in table — using raw tag as fallback")
        return [mechanism_tag]
    logger.info(f"expand_mechanism | input='{mechanism_tag}' | output={pathways}")
    return pathways


def expand_mechanisms(mechanism_tags: list[str]) -> list[str]:
    """Expands a whole mechanism list, deduped, preserving first-seen order."""
    expanded: list[str] = []
    for tag in mechanism_tags:
        for term in expand_mechanism(tag):
            if term not in expanded:
                expanded.append(term)
    return expanded
