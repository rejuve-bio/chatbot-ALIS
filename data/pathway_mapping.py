"""
Mechanism -> pathway mapping to Gene Ontology (GO) biological processes and Reactome pathways.

Source: the reviewed pathway table (pathway_go_reactome_mapping.csv, October 2026), which maps every
pathway term of the LinAge2 mechanism categories to GO and Reactome. Only rows rated High or
Medium are kept; Low and no-match rows (e.g. AMPK activation, tau pathology, mitochondrial
dysfunction, cardiac remodeling, tumor suppressor signaling, tissue remodeling) are left out.
Still under review; update this file when the table changes.

Per row: the pathway term, its GO biological-process IDs, and its Reactome pathway (or None when
only GO matched).
"""

_ENDOTHELIAL_FUNCTION = {
    "pathway": "endothelial function",
    "go_ids": ["GO:0001935", "GO:0043542", "GO:0043534", "GO:0003158", "GO:0048010"],
    "go_name": "endothelial cell proliferation and migration; endothelium development; VEGF receptor signaling",
    "reactome_id": "R-HSA-9860931", "reactome_name": "Response of endothelial cells to shear stress",
    "confidence": "Medium",
}
_ARTERIAL_STIFFNESS = {
    "pathway": "arterial stiffness",
    "go_ids": [], "go_name": None,
    "reactome_id": "R-HSA-5683057", "reactome_name": "MAPK family signaling cascades",
    "confidence": "Medium",
}
_OXIDATIVE_STRESS = {
    "pathway": "oxidative stress",
    "go_ids": ["GO:0006979"], "go_name": "response to oxidative stress",
    "reactome_id": "R-HSA-3299685", "reactome_name": "Detoxification of Reactive Oxygen Species",
    "confidence": "High",
}
_LIPID_METABOLISM = {
    "pathway": "lipid metabolism",
    "go_ids": ["GO:0006629"], "go_name": "lipid metabolic process",
    "reactome_id": "R-HSA-556833", "reactome_name": "Metabolism of lipids",
    "confidence": "High",
}
_AIRWAY_INFLAMMATION = {
    "pathway": "airway inflammation",
    "go_ids": [], "go_name": None,
    "reactome_id": "R-HSA-9818026", "reactome_name": "NFE2L2 regulating inflammation associated genes",
    "confidence": "Medium",
}

PATHWAY_MAPPING: dict[str, list[dict]] = {
    "Metabolic aging": [
        {"pathway": "insulin signaling", "go_ids": ["GO:0008286"], "go_name": "insulin receptor signaling pathway",
         "reactome_id": "R-HSA-74752", "reactome_name": "Signaling by Insulin receptor", "confidence": "High"},
        {"pathway": "mTOR signaling", "go_ids": ["GO:0031929"], "go_name": "TOR signaling",
         "reactome_id": "R-HSA-165159", "reactome_name": "MTOR signalling", "confidence": "High"},
        {"pathway": "glucose regulation", "go_ids": ["GO:0042593"], "go_name": "glucose homeostasis",
         "reactome_id": "R-HSA-70326", "reactome_name": "Glucose metabolism", "confidence": "High"},
        _LIPID_METABOLISM,
    ],
    "Vascular aging": [
        _ENDOTHELIAL_FUNCTION,
        _ARTERIAL_STIFFNESS,
        {"pathway": "angiogenesis", "go_ids": ["GO:0001525"], "go_name": "angiogenesis",
         "reactome_id": "R-HSA-4420097", "reactome_name": "VEGFA-VEGFR2 Pathway", "confidence": "High"},
        {"pathway": "nitric oxide signaling", "go_ids": ["GO:0007263"], "go_name": "nitric oxide mediated signal transduction",
         "reactome_id": "R-HSA-202131", "reactome_name": "Metabolism of nitric oxide: NOS3 activation and regulation",
         "confidence": "High"},
    ],
    "Vascular aging (smoking-related)": [
        _ENDOTHELIAL_FUNCTION,
        _OXIDATIVE_STRESS,
        _ARTERIAL_STIFFNESS,
        {"pathway": "platelet aggregation", "go_ids": ["GO:0070527"], "go_name": "platelet aggregation",
         "reactome_id": "R-HSA-76009", "reactome_name": "Platelet Aggregation (Plug Formation)", "confidence": "High"},
    ],
    "Smoking-related": [
        _OXIDATIVE_STRESS,
        {"pathway": "xenobiotic metabolism", "go_ids": ["GO:0006805"], "go_name": "xenobiotic metabolic process",
         "reactome_id": "R-HSA-211981", "reactome_name": "Xenobiotics", "confidence": "High"},
        _AIRWAY_INFLAMMATION,
    ],
    "Inflammation": [
        {"pathway": "NF-kB signaling", "go_ids": ["GO:0007249"], "go_name": "canonical NF-kappaB signal transduction",
         "reactome_id": "R-HSA-5357956", "reactome_name": "TNFR1-induced NF-kappa-B signaling pathway", "confidence": "High"},
        {"pathway": "cytokine regulation", "go_ids": ["GO:0019221"], "go_name": "cytokine-mediated signaling pathway",
         "reactome_id": "R-HSA-1280215", "reactome_name": "Cytokine Signaling in Immune system", "confidence": "High"},
        {"pathway": "senescence-associated secretory phenotype", "go_ids": ["GO:0090398"], "go_name": "cellular senescence",
         "reactome_id": "R-HSA-2559582", "reactome_name": "Senescence-Associated Secretory Phenotype (SASP)",
         "confidence": "High"},
        {"pathway": "innate immune activation", "go_ids": ["GO:0045087"], "go_name": "innate immune response",
         "reactome_id": "R-HSA-168249", "reactome_name": "Innate Immune System", "confidence": "High"},
    ],
    "Neurodegeneration": [
        {"pathway": "amyloid clearance", "go_ids": ["GO:0097242"], "go_name": "amyloid-beta clearance",
         "reactome_id": None, "reactome_name": None, "confidence": "High"},
        {"pathway": "synaptic plasticity", "go_ids": ["GO:0048167"], "go_name": "regulation of synaptic plasticity",
         "reactome_id": None, "reactome_name": None, "confidence": "High"},
        {"pathway": "neuroinflammation", "go_ids": ["GO:0150076"], "go_name": "neuroinflammatory response",
         "reactome_id": None, "reactome_name": None, "confidence": "High"},
    ],
    "Cancer-related": [
        {"pathway": "DNA damage response", "go_ids": ["GO:0006974"], "go_name": "DNA damage response",
         "reactome_id": "R-HSA-73894", "reactome_name": "DNA Repair", "confidence": "High"},
        {"pathway": "cell cycle regulation", "go_ids": ["GO:0051726"], "go_name": "regulation of cell cycle",
         "reactome_id": "R-HSA-1640170", "reactome_name": "Cell Cycle", "confidence": "High"},
        {"pathway": "apoptosis", "go_ids": ["GO:0006915"], "go_name": "apoptotic process",
         "reactome_id": "R-HSA-109581", "reactome_name": "Apoptosis", "confidence": "High"},
    ],
    "Cardiac disease-related": [
        _LIPID_METABOLISM,
        _ARTERIAL_STIFFNESS,
        {"pathway": "renin-angiotensin system", "go_ids": ["GO:0003081"],
         "go_name": "regulation of systemic arterial blood pressure by renin-angiotensin",
         "reactome_id": None, "reactome_name": None, "confidence": "High"},
    ],
    "Lung disease-related": [
        _OXIDATIVE_STRESS,
        _AIRWAY_INFLAMMATION,
    ],
}
