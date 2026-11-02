"""
Terminology only: how each patient-risk name is spelled in the external APIs. No clinical
content lives here — which drugs treat a disease comes from ChEMBL, FDA labels and NLM MED-RT.

- mesh_headings: MeSH disease headings as ChEMBL's drug_indication records them
- mesh_ids: MeSH descriptor IDs as NLM RxClass (MED-RT) uses them
- label_terms: how FDA labels name the disease in their indications section
- label_exclude_terms: words that mark a different condition sharing a label term
  ("pulmonary" arterial hypertension, diabetic "nephropathy")
- trial_condition: the condition to search ClinicalTrials.gov with

Replace with codes sent by the ALIS backend once those are available.
"""

DISEASE_TERMS: dict[str, dict] = {
    "Diabetes mellitus": {
        "mesh_headings": ["Diabetes Mellitus, Type 2"],
        "mesh_ids": ["D003924"],
        "label_terms": ["type 2 diabetes"],
        "label_exclude_terms": ["nephropathy", "neuropathy", "retinopathy", "macular"],
        "trial_condition": "Type 2 Diabetes",
    },
    "Hypertension": {
        "mesh_headings": ["Hypertension"],
        "mesh_ids": ["D006973"],
        "label_terms": ["hypertension"],
        "label_exclude_terms": ["pulmonary", "ocular", "intraocular", "portal", "intracranial"],
        "trial_condition": "Hypertension",
    },
    "Thyroid disease": {
        "mesh_headings": ["Hypothyroidism", "Hyperthyroidism"],
        "mesh_ids": ["D007037", "D006980"],
        "label_terms": ["hypothyroidism", "hyperthyroidism"],
        "label_exclude_terms": ["carcinoma", "cancer", "lipid", "cholesterol", "triglyceride"],
        "trial_condition": "Thyroid Diseases",
    },
    "Chronic lung disease": {
        "mesh_headings": ["Pulmonary Disease, Chronic Obstructive"],
        "mesh_ids": ["D029424"],
        "label_terms": ["chronic obstructive pulmonary disease", "copd"],
        "trial_condition": "COPD",
    },
    "Cognitive impairment": {
        "mesh_headings": ["Cognitive Dysfunction", "Alzheimer Disease"],
        "mesh_ids": ["D060825", "D000544"],
        "label_terms": ["mild cognitive impairment", "alzheimer"],
        "trial_condition": "Cognitive Impairment",
    },
    "Alzheimer's disease": {
        "mesh_headings": ["Alzheimer Disease"],
        "mesh_ids": ["D000544"],
        "label_terms": ["alzheimer"],
        "trial_condition": "Alzheimer Disease",
    },
    "Chronic kidney disease": {
        "mesh_headings": ["Renal Insufficiency, Chronic"],
        "mesh_ids": ["D051436"],
        "label_terms": ["chronic kidney disease"],
        "trial_condition": "Chronic Kidney Disease",
    },
    "Anemia": {
        "mesh_headings": ["Anemia, Iron-Deficiency", "Anemia"],
        "mesh_ids": ["D000740"],
        "label_terms": ["anemia"],
        "label_exclude_terms": ["myelodysplastic", "leukemia", "lymphoma", "myeloma", "aplastic"],
        "trial_condition": "Anemia",
    },
    "Cardiovascular disease, including congestive cardiac failure": {
        "mesh_headings": ["Heart Failure"],
        "mesh_ids": ["D006333"],
        "label_terms": ["heart failure"],
        "trial_condition": "Heart Failure",
    },
    "Cardiovascular disease": {
        "mesh_headings": ["Cardiovascular Diseases", "Coronary Artery Disease", "Myocardial Infarction"],
        "mesh_ids": ["D002318", "D003324"],
        "label_terms": ["myocardial infarction", "cardiovascular death", "coronary artery disease"],
        "trial_condition": "Cardiovascular Diseases",
    },
    "Chronic liver disease": {
        "mesh_headings": ["Non-alcoholic Fatty Liver Disease", "Liver Cirrhosis, Biliary"],
        "mesh_ids": ["D065626", "D008107"],
        "label_terms": ["steatohepatitis", "primary biliary"],
        "trial_condition": "Chronic Liver Disease",
    },
    "Angina": {
        "mesh_headings": ["Angina Pectoris"],
        "mesh_ids": ["D000787"],
        "label_terms": ["angina"],
        "trial_condition": "Angina Pectoris",
    },
    "Visual impairment": {
        "mesh_headings": ["Macular Degeneration", "Glaucoma, Open-Angle", "Macular Edema"],
        "mesh_ids": ["D008268", "D005902"],
        "label_terms": ["macular degeneration", "glaucoma", "macular edema"],
        "trial_condition": "Vision Disorders",
    },
    "Arthritis": {
        "mesh_headings": ["Osteoarthritis", "Arthritis, Rheumatoid"],
        "mesh_ids": ["D010003", "D001172"],
        "label_terms": ["osteoarthritis", "rheumatoid arthritis"],
        "trial_condition": "Arthritis",
    },
    "Cancer": {
        "mesh_headings": [],
        "literature_headings": ["Neoplasms/prevention and control"],
        "mesh_ids": ["D009369"],
        "label_terms": [],
        "trial_condition": "Cancer Prevention",
    },
    "Osteoporosis": {
        "mesh_headings": ["Osteoporosis", "Osteoporosis, Postmenopausal"],
        "label_terms": ["osteoporosis"],
        "trial_condition": "Osteoporosis",
    },
    "Pneumonia": {
        "mesh_headings": ["Pneumonia", "Pneumonia, Pneumococcal"],
        "label_terms": ["pneumonia"],
        "trial_condition": "Pneumonia",
    },
    "Stroke": {
        "mesh_headings": ["Stroke", "Ischemic Stroke"],
        "label_terms": ["stroke"],
        "trial_condition": "Stroke",
    },
    "Obesity": {
        "mesh_headings": ["Obesity"],
        "label_terms": ["obesity", "chronic weight management"],
        "trial_condition": "Obesity",
    },
    "Hypercholesterolemia": {
        "mesh_headings": ["Hypercholesterolemia", "Hyperlipidemias"],
        "label_terms": ["hypercholesterolemia", "hyperlipidemia", "ldl-c"],
        "trial_condition": "Hypercholesterolemia",
    },
    "Diabetes mellitus and insulin use": {
        "mesh_headings": ["Diabetes Mellitus, Type 2"],
        "label_terms": ["type 2 diabetes"],
        "label_exclude_terms": ["nephropathy", "neuropathy", "retinopathy", "macular"],
        "trial_condition": "Type 2 Diabetes",
    },
    "Diabetic complications": {
        "mesh_headings": ["Diabetic Nephropathies", "Diabetic Neuropathies", "Diabetic Retinopathy"],
        "label_terms": ["diabetic nephropathy", "diabetic neuropath", "diabetic peripheral neuropath",
                        "diabetic retinopathy", "diabetic macular edema", "diabetic kidney disease"],
        "trial_condition": "Diabetes Complications",
    },
    "Chronic lung diseases (asthma, chronic bronchitis, emphysema)": {
        "mesh_headings": ["Pulmonary Disease, Chronic Obstructive", "Asthma"],
        "label_terms": ["chronic obstructive pulmonary disease", "copd", "asthma", "emphysema", "chronic bronchitis"],
        "trial_condition": "Chronic Obstructive Pulmonary Disease",
    },
    "Diabetic peripheral neuropathy": {
        "mesh_headings": ["Diabetic Neuropathies"],
        "label_terms": ["diabetic peripheral neuropath", "diabetic neuropath"],
        "trial_condition": "Diabetic Peripheral Neuropathy",
    },
    "Peripheral arterial disease": {
        "mesh_headings": ["Peripheral Arterial Disease", "Intermittent Claudication"],
        "label_terms": ["peripheral artery disease", "peripheral arterial disease", "intermittent claudication"],
        "trial_condition": "Peripheral Arterial Disease",
    },
    "Diabetic foot ulcers": {
        "mesh_headings": ["Diabetic Foot"],
        "label_terms": ["diabetic foot ulcer", "diabetic neuropathic ulcer"],
        "trial_condition": "Diabetic Foot Ulcer",
    },
    "Asthma": {
        "mesh_headings": ["Asthma"],
        "label_terms": ["asthma"],
        "trial_condition": "Asthma",
    },
    "Emphysema": {
        "mesh_headings": ["Pulmonary Emphysema", "Pulmonary Disease, Chronic Obstructive"],
        "label_terms": ["emphysema", "chronic obstructive pulmonary disease", "copd"],
        "trial_condition": "Emphysema",
    },
    "Ischemic heart disease": {
        "mesh_headings": ["Myocardial Ischemia", "Coronary Artery Disease"],
        "label_terms": ["coronary artery disease", "coronary heart disease", "myocardial infarction", "angina"],
        "trial_condition": "Myocardial Ischemia",
    },
    "General Age-Related Decline": {
        "mesh_headings": [],
        "label_terms": [],
        "trial_condition": "Aging",
    },
}


DISEASE_TERMS.update({
    "Chronic bronchitis": {
        "mesh_headings": ["Bronchitis, Chronic", "Pulmonary Disease, Chronic Obstructive"],
        "label_terms": ["chronic bronchitis"],
        "trial_condition": "Chronic Bronchitis",
    },
    "Congestive cardiac failure": {
        "mesh_headings": ["Heart Failure"],
        "label_terms": ["heart failure"],
        "label_exclude_terms": ["pulmonary arterial"],
        "trial_condition": "Heart Failure",
    },
    "Acute myocardial infarction": {
        "mesh_headings": ["Myocardial Infarction"],
        "label_terms": ["myocardial infarction"],
        "trial_condition": "Myocardial Infarction",
    },
    "Diabetic retinopathy": {
        "mesh_headings": ["Diabetic Retinopathy"],
        "label_terms": ["diabetic retinopathy"],
        "trial_condition": "Diabetic Retinopathy",
    },
    "Hip fracture": {
        "mesh_headings": ["Hip Fractures", "Osteoporosis", "Osteoporosis, Postmenopausal"],
        "label_terms": ["hip fracture", "hip fractures", "fracture", "fractures"],
        "trial_condition": "Hip Fractures",
    },
    "Spine fracture": {
        "mesh_headings": ["Spinal Fractures", "Osteoporosis", "Osteoporosis, Postmenopausal"],
        "label_terms": ["vertebral fracture", "vertebral fractures", "fracture", "fractures"],
        "trial_condition": "Spinal Fractures",
    },
    "Wrist fracture": {
        "mesh_headings": ["Wrist Fractures", "Osteoporosis", "Osteoporosis, Postmenopausal"],
        "label_terms": ["fracture", "fractures"],
        "trial_condition": "Wrist Fractures",
    },
    "Insulin use": {
        "mesh_headings": ["Diabetes Mellitus, Type 2"],
        "label_terms": ["type 2 diabetes", "diabetes mellitus"],
        "label_exclude_terms": ["nephropathy", "neuropathy", "retinopathy", "macular"],
        "trial_condition": "Type 2 Diabetes",
    },
})
if "Diabetic peripheral neuropathy" in DISEASE_TERMS:
    DISEASE_TERMS["Diabetic peripheral neuropathy"]["label_terms"] = list(dict.fromkeys(
        DISEASE_TERMS["Diabetic peripheral neuropathy"].get("label_terms", []) +
        ["diabetic peripheral neuropathy", "diabetic peripheral neuropathic pain", "diabetic neuropathy"]))


def is_mapped(disease_name: str) -> bool:
    return _norm(disease_name) in _BY_NORM


def _norm(name: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in name.lower()).split())


_BY_NORM = {_norm(k): v for k, v in DISEASE_TERMS.items()}


def get_disease_terms(disease_name: str) -> dict:
    """The disease's API terms; a risk name not listed here is tried as its own MeSH heading and
    label term, which works for any name that is already standard medical terminology."""
    terms = _BY_NORM.get(_norm(disease_name)) or {
        "mesh_headings": [disease_name],
        "label_terms": [disease_name.lower()],
        "trial_condition": disease_name,
    }
    if "mesh_ids" not in terms:
        from app.services.rxclass_service import get_disease_mesh_ids
        terms = {**terms, "mesh_ids": [i for h in terms["mesh_headings"] for i in get_disease_mesh_ids(h)]}
    return terms
