import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

_codebook = {}        
_reverse = {}         
_force_included = set() # codes where ForceInc == 1
_questionnaire = set()  # codes where type == Q


def load_codebook(csv_path: str = None):
    if not csv_path:
        csv_path = os.path.join(
            os.path.dirname(__file__), "../../data/codebook_linAge2.csv"
        )

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        code = str(row.get("Var", "")).strip()
        human = str(row.get("Human", "")).strip()
        force = row.get("ForceInc", 0)
        vtype = str(row.get("Demo/Exam/Quest/Lab/Mort", "")).strip().upper()

        if not code or not human or code == "nan" or human == "nan":
            continue

        _codebook[code] = human

        if str(force) == "1":
            _force_included.add(code)

        if vtype == "Q":
            _questionnaire.add(code)

        # build reverse lookup from human label
        key = human.lower().strip()
        if key not in _reverse:
            _reverse[key] = []
        if code not in _reverse[key]:
            _reverse[key].append(code)

    logger.info(
        f"Codebook loaded: {len(_codebook)} variables, "
        f"{len(_force_included)} force-included, "
        f"{len(_questionnaire)} questionnaire"
    )


def get_label(code: str) -> str:
    if not _codebook:
        load_codebook()
    return _codebook.get(code, code)



def is_questionnaire(code: str) -> bool:
    if not _codebook:
        load_codebook()
    return code in _questionnaire


# Questionnaire value encoding (backend standardized):
# 1=Yes, 2=No, null=not answered/not registered (filtered out before reaching here)
_NHANES_Q_VALUES = {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"}

# Per-code scale overrides for non-Yes/No questionnaire items
_NHANES_SCALE_VALUES: dict[str, dict[int, str]] = {
    "HUQ010": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor",
               7: "Refused", 9: "Don't know"},
    "HUQ020": {1: "Much better", 2: "Somewhat better", 3: "About the same",
               4: "Somewhat worse", 5: "Much worse", 7: "Refused", 9: "Don't know"},
    "HUQ050": {1: "None", 2: "1 time", 3: "2-3 times", 4: "4-9 times",
               5: "10 or more times", 7: "Refused", 9: "Don't know"},
}

# Demographic codes that should never appear as biomarkers in context
DEMOGRAPHIC_CODES = {"SEQN", "RIAGENDR", "RIDAGEEX", "RIDRETH1", "DMDEDUC2", "INDFMPIR"}


def decode_questionnaire_value(code: str, value) -> str:
    if not is_questionnaire(code):
        return str(value)
    try:
        int_val = int(float(value))
        if code in _NHANES_SCALE_VALUES:
            return _NHANES_SCALE_VALUES[code].get(int_val, str(value))
        return _NHANES_Q_VALUES.get(int_val, str(value))
    except (ValueError, TypeError):
        return str(value)


def get_force_included_variables() -> dict[str, str]:
    """Return all ForceInc=1 variables as {code: human_label}"""
    if not _codebook:
        load_codebook()
    return {code: _codebook[code] for code in _force_included if code in _codebook}


def is_longitudinal_question(question: str) -> bool:
    """
    Quick heuristic to detect if a question is asking about
    change over time before sending to the LLM.
    """
    triggers = [
        "over time", "over the past", "over the last",
        "trend", "trending", "history",
        "last month", "last week", "last year", "last visit",
        "past month", "past week", "past year",
        "after", "before", "since",
        "across visits", "between visits", "previous visit",
        "has changed", "have changed", "has increased", "has decreased",
        "has worsened", "has improved", "has it changed",
        "how has", "how have", "how did it",
        "longitudinal", "time series", "time-series",
        "since last", "compared to last", "compared to previous",
    ]
    q = question.lower()
    return any(t in q for t in triggers)