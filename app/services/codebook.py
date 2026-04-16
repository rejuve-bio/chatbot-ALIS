import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

_codebook = {}          # code -> human label
_reverse = {}           # normalized human term -> [code, code, ...]
_force_included = set() # codes where ForceInc == 1


def load_codebook(csv_path: str = None):
    global _codebook, _reverse, _force_included
    if not csv_path:
        csv_path = os.path.join(
            os.path.dirname(__file__), "../../data/codebook_linAge2.csv"
        )

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        code = str(row.get("Var", "")).strip()
        human = str(row.get("Human", "")).strip()
        force = row.get("ForceInc", 0)

        if not code or not human or code == "nan" or human == "nan":
            continue

        _codebook[code] = human

        if str(force) == "1":
            _force_included.add(code)

        # build reverse lookup from human label
        key = human.lower().strip()
        if key not in _reverse:
            _reverse[key] = []
        if code not in _reverse[key]:
            _reverse[key].append(code)

    logger.info(
        f"Codebook loaded: {len(_codebook)} variables, "
        f"{len(_force_included)} force-included"
    )


def get_label(code: str) -> str:
    if not _codebook:
        load_codebook()
    return _codebook.get(code, code)



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
        "change", "changed", "trend", "over time", "history",
        "last month", "last week", "last year", "past month",
        "past week", "past year", "after", "before", "since",
        "progress", "progression", "improve", "improved",
        "worsen", "worsened", "increase", "decrease", "fluctuat",
        "how has", "how did", "how have", "longitudinal",
        "time series", "over the", "across visits", "between visits"
    ]
    q = question.lower()
    return any(t in q for t in triggers)