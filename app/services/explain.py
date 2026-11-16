"""
Plain-language explanations built only from the sources: what each drug's FDA label says it treats
for this disease, and what a checked paper found.

The LLM rewrites the label's own indication text in plain words; it may not add a use the label
doesn't state, and it says when the label only covers a special form of the disease (Pneumocystis
pneumonia in HIV, pulmonary hypertension) or a different condition. Results are stored with the
paper checks, so each drug and disease is explained once.
"""

import json
import logging

from app.services import atlas_store
from app.services.llm_service import call_llm
from app.services.paper_check import PaperCheckUnavailable, _parse

logger = logging.getLogger(__name__)

LABEL_USE_PROMPT = """
You read the INDICATIONS section of a US FDA drug label and say, in plain words, what the label
says the drug is used for that relates to one disease. Use ONLY what the label text says; never
add a use, a dose or a patient group it doesn't state. Answer in JSON only:
{"scope": "general"|"subtype"|"none", "use": "<at most 30 plain words>"}

- scope "general": the label covers the disease, a main form of it, or preventing it, in adults.
  For a broad disease, a main form counts (for arthritis: rheumatoid arthritis or osteoarthritis;
  for pneumonia: community-acquired or hospital pneumonia caused by susceptible bacteria). A use
  that reduces or prevents the disease counts too (for hip fracture: osteoporosis treatment the
  label says reduces hip or other fractures).
- scope "subtype": the label covers only a rare form or a special population (for pneumonia:
  Pneumocystis pneumonia in people with HIV; for hypertension: pulmonary hypertension; children only).
- scope "none": the label's use is a different condition (for pneumonia: a COPD flare-up or
  aspiration pneumonitis).
- use: which form of the disease, which patients, and (for infections) which organisms, as the
  label states them. Write it as a phrase, e.g. "community-acquired pneumonia caused by
  susceptible bacteria such as Streptococcus pneumoniae, in adults who can take tablets".
"""


def label_use(drug: str, label: dict | None, disease: str) -> dict | None:
    """{"scope", "use"} for this drug's FDA label and this disease, or None without a label."""
    if not label:
        return None
    key = f"label:{drug.lower()}"
    stored = atlas_store.get_paper_record(key, drug, f"x2|{disease}")
    if stored is not None:
        return json.loads(stored[1])
    text = " ".join(label.get("all_indications") or [label.get("indications_and_usage") or ""])[:6000]
    try:
        answer = _parse(call_llm(f"Disease: {disease}\nDrug: {drug}\n\nINDICATIONS text:\n{text}",
                                 system_prompt=LABEL_USE_PROMPT))
    except Exception as e:
        logger.warning(f"explain | label use of {drug} / {disease} couldn't be read: {e}")
        raise PaperCheckUnavailable(key) from e
    result = {"scope": answer.get("scope") if answer.get("scope") in ("general", "subtype", "none") else "none",
              "use": (answer.get("use") or "").strip().rstrip(".")}
    atlas_store.put_paper_verdict(key, drug, f"x2|{disease}", result["scope"] == "general", json.dumps(result))
    logger.info(f"explain | {drug} / {disease}: {result}")
    return result


def paper_finding(paper: dict, item: str, disease: str, sex: str | None) -> str | None:
    """What the paper check found the study showed (one sentence), or None."""
    stored = atlas_store.get_paper_record(paper.get("pmid") or "", item, f"c3|{disease}|{sex or '-'}")
    if not stored or not stored[0] or not stored[1]:
        return None
    return stored[1].strip().rstrip(".")
