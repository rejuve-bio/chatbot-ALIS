"""
General-question short-circuit for the chat pipeline — classify, then
explain, before the full patient/data pipeline ever runs.
"""

import logging

from app.services.llm_service import call_llm

logger = logging.getLogger(__name__)

GENERAL = "GENERAL"
NEEDS_DATA = "NEEDS_DATA"


def _build_classifier_prompt(has_patient: bool) -> str:
    patient_context = (
        "A patient IS currently selected in this conversation."
        if has_patient else
        "NO patient is currently selected in this conversation."
    )
    return f"""
You're the first checkpoint in a clinical chatbot's backend, deciding one
thing before anything else happens: can you answer this message right
now, yourself, or does answering it correctly require real work from the
backend behind you?

Here's concretely what that backend can actually go do, so you know what
you're weighing against, not just an abstract "it's expensive":
- Fetch a specific patient's real record — their biomarkers, PC
  contributions, disease risks, life events — from the clinical API.
  Only relevant if a real, specific patient is who this is actually about.
- Look up a NAMED PC group's own real clinical data — its associated
  diseases, causes of death, interventions — from the PC knowledge
  database. Relevant when the question is genuinely about what a
  specific PC code (not PCs in general) is associated with.
- Search real investigational/preclinical evidence collections (DrugAge,
  ClinPGx, GenAge, CellAge, Evipedia, live ClinicalTrials.gov data) for
  compounds, genes, or trials. Relevant when the question is actually
  about real interventions or evidence for a condition or mechanism.
- Fetch a patient's history for a specific named variable over time (a
  trend), or the whole clinic's patient list for a genuine population
  question ("how are my patients doing overall") — both real, and both
  expensive, the second one especially so (refreshes every patient one
  by one).

Your job is to recognize whether THIS message genuinely needs one of
those, not to protect against cost in the abstract.

{patient_context}

Ask yourself plainly: does answering this message require any real,
specific data — a patient's actual values, a lookup in a medical
database, something that could be wrong if you just guessed — or could
any reasonably knowledgeable assistant answer it right now, correctly,
without looking anything up? That second case covers far more than
definitions of terms like "PC" or "biomarker" — it covers greetings,
thanks, small talk, questions about what you can do, or anything else
where guessing costs nothing because there's nothing real to get wrong.

The one thing to stay careful about: this platform has real, specific
records behind certain kinds of names and numbers — not just patients,
but things like a specific PC group (PC1, PC24M, PC7F, ...), a specific
gene, drug, or trial. Each of those specific identifiers has its own
real stored data (associated diseases, mechanisms, evidence) — asking
what THAT identifier is actually associated with (its diseases, its
interventions, its evidence) always means real work is needed, even if
the question is phrased simply, because getting those specifics wrong
would be a real, damaging guess.

But naming a specific identifier doesn't automatically mean the
question is asking for its stored data. If the question is really about
the general naming convention or notation itself — what a suffix like
the M or F on a PC code signifies, why PC codes come in gender variants
at all, how PC numbering works — the specific code is just there as an
example of the pattern being asked about, not the actual subject of a
data lookup. That's still a general knowledge question, answerable
yourself, the same as "what is a PC" would be. Ask which one the
question is really after: THIS identifier's own real associated data,
or the general pattern/convention it happens to illustrate.

When you're genuinely unsure which way something leans, choose the safe
side and say it needs real data — a wrong guess that fabricates
specifics instead of looking them up is a much worse mistake than
occasionally doing more work than strictly necessary.

Respond with exactly one word — {NEEDS_DATA} if this needs the real
backend, {GENERAL} if you can just answer it yourself right now.
"""

EXPLAIN_SYSTEM_PROMPT = """
You're a clinical chatbot's assistant, replying to a message that has
already been checked and confirmed to need no real backend work — no
specific patient's data, no database lookup.

Here's what this platform actually IS, at a general level — not a
glossary of every term, just the real shape of the thing, so you can
reason about anything within it instead of needing a pre-written
definition for each possible word:

It estimates a person's biological age from their biomarkers (lab
results, vital signs, questionnaire answers) and compares it to their
chronological age. Biomarkers that move together get grouped into
Principal Components (PCs), each one a shared pattern connecting those
biomarkers to specific disease risks; PCs are computed separately per
gender (a trailing M or F on a PC code, e.g. PC1M vs PC1F, is that
gender split — same underlying pattern, not a different concept),
because which biomarkers move together and what that means for risk can
differ by gender. Beyond a patient's own data, the platform also draws
on named external evidence sources (DrugAge, ClinPGx, GenAge, CellAge,
Evipedia) for investigational compounds, genes, and trials, each tagged
by how strong its evidence is (animal-model vs. human).

Reason from that general shape, don't just match it to a checklist — a
term or phrasing you haven't seen written out above should still be
answerable if it clearly fits within what's described (e.g. a question
about how two related codes or concepts differ, or what a naming pattern
means), the way someone who actually understood this platform would
work it out rather than only knowing pre-memorized definitions.

Answer directly, confidently, and naturally: real sentences, not a list,
no headers or markdown, more than one sentence so it doesn't feel
clipped. This message was already checked before it reached you — it
doesn't need a real lookup, so just answer it.
"""


def classify_question(question: str, has_patient: bool = False) -> str:
    try:
        reply = call_llm(question, system_prompt=_build_classifier_prompt(has_patient))
    except Exception as e:
        logger.warning(f"glossary | classifier call failed, defaulting to NEEDS_DATA: {e}")
        return NEEDS_DATA
    reply = (reply or "").strip().upper()
    if GENERAL in reply and NEEDS_DATA not in reply:
        return GENERAL
    return NEEDS_DATA


def explain_with_llm(question: str) -> str | None:
    try:
        answer = call_llm(question, system_prompt=EXPLAIN_SYSTEM_PROMPT)
    except Exception as e:
        logger.warning(f"glossary | explanation call failed, falling back to full pipeline: {e}")
        return None
    return answer.strip() if answer else None


def match_glossary_question(question: str, has_patient: bool = False) -> str | None:
    classification = classify_question(question, has_patient=has_patient)
    logger.info(f"glossary | classified {question!r} as {classification}")
    if classification != GENERAL:
        return None

    answer = explain_with_llm(question)
    if answer:
        logger.info(f"glossary | answered generally, skipping full pipeline: {question!r}")
    return answer


POPULATION = "POPULATION"
SPECIFIC = "SPECIFIC"


def needs_population_data(question: str) -> bool:
    prompt = f"""
A clinical chatbot is about to decide whether to refresh EVERY patient in
the clinic from the backend one by one — a slow, expensive operation —
to answer this message. That's only actually useful if the message is
genuinely asking about patients as a group or population (e.g. "how are
my patients doing overall," "which patients are highest risk").

If the message is asking about something else entirely — a general
lookup (like a specific PC group's meaning), a definition, or anything
that doesn't actually require looking across the whole patient list —
that expensive refresh would be wasted work.

Message: {question!r}

Respond with exactly one word: {POPULATION} if this genuinely needs
data about patients as a group, {SPECIFIC} if it doesn't.
"""
    try:
        reply = call_llm(question, system_prompt=prompt)
    except Exception as e:
        logger.warning(f"glossary | population-need check failed, defaulting to POPULATION (safe default): {e}")
        return True
    reply = (reply or "").strip().upper()
    result = SPECIFIC not in reply
    logger.info(f"glossary | needs_population_data({question!r}) = {result}")
    return result


RELEVANT = "RELEVANT"
NOT_RELEVANT = "NOT_RELEVANT"


def needs_biology_evidence(question: str) -> bool:
    prompt = f"""
A clinical chatbot is about to search its investigational, preclinical
evidence collections — DrugAge, GenAge, CellAge, ClinPGx, and Evipedia.
These cover experimental compounds, genes, and interventions studied for
aging/longevity, and clinical trial evidence — NOT a patient's own
biomarker values, disease risks, or PC scores, which live elsewhere and
are answered without this search.

That search costs real time, and stuffs a whole extra section of compound
and gene data into the final answer's context even when none of it is
relevant — worth doing only when the question is actually about a
compound, drug, gene, intervention, or investigational/preclinical
evidence for a mechanism or condition.

Message: {question!r}

Respond with exactly one word: {RELEVANT} if this question is actually
asking about compounds, genes, drugs, interventions, or investigational
evidence, {NOT_RELEVANT} if it isn't.
"""
    try:
        reply = call_llm(question, system_prompt=prompt)
    except Exception as e:
        logger.warning(f"glossary | biology-evidence relevance check failed, defaulting to RELEVANT (safe default): {e}")
        return True
    reply = (reply or "").strip().upper()
    result = NOT_RELEVANT not in reply
    logger.info(f"glossary | needs_biology_evidence({question!r}) = {result}")
    return result


def resolve_disease_for_evidence(
    question: str, prior_question: str | None = None, known_diseases: list[str] | None = None
) -> str | None:
    """Which disease/condition an investigational-evidence question is actually about — resolves
    a pronoun reference ("this disease") against the prior message, never invents one."""
    known = f"\nThis patient's known disease risks: {', '.join(known_diseases)}." if known_diseases else ""
    prompt = f"""
A clinician asked a question about investigational compounds, genes, or evidence.
Determine which specific disease or condition it's actually about.

Current message: {question!r}
Previous message in this conversation: {prior_question!r}
{known}

RULES:
- If the current message names a disease/condition directly, return that.
- If it refers back with a pronoun ("this disease", "it", "that condition"), resolve it
  using the previous message only — never guess or assume one from the known-diseases list
  unless the question or previous message actually points to it.
- If no specific disease can be identified this way, reply with exactly: NONE
- Reply with ONLY the disease name, or NONE. No explanation, no punctuation.
"""
    try:
        reply = call_llm(question, system_prompt=prompt)
    except Exception as e:
        logger.warning(f"glossary | disease resolution failed, treating as NONE: {e}")
        return None
    reply = (reply or "").strip()
    if not reply or reply.upper() == "NONE":
        return None
    logger.info(f"glossary | resolve_disease_for_evidence({question!r}, prior={prior_question!r}) = {reply!r}")
    return reply
