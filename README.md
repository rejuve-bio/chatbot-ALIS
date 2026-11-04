# chatbot-ALIS

A chat endpoint backend built with FastAPI.

## Requirements

- Python 3.11
- Docker

## Running it

```bash
docker compose up --build
```

## Clinician-feedback CLI

Run from the repository root with an authorised intervention JSON handoff.

```bash
python3 scripts/feedback --input 'path/to/interventions.json'
```

Sign in, choose an intervention number, read its claim and evidence, then select
Support, Oppose or Uncertain and an optional note. The result is saved and the
score updates through native PLN. No commands, citation entry or model calls are
needed for ordinary reviews. Existing reviews and history are preserved.

Uncertain responses remain visible without a binary PLN vote. The list separates
More support, Needs review, More opposition and unreviewed entries, with optional
coverage filters. Correct your assessment through the same review choices, or
record an attributed concern about the intervention entry. Paper text is preserved.
Study details and correction history remain on the same screen. Switch
reviewers from the list to see shared responses under another authenticated account.

[Short guide](docs/feedback-cli.md) · [Two-account demo](docs/feedback-claim-demo.md)

The reviewed-atlas workflow requires the existing pinned native Omega container
(`chatbot-alis-omega-mapping-1`, configurable with `OMEGA_FEEDBACK_CONTAINER`).
The optional Omega review brief makes one explicitly requested model call and
reuses its saved result; it does not change scores or reviews.

Existing ALIS signup/login is used. For isolated local test accounts add
`--auth local --db PATH`. Python 3.9+ and SWI-Prolog 9.3+ are required. The former
technical command interface is preserved behind `--advanced`.

## APIs

### `/chat`

Clinical chat endpoint. Answers general/definitional questions on its own, and for
patient-specific or population questions calls the ALIS backend for patient data plus
Qdrant (vector DB) for PC chunk data already prepared in the `data/` folder.

Example query:

```bash
curl -s -X POST "http://localhost:8082/chat" \
  -H "Authorization: $TOKEN" \
  -F "message=What are this patient's top disease risks?" \
  -F "patient_id={patient_id}"
```

### `/atlas/{patient_id}/risk`

Stores a list of resources as collections of disease evidence, and uses them to give
investigational-compound recommendations for a disease. Resources (fetched in CSV
format from their own websites):

- DrugAge
- GenAge (human + model organisms)
- CellAge
- ClinPGx
- Evipedia

Example query:

```bash
curl -s -G "http://localhost:8082/atlas/{patient_id}/risk" \
  --data-urlencode "disease=Cognitive impairment" \
  -H "Authorization: $TOKEN"
```

### Reviewed intervention atlas

The clinician CLI now closes the post-intervention feedback loop: saving a
judgment revises its appraisal through native PLN, then native Omega tools
assemble the next atlas using the current reviews, uncertainty and record
concerns. The screen shows before/after values, research order, status and the
responsible review IDs. **r** opens the saved atlas, change history, native trace
and JSON export. The ordinary workflow makes zero model calls.

See [the review workflow](docs/feedback-cli.md). This operates on the supplied
intervention JSON and preserves the original evidence and review history.
