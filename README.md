# chatbot-ALIS

A chat endpoint backend built with FastAPI.

## Requirements

- Python 3.11
- Docker

## Running it

```bash
docker compose up --build
```

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
