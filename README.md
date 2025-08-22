# Real Estate RAG API

A multi-tenant Retrieval Augmented Generation API for real estate teams. Each
client can ingest their own documents (policies, listings, HOA/MLS rules) and
ask grounded questions that return concise answers with citations.

## Features
- Ingestion via JSON or file upload (pdf/txt/md) with automatic chunking.
- Per-tenant TF‑IDF indexing on disk under `data/processed/{client_id}`.
- Deterministic dummy LLM for local development and tests (no API key required).
- Optional local [Transformers](https://huggingface.co/docs/transformers/index)
  model when `RAG_MODEL_NAME` is set (e.g., `google/flan-t5-small`).
- REST endpoints:
  - `GET /healthz`
  - `POST /v1/ingest-json`
  - `POST /v1/ingest-files`
  - `POST /v1/ask`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py  # starts the API on :8000
```

To use a local Transformers model:

```bash
pip install transformers torch  # run once
RAG_MODEL_NAME=google/flan-t5-small python main.py
```

## Running Tests
```bash
pytest
```
