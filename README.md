# RAG Inference Service

Async FastAPI microservice that handles the embedding, retrieval, and LLM-generation path for the [RAG Document Chatbot](https://github.com/kumarvijay35/rag-document-chatbot). Called internally by the Django application over HTTP.

## Architecture

```
                          ┌──────────────────────────────┐
  Browser ── JWT ──────▶  │  Django + DRF (existing app) │
                          │  auth · upload · extraction  │
                          │  user/document CRUD          │
                          └──────────────┬───────────────┘
                                         │  httpx + X-Internal-Api-Key
                                         ▼
                          ┌──────────────────────────────┐
                          │  FastAPI Inference Service   │
                          │                              │
                          │  POST /v1/embed              │
                          │   text → chunk → embed(*) →  │
                          │   ChromaDB upsert            │
                          │                              │
                          │  POST /v1/query              │
                          │   embed(*) → top-k retrieve  │
                          │   → Groq LLaMA 3.3 (async)   │
                          └──────────────────────────────┘
                          (*) sentence-transformers, CPU-bound,
                              offloaded to threadpool
```

**Division of responsibility**

| Concern | Owner |
|---|---|
| User auth (JWT), ownership checks | Django |
| File upload + PDF/TXT text extraction | Django |
| Chunking, embeddings, vector search | FastAPI service |
| LLM generation (Groq) | FastAPI service |
| Service-to-service auth | Shared secret header |

## Why FastAPI here (and not more Django)?

The inference path is dominated by an **I/O-bound wait**: the request spends most of its life waiting on the Groq API. With async endpoints, one worker can hold many in-flight requests concurrently while they wait — a sync Django worker would be pinned for the whole duration. Embedding is the opposite: **CPU-bound**, so it's explicitly offloaded to the threadpool (`run_in_threadpool`) instead of blocking the event loop. Handling both correctly in one service is the point of this repo.

Django stays where it's strongest: auth, ORM-backed CRUD, admin, file handling.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in GROQ_API_KEY and INTERNAL_API_KEY

uvicorn app.main:app --reload --port 8001
```

Interactive docs (auto-generated from Pydantic models): http://localhost:8001/docs

### Try it

```bash
curl -X POST http://localhost:8001/v1/embed \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Key: $INTERNAL_API_KEY" \
  -d '{"user_id":"1","document_id":"42","text":"Redis is an in-memory data store.\n\nIt is commonly used for caching."}'

curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Key: $INTERNAL_API_KEY" \
  -d '{"user_id":"1","question":"What is Redis used for?","document_id":"42"}'
```

## Tests

```bash
INTERNAL_API_KEY=test-key GROQ_API_KEY=fake pytest -v
```

Embedding model and Groq are mocked — tests cover auth, Pydantic validation, chunking logic, the embed→query flow, and per-user isolation.

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GROQ_API_KEY` | yes | — | |
| `INTERNAL_API_KEY` | yes | — | Shared secret with Django |
| `GROQ_MODEL` | no | `llama-3.3-70b-versatile` | |
| `EMBEDDING_MODEL_NAME` | no | `sentence-transformers/all-MiniLM-L6-v2` | |
| `CHROMA_DIR` | no | `./chroma_data` | **On Render: set to a persistent-disk mount** (e.g. `/var/data/chroma`), otherwise the index is wiped on every deploy |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | no | `1000` / `150` | characters |

## Deploying on Render

1. New **Web Service** from this repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add a **persistent disk** and set `CHROMA_DIR` to its mount path.
5. Set env vars above. In the Django service, set `INFERENCE_SERVICE_URL` and the same `INFERENCE_INTERNAL_API_KEY`.

Free-tier note: the instance has 512MB RAM; MiniLM (~90MB) fits, but keep a single uvicorn worker.

## Django integration

See [`django_integration/inference_client.py`](django_integration/inference_client.py) — a thin `httpx` client with explicit timeouts and a 503 fallback pattern for the DRF views.

## Design decisions (short version)

- **Model loaded once at startup** (FastAPI lifespan), not per request.
- **CPU-bound embeddings → threadpool; I/O-bound LLM call → native await.**
- **Per-user Chroma collections** (`user_<id>`) — a query physically cannot read another user's chunks.
- **No LangChain in this service** — sentence-transformers + chromadb + groq directly. Fewer layers, smaller image, and the retrieval logic is explicit.
- **Pydantic v2 contracts** — invalid input dies with a 422 before touching the pipeline; OpenAPI docs come for free.
- **Not publicly exposed** — shared-secret header; Django remains the single public entry point and keeps all ownership checks.
