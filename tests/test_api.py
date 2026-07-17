"""Tests for the RAG inference service.

The embedding model and Groq are mocked so tests run fast and offline —
we're testing OUR logic (auth, validation, chunking, wiring), not
HuggingFace or Groq.

Run:  INTERNAL_API_KEY=test-key GROQ_API_KEY=fake pytest -v
"""

import os

os.environ.setdefault("INTERNAL_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "fake")
os.environ.setdefault("CHROMA_DIR", "/tmp/test_chroma")

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import embeddings
from app.services.chunking import chunk_text

HEADERS = {"X-Internal-Api-Key": "test-key"}


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        # deterministic fake vectors, correct shape for MiniLM (384 dims)
        return np.array([[hash(t) % 100 / 100.0] * 384 for t in texts])


@pytest.fixture()
def client(monkeypatch):
    async def fake_generate_answer(*args, **kwargs):
        return "This is a grounded test answer."

    # Routers (and main's warm thread) look up get_model on the embeddings
    # module at call time, so patching the module attribute replaces it
    # everywhere — and bypasses lru_cache, since the real function never runs.
    monkeypatch.setattr(embeddings, "get_model", lambda name: FakeEmbeddingModel())
    monkeypatch.setattr("app.routers.query.llm.generate_answer", fake_generate_answer)

    with TestClient(main.app) as c:
        yield c


# ---------- chunking (pure logic, no app needed) ----------

def test_chunk_text_respects_size():
    text = "para one. " * 50 + "\n\n" + "para two. " * 50
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # size + small tolerance for overlap seed


def test_chunk_text_single_short_paragraph():
    assert chunk_text("hello world") == ["hello world"]


# ---------- auth ----------

def test_missing_api_key_rejected(client):
    resp = client.post("/v1/query", json={"user_id": "u1", "question": "hi there"})
    assert resp.status_code == 401


# ---------- validation ----------

def test_embed_rejects_empty_text(client):
    resp = client.post(
        "/v1/embed",
        json={"user_id": "u1", "document_id": "d1", "text": ""},
        headers=HEADERS,
    )
    assert resp.status_code == 422


def test_query_rejects_top_k_over_limit(client):
    resp = client.post(
        "/v1/query",
        json={"user_id": "u1", "question": "valid question", "top_k": 50},
        headers=HEADERS,
    )
    assert resp.status_code == 422


# ---------- end-to-end (mocked model + LLM) ----------

def test_embed_then_query_flow(client):
    embed_resp = client.post(
        "/v1/embed",
        json={
            "user_id": "u1",
            "document_id": "doc-1",
            "text": "Redis is an in-memory data store.\n\nIt is often used for caching.",
            "metadata": {"filename": "redis.txt"},
        },
        headers=HEADERS,
    )
    assert embed_resp.status_code == 200
    assert embed_resp.json()["chunks_indexed"] >= 1

    query_resp = client.post(
        "/v1/query",
        json={"user_id": "u1", "question": "What is Redis used for?", "document_ids": ["doc-1"]},
        headers=HEADERS,
    )
    assert query_resp.status_code == 200
    body = query_resp.json()
    assert body["answer"] == "This is a grounded test answer."
    assert body["sources"][0]["document_id"] == "doc-1"


def test_user_isolation(client):
    # user u2 must not see u1's documents
    resp = client.post(
        "/v1/query",
        json={"user_id": "u2-empty", "question": "What is Redis used for?"},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["sources"] == []