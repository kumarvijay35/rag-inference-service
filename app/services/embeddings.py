"""Embedding service.

KEY DESIGN POINT (interview gold):
Embedding with sentence-transformers is CPU-bound. If we called
model.encode() directly inside an async endpoint, it would BLOCK the event
loop and stall every concurrent request. So we offload it to the threadpool
with fastapi.concurrency.run_in_threadpool, keeping the event loop free.

- I/O-bound work (Groq API call)  -> native async/await
- CPU-bound work (embeddings)     -> threadpool offload

The model is loaded ONCE at startup (see main.py lifespan) — loading it per
request would add seconds of latency and blow up memory.
"""

from fastapi.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer


def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


async def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    def _encode() -> list[list[float]]:
        # normalize_embeddings=True -> vectors are unit length, so cosine
        # similarity == dot product (cheaper, and what Chroma expects with
        # the cosine space).
        return model.encode(texts, normalize_embeddings=True).tolist()

    return await run_in_threadpool(_encode)
