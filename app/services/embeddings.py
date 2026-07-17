"""Embedding service.

KEY DESIGN POINT (interview gold):
Embedding with sentence-transformers is CPU-bound. If we called
model.encode() directly inside an async endpoint, it would BLOCK the event
loop and stall every concurrent request. So we offload it to the threadpool
with fastapi.concurrency.run_in_threadpool, keeping the event loop free.

- I/O-bound work (Groq API call)  -> native async/await
- CPU-bound work (embeddings)     -> threadpool offload

DEPLOYMENT NOTE: the model loads lazily via lru_cache (warmed by a
background thread at startup — see main.py). Loading it in lifespan
blocked port binding on Render's CPU tier; lazy loading lets uvicorn
bind immediately while keeping single-instance semantics.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi.concurrency import run_in_threadpool

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_model(model_name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer  # heavy import, runtime-deferred
    return SentenceTransformer(model_name)


async def embed_texts(model: "SentenceTransformer", texts: list[str]) -> list[list[float]]:
    def _encode() -> list[list[float]]:
        # normalize_embeddings=True -> vectors are unit length, so cosine
        # similarity == dot product (cheaper, and what Chroma expects with
        # the cosine space).
        return model.encode(texts, normalize_embeddings=True).tolist()

    return await run_in_threadpool(_encode)