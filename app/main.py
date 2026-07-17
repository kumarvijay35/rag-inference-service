"""RAG Inference Service — FastAPI microservice for the RAG Document Chatbot.

Django owns: auth (JWT), file upload, PDF/TXT text extraction, user/document CRUD.
This service owns: chunking, embeddings, vector search, LLM generation.

Startup design:
- The embedding model and ChromaDB client load LAZILY (lru_cache in the
  services layer), warmed by a background thread at startup. Loading them
  in lifespan blocked port binding on Render's CPU tier — lazy loading
  lets uvicorn bind the port immediately.
- The AsyncGroq client is cheap to create, so it lives in app.state.
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from groq import AsyncGroq

from app.config import get_settings
from app.routers import embed, query
from app.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.groq_client = AsyncGroq(api_key=settings.groq_api_key)

    # Warm the embedding model in the background — doesn't block port binding
    def warm():
        from app.services.embeddings import get_model
        get_model(settings.embedding_model_name)

    threading.Thread(target=warm, daemon=True).start()
    yield
    await app.state.groq_client.close()


app = FastAPI(
    title="RAG Inference Service",
    description="Async embedding + retrieval + generation microservice, called internally by the Django RAG app.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(embed.router)
app.include_router(query.router)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        embedding_model=settings.embedding_model_name,
        llm_model=settings.groq_model,
    )