"""RAG Inference Service — FastAPI microservice for the RAG Document Chatbot.

Django owns: auth (JWT), file upload, PDF/TXT text extraction, user/document CRUD.
This service owns: chunking, embeddings, vector search, LLM generation.

Startup (lifespan):
- Load the sentence-transformers model ONCE (expensive: ~seconds + ~100MB RAM)
- Open one persistent ChromaDB client
- Create one AsyncGroq client (reused connection pool)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from groq import AsyncGroq

from app.config import get_settings
from app.routers import embed, query
from app.schemas import HealthResponse
from app.services import embeddings, vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.embedding_model = embeddings.load_model(settings.embedding_model_name)
    app.state.chroma_client = vectorstore.create_client(settings.chroma_dir)
    app.state.groq_client = AsyncGroq(api_key=settings.groq_api_key)
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
