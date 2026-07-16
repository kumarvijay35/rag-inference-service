"""Pydantic v2 request/response models.

These give the service a typed, self-documenting contract:
- invalid payloads are rejected with a 422 before any code runs
- FastAPI auto-generates the OpenAPI schema from these models
"""

from pydantic import BaseModel, Field


# ---------- /v1/embed ----------

class EmbedRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Owner of the document (from Django auth)")
    document_id: str = Field(..., min_length=1, description="Unique id of the document in Django")
    text: str = Field(..., min_length=1, description="Extracted plain text of the document")
    metadata: dict[str, str] | None = Field(
        default=None, description="Optional metadata stored with every chunk (e.g. filename)"
    )


class EmbedResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    embedding_model: str


# ---------- /v1/query ----------

class QueryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=3, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None,
        description=(
            "Restrict retrieval to these documents (e.g. all docs in a chat session); "
            "None searches all of the user's documents"
        ),
    )
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    text: str
    document_id: str
    score: float = Field(..., description="Cosine similarity (higher = more relevant)")


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str


# ---------- misc ----------

class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
