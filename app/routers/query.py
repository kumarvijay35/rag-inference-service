"""POST /v1/query — retrieve relevant chunks and generate a grounded answer."""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_state, verify_internal_key
from app.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services import embeddings, llm, vectorstore

router = APIRouter(prefix="/v1", tags=["inference"], dependencies=[Depends(verify_internal_key)])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    state=Depends(get_state),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    # Lazily-loaded singletons (lru_cache); Groq client still lives in app.state
    model = embeddings.get_model(settings.embedding_model_name)
    client = vectorstore.get_client(settings.chroma_dir)

    # 1. Embed the question (CPU-bound -> threadpool)
    [query_vector] = await embeddings.embed_texts(model, [payload.question])

    # 2. Retrieve top-k chunks from the user's isolated collection
    collection = vectorstore.get_user_collection(client, payload.user_id)
    hits = vectorstore.query_chunks(
        collection, query_vector, payload.top_k, payload.document_ids
    )

    # 3. Generate a grounded answer (I/O-bound -> native await, event loop stays free)
    answer = await llm.generate_answer(
        state.groq_client,
        settings.groq_model,
        payload.question,
        hits,
        settings.llm_temperature,
        settings.llm_max_tokens,
    )

    return QueryResponse(
        answer=answer,
        sources=[SourceChunk(**h) for h in hits],
        model=settings.groq_model,
    )