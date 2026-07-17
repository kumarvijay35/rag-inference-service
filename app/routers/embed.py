"""POST /v1/embed — ingest a document's text into the user's vector index."""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import verify_internal_key
from app.schemas import DeleteResponse, EmbedRequest, EmbedResponse
from app.services import embeddings, vectorstore
from app.services.chunking import chunk_text

router = APIRouter(prefix="/v1", tags=["indexing"], dependencies=[Depends(verify_internal_key)])


@router.post("/embed", response_model=EmbedResponse)
async def embed_document(
    payload: EmbedRequest,
    settings: Settings = Depends(get_settings),
) -> EmbedResponse:
    chunks = chunk_text(payload.text, settings.chunk_size, settings.chunk_overlap)

    # Lazily-loaded singletons (lru_cache) — first call pays the load cost,
    # the startup warm thread usually pays it before any request arrives.
    model = embeddings.get_model(settings.embedding_model_name)
    client = vectorstore.get_client(settings.chroma_dir)

    # CPU-bound -> threadpool (see services/embeddings.py)
    vectors = await embeddings.embed_texts(model, chunks)

    collection = vectorstore.get_user_collection(client, payload.user_id)
    count = vectorstore.upsert_chunks(
        collection, payload.document_id, chunks, vectors, payload.metadata
    )

    return EmbedResponse(
        document_id=payload.document_id,
        chunks_indexed=count,
        embedding_model=settings.embedding_model_name,
    )


@router.delete("/documents/{user_id}/{document_id}", response_model=DeleteResponse)
async def delete_document(
    user_id: str,
    document_id: str,
    settings: Settings = Depends(get_settings),
) -> DeleteResponse:
    client = vectorstore.get_client(settings.chroma_dir)
    collection = vectorstore.get_user_collection(client, user_id)
    vectorstore.delete_document(collection, document_id)
    return DeleteResponse(document_id=document_id, deleted=True)