"""POST /v1/embed — ingest a document's text into the user's vector index."""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_state, verify_internal_key
from app.schemas import DeleteResponse, EmbedRequest, EmbedResponse
from app.services import embeddings, vectorstore
from app.services.chunking import chunk_text

router = APIRouter(prefix="/v1", tags=["indexing"], dependencies=[Depends(verify_internal_key)])


@router.post("/embed", response_model=EmbedResponse)
async def embed_document(
    payload: EmbedRequest,
    state=Depends(get_state),
    settings: Settings = Depends(get_settings),
) -> EmbedResponse:
    chunks = chunk_text(payload.text, settings.chunk_size, settings.chunk_overlap)

    # CPU-bound -> threadpool (see services/embeddings.py)
    vectors = await embeddings.embed_texts(state.embedding_model, chunks)

    collection = vectorstore.get_user_collection(state.chroma_client, payload.user_id)
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
    state=Depends(get_state),
) -> DeleteResponse:
    collection = vectorstore.get_user_collection(state.chroma_client, user_id)
    vectorstore.delete_document(collection, document_id)
    return DeleteResponse(document_id=document_id, deleted=True)
