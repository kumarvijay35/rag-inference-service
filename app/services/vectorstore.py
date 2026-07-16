"""ChromaDB vector store operations.

Per-user isolation: each user gets their own Chroma collection
(collection name = "user_<user_id>"). A query can never read another
user's chunks because it only ever touches that user's collection —
the same multi-tenant isolation model as the Django RAG app.
"""

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


def create_client(persist_dir: str) -> ClientAPI:
    return chromadb.PersistentClient(path=persist_dir)


def get_user_collection(client: ClientAPI, user_id: str) -> Collection:
    return client.get_or_create_collection(
        name=f"user_{user_id}",
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    collection: Collection,
    document_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    extra_metadata: dict[str, str] | None = None,
) -> int:
    base_meta = {"document_id": document_id, **(extra_metadata or {})}
    collection.upsert(
        ids=[f"{document_id}:{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{**base_meta, "chunk_index": i} for i in range(len(chunks))],
    )
    return len(chunks)


def query_chunks(
    collection: Collection,
    query_embedding: list[float],
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    # $in filter -> one globally-ranked top-k across all selected documents,
    # instead of merging per-document result lists (which can't rank across docs).
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    hits: list[dict] = []
    if result["documents"] and result["documents"][0]:
        for text, meta, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {
                    "text": text,
                    "document_id": meta.get("document_id", ""),
                    # Chroma cosine 'distance' = 1 - similarity
                    "score": round(1.0 - distance, 4),
                }
            )
    return hits


def delete_document(collection: Collection, document_id: str) -> None:
    collection.delete(where={"document_id": document_id})
