"""Application configuration.

All settings come from environment variables (or a .env file locally).
Using pydantic-settings gives us type-validated config at startup —
the service fails fast with a clear error if GROQ_API_KEY is missing,
instead of failing on the first request.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- LLM ---
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- Embeddings ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store ---
    # NOTE: on Render, point this at a persistent disk mount (e.g. /var/data/chroma)
    # otherwise the index is wiped on every deploy/restart.
    chroma_dir: str = "./chroma_data"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    default_top_k: int = 3

    # --- Service-to-service auth ---
    # Shared secret between Django and this service. Django sends it in the
    # X-Internal-Api-Key header. This service is NOT meant to be public.
    internal_api_key: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
