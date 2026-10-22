"""Application settings, loaded from environment variables or a `.env` file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Chunking: ~800 characters with 150 characters of overlap keeps a full
    # paragraph per chunk while making sure sentences cut at a boundary still
    # appear whole in the neighbouring chunk.
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4
    # Cosine distance (0 = identical, 2 = opposite). If even the closest chunk
    # is further than this, the question is treated as out of scope and the
    # LLM is never called.
    max_distance: float = 0.75

    chroma_path: str = "./chroma_data"
    collection_name: str = "docurag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
