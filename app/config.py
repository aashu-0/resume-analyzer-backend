from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Resume <-> JD Analyzer.
    All values are loaded from environment variables / a .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Provider keys ---
    cohere_api_key: str = ""
    google_api_key: str = ""

    # --- Vector store ---
    vector_store: str = "pinecone"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "tensor-vector"

    # --- Models ---
    cohere_embedding_model: str = "embed-english-v3.0"
    gemini_model: str = "gemini-2.5-flash-lite"

    # --- RAG tuning ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_top_k: int = 4

    # --- App ---
    app_env: str = "development"
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # Pinecone embedding dimension for embed-english-v3.0 (Cohere)
    @property
    def embedding_dimension(self) -> int:
        # embed-english-v3.0 -> 1024 dims
        return 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
