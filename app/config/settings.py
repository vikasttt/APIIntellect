"""Application configuration using Pydantic Settings."""
from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
class AppSettings(BaseSettings):
    """Core application settings."""
    env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    debug: bool = Field(default=False, alias="APP_DEBUG")
    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8000, alias="APP_PORT")
class DatabaseSettings(BaseSettings):
    """PostgreSQL database settings."""
    url: str = Field(alias="DATABASE_URL")
    pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, alias="DATABASE_POOL_TIMEOUT")
    echo: bool = Field(default=False, alias="DATABASE_ECHO")
class OpenAISettings(BaseSettings):
    """OpenAI API settings."""
    api_key: str = Field(alias="OPENAI_API_KEY")
    embedding_model: str = Field(
        default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(
        default=3072, alias="OPENAI_EMBEDDING_DIMENSIONS"
    )
    chat_model: str = Field(default="gpt-4o", alias="OPENAI_CHAT_MODEL")
    chat_temperature: float = Field(default=0.0, alias="OPENAI_CHAT_TEMPERATURE")
    max_tokens: int = Field(default=4096, alias="OPENAI_MAX_TOKENS")
class CohereSettings(BaseSettings):
    """Cohere API settings for reranking."""
    api_key: str = Field(alias="COHERE_API_KEY")
    rerank_model: str = Field(
        default="rerank-english-v3.0", alias="COHERE_RERANK_MODEL"
    )
    rerank_top_n: int = Field(default=5, alias="COHERE_RERANK_TOP_N")
class VectorSearchSettings(BaseSettings):
    """Vector / semantic search settings."""
    top_k: int = Field(default=20, alias="VECTOR_SEARCH_TOP_K")
    hybrid_alpha: float = Field(default=0.7, alias="HYBRID_SEARCH_ALPHA")
    multi_query_count: int = Field(default=3, alias="MULTI_QUERY_COUNT")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
class ChunkingSettings(BaseSettings):
    """Document chunking settings."""
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
class StorageSettings(BaseSettings):
    """File storage settings."""
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    allowed_extensions: list[str] = Field(
        default=["pdf", "docx"], alias="ALLOWED_EXTENSIONS"
    )
    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
class TelemetrySettings(BaseSettings):
    """Azure Monitor / OpenTelemetry settings."""
    connection_string: str | None = Field(
        default=None, alias="APPLICATIONINSIGHTS_CONNECTION_STRING"
    )
    service_name: str = Field(
        default="smart-client-rag", alias="OTEL_SERVICE_NAME"
    )
    service_version: str = Field(default="0.1.0", alias="OTEL_SERVICE_VERSION")
    @property
    def enabled(self) -> bool:
        return self.connection_string is not None
class AgentSettings(BaseSettings):
    """Settings specific to the LangGraph agentic chat module."""
    # psycopg3 plain DSN used by AsyncPostgresSaver.
    # Falls back to DATABASE_URL with +asyncpg stripped if not provided.
    checkpointer_dsn: str | None = Field(
        default=None, alias="AGENT_CHECKPOINTER_DSN"
    )
    # RAG pipeline tuning — separate from the main search settings so the
    # agent can use a lighter configuration without impacting regular search.
    rag_top_k: int = Field(default=20, alias="AGENT_RAG_TOP_K")
    rag_multi_query_count: int = Field(default=3, alias="AGENT_RAG_MULTI_QUERY_COUNT")
    rag_rerank_top_n: int = Field(default=5, alias="AGENT_RAG_RERANK_TOP_N")
    rag_rerank_enabled: bool = Field(default=True, alias="AGENT_RAG_RERANK_ENABLED")

class Settings(BaseSettings):
    """Aggregated application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    # Sub-settings (loaded individually to allow alias resolution)
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_DEBUG: bool = False
    APP_HOST: str = "localhost"
    APP_PORT: int = 8000
    DATABASE_URL: str = "postgresql+asyncpg://postgres:admin@localhost:5432/smart_client_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    OPENAI_EMBEDDING_DIMENSIONS: int = 3072
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    OPENAI_CHAT_TEMPERATURE: float = 0.0
    OPENAI_MAX_TOKENS: int = 4096
    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"
    COHERE_RERANK_TOP_N: int = 5
    VECTOR_SEARCH_TOP_K: int = 20
    HYBRID_SEARCH_ALPHA: float = 0.7
    MULTI_QUERY_COUNT: int = 3
    RERANK_ENABLED: bool = True
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = ["pdf", "docx"]
    APPLICATIONINSIGHTS_CONNECTION_STRING: str | None = None
    OTEL_SERVICE_NAME: str = "smart-client-rag"
    OTEL_SERVICE_VERSION: str = "0.1.0"
    # ── Agent / LangGraph settings ────────────────────────────────────────
    # Plain psycopg3 DSN (postgresql://...) used by AsyncPostgresSaver.
    # If omitted, derived automatically from DATABASE_URL.
    AGENT_CHECKPOINTER_DSN: str | None = None
    AGENT_RAG_TOP_K: int = 20
    AGENT_RAG_MULTI_QUERY_COUNT: int = 3
    AGENT_RAG_RERANK_TOP_N: int = 5
    AGENT_RAG_RERANK_ENABLED: bool = True
    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def parse_extensions(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    @property
    def app(self) -> AppSettings:
        return AppSettings(
            APP_ENV=self.APP_ENV,
            APP_DEBUG=self.APP_DEBUG,
            APP_HOST=self.APP_HOST,
            APP_PORT=self.APP_PORT,
        )
    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings(
            DATABASE_URL=self.DATABASE_URL,
            DATABASE_POOL_SIZE=self.DATABASE_POOL_SIZE,
            DATABASE_MAX_OVERFLOW=self.DATABASE_MAX_OVERFLOW,
            DATABASE_POOL_TIMEOUT=self.DATABASE_POOL_TIMEOUT,
            DATABASE_ECHO=self.DATABASE_ECHO,
        )
    @property
    def openai(self) -> OpenAISettings:
        return OpenAISettings(
            OPENAI_API_KEY=self.OPENAI_API_KEY,
            OPENAI_EMBEDDING_MODEL=self.OPENAI_EMBEDDING_MODEL,
            OPENAI_EMBEDDING_DIMENSIONS=self.OPENAI_EMBEDDING_DIMENSIONS,
            OPENAI_CHAT_MODEL=self.OPENAI_CHAT_MODEL,
            OPENAI_CHAT_TEMPERATURE=self.OPENAI_CHAT_TEMPERATURE,
            OPENAI_MAX_TOKENS=self.OPENAI_MAX_TOKENS,
        )
    @property
    def cohere(self) -> CohereSettings:
        return CohereSettings(
            COHERE_API_KEY=self.COHERE_API_KEY,
            COHERE_RERANK_MODEL=self.COHERE_RERANK_MODEL,
            COHERE_RERANK_TOP_N=self.COHERE_RERANK_TOP_N,
        )
    @property
    def vector_search(self) -> VectorSearchSettings:
        return VectorSearchSettings(
            VECTOR_SEARCH_TOP_K=self.VECTOR_SEARCH_TOP_K,
            HYBRID_SEARCH_ALPHA=self.HYBRID_SEARCH_ALPHA,
            MULTI_QUERY_COUNT=self.MULTI_QUERY_COUNT,
            RERANK_ENABLED=self.RERANK_ENABLED,
        )
    @property
    def chunking(self) -> ChunkingSettings:
        return ChunkingSettings(
            CHUNK_SIZE=self.CHUNK_SIZE,
            CHUNK_OVERLAP=self.CHUNK_OVERLAP,
        )
    @property
    def storage(self) -> StorageSettings:
        return StorageSettings(
            UPLOAD_DIR=self.UPLOAD_DIR,
            MAX_FILE_SIZE_MB=self.MAX_FILE_SIZE_MB,
            ALLOWED_EXTENSIONS=self.ALLOWED_EXTENSIONS,
        )
    @property
    def telemetry(self) -> TelemetrySettings:
        return TelemetrySettings(
            APPLICATIONINSIGHTS_CONNECTION_STRING=self.APPLICATIONINSIGHTS_CONNECTION_STRING,
            OTEL_SERVICE_NAME=self.OTEL_SERVICE_NAME,
            OTEL_SERVICE_VERSION=self.OTEL_SERVICE_VERSION,
        )
    @property
    def agent(self) -> AgentSettings:
        return AgentSettings(
            AGENT_CHECKPOINTER_DSN=self.AGENT_CHECKPOINTER_DSN,
            AGENT_RAG_TOP_K=self.AGENT_RAG_TOP_K,
            AGENT_RAG_MULTI_QUERY_COUNT=self.AGENT_RAG_MULTI_QUERY_COUNT,
            AGENT_RAG_RERANK_TOP_N=self.AGENT_RAG_RERANK_TOP_N,
            AGENT_RAG_RERANK_ENABLED=self.AGENT_RAG_RERANK_ENABLED,
        )
    def get_checkpointer_dsn(self) -> str:
        """Return the psycopg3 DSN for the LangGraph checkpointer.

        If AGENT_CHECKPOINTER_DSN is set, use it as-is.
        Otherwise derive from DATABASE_URL by stripping the '+asyncpg' driver
        (psycopg3 uses plain 'postgresql://' scheme).
        """
        if self.AGENT_CHECKPOINTER_DSN:
            return self.AGENT_CHECKPOINTER_DSN
        # e.g. postgresql+asyncpg://... → postgresql://...
        return self.DATABASE_URL.replace("+asyncpg", "")
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()