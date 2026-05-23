"""
AnyDI Dependency Injection Container.
Scopes:
- Singleton: LLM, embedding model, multi-query, reranker
- Scoped: DB session
- Transient: Repositories, Use cases
"""
from __future__ import annotations
from typing import AsyncIterator
import anydi
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, AsyncEngine
from app.config.settings import Settings, get_settings
from app.infrastructure.db.database import create_engine, create_session_factory
from app.infrastructure.db.repositories.chunk_repository import SqlChunkRepository
from app.infrastructure.db.repositories.conversation_repository import SqlConversationRepository
from app.infrastructure.db.repositories.document_repository import SqlDocumentRepository
from app.infrastructure.chunking.docling_chunker import DoclingChunkingService
from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddingService
from app.infrastructure.llm.cohere_rerank import CohereRerankService
from app.infrastructure.llm.multi_query import MultiQueryService
from app.infrastructure.llm.openai_chat import OpenAIChatService
from app.infrastructure.parsing.docling_parser import DoclingParser
from app.domain.services.document_service import DocumentDomainService
from app.application.use_cases.upload_document import UploadDocumentUseCase
from app.application.use_cases.process_document import ProcessDocumentUseCase
from app.application.use_cases.document_crud import (
    DeleteDocumentUseCase,
    GetDocumentUseCase,
    GetDocumentsUseCase,
    UpdateDocumentUseCase,
)
from app.application.use_cases.search import HybridSearchUseCase, SemanticSearchUseCase
from app.application.use_cases.conversation import (
    ChatUseCase,
    CreateConversationUseCase,
    DeleteConversationUseCase,
    GetConversationsUseCase,
    GetMessagesUseCase,
)
def build_container(settings: Settings) -> anydi.Container:
    """Build and configure the AnyDI container."""
    container = anydi.Container()
    # ── Singletons ─────────────────────────────────────────────────────────
    @container.provider(scope="singleton")
    def provide_settings() -> Settings:
        return settings
    @container.provider(scope="singleton")
    def provide_engine(s: Settings) -> AsyncEngine:
        return create_engine(s)
    @container.provider(scope="singleton")
    def provide_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
        return create_session_factory(engine)
    @container.provider(scope="singleton")
    def provide_embedding_service(s: Settings) -> OpenAIEmbeddingService:
        return OpenAIEmbeddingService(
            api_key=s.OPENAI_API_KEY,
            model=s.OPENAI_EMBEDDING_MODEL,
            dimensions=s.OPENAI_EMBEDDING_DIMENSIONS,
        )
    @container.provider(scope="singleton")
    def provide_multi_query_service(s: Settings) -> MultiQueryService:
        return MultiQueryService(api_key=s.OPENAI_API_KEY)
    @container.provider(scope="singleton")
    def provide_rerank_service(s: Settings) -> CohereRerankService:
        return CohereRerankService(
            api_key=s.COHERE_API_KEY,
            model=s.COHERE_RERANK_MODEL,
            top_n=s.COHERE_RERANK_TOP_N,
        )
    @container.provider(scope="singleton")
    def provide_llm_service(s: Settings) -> OpenAIChatService:
        return OpenAIChatService(
            api_key=s.OPENAI_API_KEY,
            model=s.OPENAI_CHAT_MODEL,
            temperature=s.OPENAI_CHAT_TEMPERATURE,
            max_tokens=s.OPENAI_MAX_TOKENS,
        )
    @container.provider(scope="singleton")
    def provide_parser(s: Settings) -> DoclingParser:
        return DoclingParser(openai_api_key=s.OPENAI_API_KEY)
    @container.provider(scope="singleton")
    def provide_chunker(s: Settings) -> DoclingChunkingService:
        return DoclingChunkingService(
            chunk_size=s.CHUNK_SIZE, chunk_overlap=s.CHUNK_OVERLAP
        )
    @container.provider(scope="singleton")
    def provide_domain_service() -> DocumentDomainService:
        return DocumentDomainService()
    return container
# Module-level container instance (initialized in main.py lifespan)
_container: anydi.Container | None = None
def get_container() -> anydi.Container:
    if _container is None:
        raise RuntimeError("DI container not initialized")
    return _container
def set_container(container: anydi.Container) -> None:
    global _container
    _container = container