"""FastAPI dependency providers using AnyDI container."""
from __future__ import annotations
from typing import AsyncIterator
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from app.interfaces.api.dependencies.container import get_container
from app.infrastructure.db.repositories.chunk_repository import SqlChunkRepository
from app.infrastructure.db.repositories.conversation_repository import SqlConversationRepository
from app.infrastructure.db.repositories.document_repository import SqlDocumentRepository
from app.infrastructure.db.repositories.tenant_repository import SqlTenantRepository
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
from app.config.settings import Settings
# ── Singleton dependencies ─────────────────────────────────────────────────
def get_settings(request: Request) -> Settings:
    return get_container().resolve(Settings)
def get_embedding_service(request: Request) -> OpenAIEmbeddingService:
    return get_container().resolve(OpenAIEmbeddingService)
def get_multi_query_service(request: Request) -> MultiQueryService:
    return get_container().resolve(MultiQueryService)
def get_rerank_service(request: Request) -> CohereRerankService:
    return get_container().resolve(CohereRerankService)
def get_llm_service(request: Request) -> OpenAIChatService:
    return get_container().resolve(OpenAIChatService)
def get_parser(request: Request) -> DoclingParser:
    return get_container().resolve(DoclingParser)
def get_chunker(request: Request) -> DoclingChunkingService:
    return get_container().resolve(DoclingChunkingService)
def get_domain_service(request: Request) -> DocumentDomainService:
    return get_container().resolve(DocumentDomainService)
# ── Scoped: DB session per request ────────────────────────────────────────
async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = get_container().resolve(
        async_sessionmaker[AsyncSession]
    )
    async with factory() as session:
        async with session.begin():
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
# ── Transient: Repositories ───────────────────────────────────────────────
def get_document_repo(
    session: AsyncSession = Depends(get_db_session),
) -> SqlDocumentRepository:
    return SqlDocumentRepository(session)
def get_chunk_repo(
    session: AsyncSession = Depends(get_db_session),
) -> SqlChunkRepository:
    return SqlChunkRepository(session)
def get_conversation_repo(
    session: AsyncSession = Depends(get_db_session),
) -> SqlConversationRepository:
    return SqlConversationRepository(session)
def get_tenant_repo(
    session: AsyncSession = Depends(get_db_session),
) -> SqlTenantRepository:
    return SqlTenantRepository(session)
# ── Transient: Use cases ──────────────────────────────────────────────────
def get_upload_use_case(
    doc_repo: SqlDocumentRepository = Depends(get_document_repo),
    domain_svc: DocumentDomainService = Depends(get_domain_service),
) -> UploadDocumentUseCase:
    return UploadDocumentUseCase(doc_repo, domain_svc)
def get_process_use_case(
    request: Request,
    doc_repo: SqlDocumentRepository = Depends(get_document_repo),
    chunk_repo: SqlChunkRepository = Depends(get_chunk_repo),
    domain_svc: DocumentDomainService = Depends(get_domain_service),
) -> ProcessDocumentUseCase:
    return ProcessDocumentUseCase(
        document_repo=doc_repo,
        chunk_repo=chunk_repo,
        parser=get_parser(request),
        chunker=get_chunker(request),
        embedder=get_embedding_service(request),
        domain_service=domain_svc,
    )
def get_documents_use_case(
    doc_repo: SqlDocumentRepository = Depends(get_document_repo),
) -> GetDocumentsUseCase:
    return GetDocumentsUseCase(doc_repo)
def get_document_use_case(
    doc_repo: SqlDocumentRepository = Depends(get_document_repo),
) -> GetDocumentUseCase:
    return GetDocumentUseCase(doc_repo)
def get_update_use_case(
    doc_repo: SqlDocumentRepository = Depends(get_document_repo),
) -> UpdateDocumentUseCase:
    return UpdateDocumentUseCase(doc_repo)
def get_delete_use_case(
    doc_repo: SqlDocumentRepository = Depends(get_document_repo),
) -> DeleteDocumentUseCase:
    return DeleteDocumentUseCase(doc_repo)
def get_semantic_search_use_case(
    request: Request,
    chunk_repo: SqlChunkRepository = Depends(get_chunk_repo),
) -> SemanticSearchUseCase:
    s = get_settings(request)
    return SemanticSearchUseCase(
        chunk_repo=chunk_repo,
        embedder=get_embedding_service(request),
        multi_query=get_multi_query_service(request),
        reranker=get_rerank_service(request),
        top_k=s.VECTOR_SEARCH_TOP_K,
        multi_query_count=s.MULTI_QUERY_COUNT,
        rerank_enabled=s.RERANK_ENABLED,
        rerank_top_n=s.COHERE_RERANK_TOP_N,
    )
def get_hybrid_search_use_case(
    request: Request,
    chunk_repo: SqlChunkRepository = Depends(get_chunk_repo),
) -> HybridSearchUseCase:
    s = get_settings(request)
    return HybridSearchUseCase(
        chunk_repo=chunk_repo,
        embedder=get_embedding_service(request),
        reranker=get_rerank_service(request),
        top_k=s.VECTOR_SEARCH_TOP_K,
        alpha=s.HYBRID_SEARCH_ALPHA,
        rerank_enabled=s.RERANK_ENABLED,
        rerank_top_n=s.COHERE_RERANK_TOP_N,
    )
def get_create_conversation_use_case(
    conv_repo: SqlConversationRepository = Depends(get_conversation_repo),
) -> CreateConversationUseCase:
    return CreateConversationUseCase(conv_repo)
def get_conversations_use_case(
    conv_repo: SqlConversationRepository = Depends(get_conversation_repo),
) -> GetConversationsUseCase:
    return GetConversationsUseCase(conv_repo)
def get_delete_conversation_use_case(
    conv_repo: SqlConversationRepository = Depends(get_conversation_repo),
) -> DeleteConversationUseCase:
    return DeleteConversationUseCase(conv_repo)
def get_messages_use_case(
    conv_repo: SqlConversationRepository = Depends(get_conversation_repo),
) -> GetMessagesUseCase:
    return GetMessagesUseCase(conv_repo)
def get_chat_use_case(
    request: Request,
    conv_repo: SqlConversationRepository = Depends(get_conversation_repo),
    chunk_repo: SqlChunkRepository = Depends(get_chunk_repo),
) -> ChatUseCase:
    s = get_settings(request)
    semantic_uc = SemanticSearchUseCase(
        chunk_repo=chunk_repo,
        embedder=get_embedding_service(request),
        multi_query=get_multi_query_service(request),
        reranker=get_rerank_service(request),
        top_k=s.VECTOR_SEARCH_TOP_K,
        multi_query_count=s.MULTI_QUERY_COUNT,
        rerank_enabled=s.RERANK_ENABLED,
        rerank_top_n=s.COHERE_RERANK_TOP_N,
    )
    class _SearchAdapter:
        async def search(self, query: str, user_id: str, document_ids: list) -> list:
            result = await semantic_uc.execute(
                query, user_id, document_ids=document_ids or None
            )
            return result.results
    return ChatUseCase(
        conversation_repo=conv_repo,
        llm=get_llm_service(request),
        search=_SearchAdapter(),
    )
