"""Use case: semantic and hybrid search."""
from __future__ import annotations
import uuid
from typing import Protocol
from app.application.dto.chunk_dto import ChunkResponseDTO, SearchResponseDTO, SearchResultDTO
from app.config.logging import get_logger
from app.domain.entities.chunk import Chunk
from app.domain.repositories.chunk_repository import IChunkRepository
logger = get_logger(__name__)
class IEmbeddingPort(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...
class IMultiQueryPort(Protocol):
    async def generate_queries(self, query: str, n: int) -> list[str]: ...
class IRerankPort(Protocol):
    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int
    ) -> list[tuple[Chunk, float]]: ...
def _chunk_to_dto(chunk: Chunk) -> ChunkResponseDTO:
    return ChunkResponseDTO(
        id=chunk.id,
        document_id=chunk.document_id,
        user_id=chunk.user_id,
        content=chunk.content,
        chunk_index=chunk.chunk_index,
        token_count=chunk.token_count,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        chunk_type=chunk.chunk_type,
        metadata=chunk.metadata,
        created_at=chunk.created_at,
    )
class SemanticSearchUseCase:
    """Multi-query → vector search → Cohere rerank pipeline."""
    def __init__(
        self,
        chunk_repo: IChunkRepository,
        embedder: IEmbeddingPort,
        multi_query: IMultiQueryPort,
        reranker: IRerankPort,
        top_k: int = 20,
        multi_query_count: int = 3,
        rerank_enabled: bool = True,
        rerank_top_n: int = 5,
    ) -> None:
        self._repo = chunk_repo
        self._embedder = embedder
        self._multi_query = multi_query
        self._reranker = reranker
        self._top_k = top_k
        self._multi_query_count = multi_query_count
        self._rerank_enabled = rerank_enabled
        self._rerank_top_n = rerank_top_n
    async def execute(
        self,
        query: str,
        user_id: str,
        *,
        document_ids: list[uuid.UUID] | None = None,
        top_k: int | None = None,
    ) -> SearchResponseDTO:
        effective_top_k = top_k or self._top_k
        logger.info("Semantic search", query=query, user=user_id, top_k=effective_top_k)
        # 1. Generate multiple queries for recall diversity
        sub_queries = await self._multi_query.generate_queries(
            query, self._multi_query_count
        )
        all_queries = [query] + sub_queries
        logger.debug("Generated sub-queries", count=len(sub_queries))
        # 2. Embed all queries and collect candidate chunks (deduplicated by ID)
        seen: dict[uuid.UUID, tuple[Chunk, float]] = {}
        for q in all_queries:
            embedding = await self._embedder.embed_query(q)
            print(embedding,  "embedding length:", len(embedding));
            results = await self._repo.similarity_search(
                embedding,
                user_id=user_id,
                top_k=effective_top_k,
                document_ids=document_ids,
            )
            for chunk, score in results:
                if chunk.id not in seen or seen[chunk.id][1] < score:
                    seen[chunk.id] = (chunk, score)
        candidates = list(seen.values())
        # 3. Optional Cohere reranking (inverse fusion order → reranker)
        if self._rerank_enabled and candidates:
            chunks_only = [c for c, _ in candidates]
            reranked = await self._reranker.rerank(
                query, chunks_only, self._rerank_top_n
            )
            final: list[tuple[Chunk, float]] = reranked
        else:
            final = sorted(candidates, key=lambda x: x[1], reverse=True)[
                : self._rerank_top_n
            ]
        results_dto = [
            SearchResultDTO(
                chunk=_chunk_to_dto(chunk),
                score=score,
                document_id=chunk.document_id,
            )
            for chunk, score in final
        ]
        return SearchResponseDTO(query=query, results=results_dto, total=len(results_dto))
class HybridSearchUseCase:
    """Hybrid vector + keyword search with RRF fusion → rerank."""
    def __init__(
        self,
        chunk_repo: IChunkRepository,
        embedder: IEmbeddingPort,
        reranker: IRerankPort,
        top_k: int = 20,
        alpha: float = 0.7,
        rerank_enabled: bool = True,
        rerank_top_n: int = 5,
    ) -> None:
        self._repo = chunk_repo
        self._embedder = embedder
        self._reranker = reranker
        self._top_k = top_k
        self._alpha = alpha
        self._rerank_enabled = rerank_enabled
        self._rerank_top_n = rerank_top_n
    async def execute(
        self,
        query: str,
        user_id: str,
        *,
        document_ids: list[uuid.UUID] | None = None,
        top_k: int | None = None,
    ) -> SearchResponseDTO:
        effective_top_k = top_k or self._top_k
        embedding = await self._embedder.embed_query(query)
        candidates = await self._repo.hybrid_search(
            embedding,
            query,
            user_id=user_id,
            top_k=effective_top_k,
            alpha=self._alpha,
            document_ids=document_ids,
        )
        if self._rerank_enabled and candidates:
            chunks_only = [c for c, _ in candidates]
            final = await self._reranker.rerank(query, chunks_only, self._rerank_top_n)
        else:
            final = candidates[: self._rerank_top_n]
        results_dto = [
            SearchResultDTO(
                chunk=_chunk_to_dto(chunk),
                score=score,
                document_id=chunk.document_id,
            )
            for chunk, score in final
        ]
        return SearchResponseDTO(query=query, results=results_dto, total=len(results_dto))
