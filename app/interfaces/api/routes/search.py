"""Search API routes."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from app.application.use_cases.search import HybridSearchUseCase, SemanticSearchUseCase
from app.interfaces.api.dependencies.providers import (
    get_hybrid_search_use_case,
    get_semantic_search_use_case,
)
from app.interfaces.api.schemas.schemas import (
    ChunkResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
router = APIRouter(prefix="/search", tags=["Search"])
@router.post(
    "",
    response_model=SearchResponse,
    summary="Semantic search (vector search with multi-query + reranking)",
)
async def semantic_search(
    body: SearchRequest,
    search_uc: SemanticSearchUseCase = Depends(get_semantic_search_use_case),
) -> SearchResponse:
    result = await search_uc.execute(
        body.query,
        body.user_id,
        document_ids=body.document_ids or None,
        top_k=body.top_k,
    )
    return SearchResponse(
        query=result.query,
        results=[
            SearchResultItem(
                chunk=ChunkResponse(
                    id=r.chunk.id,
                    document_id=r.chunk.document_id,
                    content=r.chunk.content,
                    chunk_index=r.chunk.chunk_index,
                    token_count=r.chunk.token_count,
                    page_number=r.chunk.page_number,
                    section_title=r.chunk.section_title,
                    chunk_type=r.chunk.chunk_type,
                    metadata=r.chunk.metadata,
                    created_at=r.chunk.created_at,
                ),
                score=r.score,
                document_id=r.document_id,
            )
            for r in result.results
        ],
        total=result.total,
    )
@router.post(
    "/hybrid",
    response_model=SearchResponse,
    summary="Hybrid search (vector + keyword with RRF fusion + reranking)",
)
async def hybrid_search(
    body: SearchRequest,
    search_uc: HybridSearchUseCase = Depends(get_hybrid_search_use_case),
) -> SearchResponse:
    result = await search_uc.execute(
        body.query,
        body.user_id,
        document_ids=body.document_ids or None,
        top_k=body.top_k,
    )
    return SearchResponse(
        query=result.query,
        results=[
            SearchResultItem(
                chunk=ChunkResponse(
                    id=r.chunk.id,
                    document_id=r.chunk.document_id,
                    content=r.chunk.content,
                    chunk_index=r.chunk.chunk_index,
                    token_count=r.chunk.token_count,
                    page_number=r.chunk.page_number,
                    section_title=r.chunk.section_title,
                    chunk_type=r.chunk.chunk_type,
                    metadata=r.chunk.metadata,
                    created_at=r.chunk.created_at,
                ),
                score=r.score,
                document_id=r.document_id,
            )
            for r in result.results
        ],
        total=result.total,
    )
