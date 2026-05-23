"""
RagServices — a dependency bundle passed to the LangGraph graph at build time.

The `rag_node` inside `agents.py` uses this to call the existing production
vector-search pipeline (multi-query → pgvector similarity → Cohere rerank)
instead of answering from a static company_info string.

The bundle is constructed once during FastAPI lifespan (from the AnyDI
container) and stored on a module-level variable that `get_rag_services()`
returns.  This avoids passing services through the LangGraph state (which
must be JSON-serialisable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Optional, Tuple
import uuid


# ── Protocol-style type aliases (kept simple to avoid circular imports) ───────

class _EmbedderProto:
    async def embed_query(self, text: str) -> List[float]: ...  # type: ignore[empty-body]


class _MultiQueryProto:
    async def generate_queries(self, query: str, n: int) -> List[str]: ...  # type: ignore[empty-body]


class _RerankProto:
    async def rerank(self, query: str, chunks: Any, top_n: int) -> List[Tuple[Any, float]]: ...  # type: ignore[empty-body]


# ── Services bundle ───────────────────────────────────────────────────────────

@dataclass
class RagServices:
    """
    Holds references to the live infrastructure services needed for RAG.

    Fields
    ------
    embedder
        OpenAIEmbeddingService — embeds a query string → float list.
    multi_query
        MultiQueryService — generates query variants to improve recall.
    reranker
        CohereRerankService — reranks candidate chunks by relevance.
    session_factory
        async_sessionmaker — used to open a short-lived DB session per call.
    top_k
        Number of candidate chunks to retrieve before reranking.
    multi_query_count
        Number of alternative queries to generate.
    rerank_top_n
        How many chunks to keep after reranking.
    rerank_enabled
        If False, skip Cohere reranking and return top-k by score.
    """
    embedder:          Any        # OpenAIEmbeddingService
    multi_query:       Any        # MultiQueryService
    reranker:          Any        # CohereRerankService
    session_factory:   Any        # async_sessionmaker[AsyncSession]
    top_k:             int = 20
    multi_query_count: int = 3
    rerank_top_n:      int = 5
    rerank_enabled:    bool = True


# ── Module-level singleton ────────────────────────────────────────────────────

_rag_services: Optional[RagServices] = None


def set_rag_services(services: RagServices) -> None:
    """Called once during FastAPI lifespan after AnyDI container is built."""
    global _rag_services
    _rag_services = services


def get_rag_services() -> Optional[RagServices]:
    """Returns the shared RagServices instance, or None if not yet configured."""
    return _rag_services
