"""Cohere reranking service."""
from __future__ import annotations
from app.config.logging import get_logger
from app.domain.entities.chunk import Chunk
logger = get_logger(__name__)
class CohereRerankService:
    """Rerank chunks using Cohere Rerank API."""
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        top_n: int = 5,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._top_n = top_n
    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rerank chunks against query; return top_n with scores."""
        effective_top_n = top_n or self._top_n
        if not chunks:
            return []
        try:
            import cohere
            co = cohere.AsyncClient(api_key=self._api_key)
            documents = [c.content for c in chunks]
            response = await co.rerank(
                query=query,
                documents=documents,
                model=self._model,
                top_n=min(effective_top_n, len(chunks)),
            )
            results = [
                (chunks[r.index], r.relevance_score)
                for r in response.results
            ]
            logger.debug(
                "Cohere rerank complete",
                query=query,
                returned=len(results),
            )
            return results
        except Exception as exc:
            logger.warning(
                "Cohere rerank failed, falling back to original order",
                error=str(exc),
            )
            return [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks[:effective_top_n])]