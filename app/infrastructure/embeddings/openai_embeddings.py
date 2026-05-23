"""OpenAI embedding service with batching and retry."""
from __future__ import annotations
import asyncio
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config.logging import get_logger
from app.domain.entities.chunk import Chunk
logger = get_logger(__name__)
_BATCH_SIZE = 512
class OpenAIEmbeddingService:
    """Async OpenAI embedding service with batching."""
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._api_key = api_key
    def _get_client(self):  # type: ignore[return]
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self._api_key)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = await client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        embeddings = await self._embed_batch([text])
        return embeddings[0]
    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all chunks in batches; mutate and return chunks."""
        texts = [chunk.content for chunk in chunks]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)
            logger.debug(
                "Embedded batch",
                batch_start=i,
                batch_size=len(batch),
            )
        for chunk, embedding in zip(chunks, all_embeddings, strict=True):
            chunk.embedding = embedding
            chunk.token_count = len(chunk.content.split())
        return chunks
