"""Abstract chunk repository interface."""
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from app.domain.entities.chunk import Chunk
class IChunkRepository(ABC):
    """Port: chunk persistence and vector operations."""
    @abstractmethod
    async def bulk_create(self, chunks: list[Chunk]) -> list[Chunk]:
        """Bulk-insert chunks and return persisted entities."""
        ...
    @abstractmethod
    async def get_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        """Return all chunks for a document ordered by chunk_index."""
        ...
    @abstractmethod
    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        """Delete all chunks for a document; return deleted count."""
        ...
    @abstractmethod
    async def similarity_search(
        self,
        embedding: list[float],
        *,
        user_id: str,
        top_k: int = 10,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return chunks ordered by cosine similarity with scores."""
        ...
    @abstractmethod
    async def hybrid_search(
        self,
        embedding: list[float],
        query_text: str,
        *,
        user_id: str,
        top_k: int = 10,
        alpha: float = 0.7,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Hybrid vector + keyword search with RRF fusion."""
        ...