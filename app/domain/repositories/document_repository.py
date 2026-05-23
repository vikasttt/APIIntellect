"""Abstract document repository interface."""
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from typing import Any
from app.domain.entities.document import Document, DocumentStatus
class IDocumentRepository(ABC):
    """Port: document persistence operations."""
    @abstractmethod
    async def create(self, document: Document) -> Document:
        """Persist a new document and return it with DB-assigned fields."""
        ...
    @abstractmethod
    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        """Retrieve a document by its ID, or None."""
        ...
    @abstractmethod
    async def get_by_user(
        self,
        user_id: str,
        *,
        skip: int = 0,
        limit: int = 20,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        """Return paginated documents for a user plus total count."""
        ...
    @abstractmethod
    async def update(self, document: Document) -> Document:
        """Persist changes to an existing document."""
        ...
    @abstractmethod
    async def delete(self, document_id: uuid.UUID) -> bool:
        """Delete a document and return True if it existed."""
        ...
    @abstractmethod
    async def update_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Lightweight status update without loading full entity."""
        ...