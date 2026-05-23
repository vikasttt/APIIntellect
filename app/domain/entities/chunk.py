"""Chunk domain entity."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any
class Chunk:
    """Represents a single text chunk derived from a document."""
    def __init__(
        self,
        *,
        id: uuid.UUID | None = None,
        document_id: uuid.UUID,
        user_id: str,
        content: str,
        chunk_index: int,
        token_count: int = 0,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        page_number: int | None = None,
        section_title: str | None = None,
        chunk_type: str = "text",
        created_at: datetime | None = None,
    ) -> None:
        self.id: uuid.UUID = id or uuid.uuid4()
        self.document_id = document_id
        self.user_id = user_id
        self.content = content
        self.chunk_index = chunk_index
        self.token_count = token_count
        self.embedding = embedding
        self.metadata: dict[str, Any] = metadata or {}
        self.page_number = page_number
        self.section_title = section_title
        self.chunk_type = chunk_type
        self.created_at: datetime = created_at or datetime.utcnow()
    def has_embedding(self) -> bool:
        return self.embedding is not None and len(self.embedding) > 0
    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.id} doc={self.document_id} "
            f"index={self.chunk_index} type={self.chunk_type}>"
        )