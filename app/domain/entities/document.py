"""Document domain entity."""
from __future__ import annotations
import uuid
from datetime import datetime
from enum import Enum
from typing import Any
class DocumentStatus(str, Enum):
    """Lifecycle states of a document."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
class Document:
    """Core document domain entity (plain Python, no ORM dependency)."""
    def __init__(
        self,
        *,
        id: uuid.UUID | None = None,
        user_id: str,
        filename: str,
        original_filename: str,
        file_path: str,
        file_size: int,
        content_type: str,
        status: DocumentStatus = DocumentStatus.UPLOADED,
        title: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_count: int = 0,
        error_message: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        processed_at: datetime | None = None,
    ) -> None:
        self.id: uuid.UUID = id or uuid.uuid4()
        self.user_id = user_id
        self.filename = filename
        self.original_filename = original_filename
        self.file_path = file_path
        self.file_size = file_size
        self.content_type = content_type
        self.status = status
        self.title = title
        self.description = description
        self.metadata: dict[str, Any] = metadata or {}
        self.chunk_count = chunk_count
        self.error_message = error_message
        self.created_at: datetime = created_at or datetime.utcnow()
        self.updated_at: datetime = updated_at or datetime.utcnow()
        self.processed_at = processed_at
    def mark_processing(self) -> None:
        """Transition to processing state."""
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.utcnow()
    def mark_processed(self, chunk_count: int) -> None:
        """Transition to processed state."""
        self.status = DocumentStatus.PROCESSED
        self.chunk_count = chunk_count
        self.processed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    def mark_failed(self, error: str) -> None:
        """Transition to failed state."""
        self.status = DocumentStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.utcnow()
    @property
    def extension(self) -> str:
        """Return lowercase file extension without dot."""
        return self.original_filename.rsplit(".", 1)[-1].lower()
    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"