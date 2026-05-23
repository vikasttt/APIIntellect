"""Application-layer Data Transfer Objects for documents."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from app.domain.entities.document import DocumentStatus
class UploadDocumentDTO(BaseModel):
    """Input DTO for document upload."""
    user_id: str
    original_filename: str
    content_type: str
    file_size: int
    file_path: str
    title: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
class UpdateDocumentDTO(BaseModel):
    """Input DTO for updating document metadata."""
    title: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None
class DocumentResponseDTO(BaseModel):
    """Output DTO representing a document."""
    id: uuid.UUID
    user_id: str
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    status: DocumentStatus
    title: str | None
    description: str | None
    metadata: dict[str, Any]
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None
    model_config = {"from_attributes": True}
class DocumentListDTO(BaseModel):
    """Paginated list of documents."""
    items: list[DocumentResponseDTO]
    total: int
    skip: int
    limit: int