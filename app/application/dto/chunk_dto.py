"""Application-layer DTOs for chunks."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel
class ChunkResponseDTO(BaseModel):
    """Output DTO for a document chunk."""
    id: uuid.UUID
    document_id: uuid.UUID
    user_id: str
    content: str
    chunk_index: int
    token_count: int
    page_number: int | None
    section_title: str | None
    chunk_type: str
    metadata: dict[str, Any]
    created_at: datetime
    model_config = {"from_attributes": True}
class SearchResultDTO(BaseModel):
    """Single semantic search result."""
    chunk: ChunkResponseDTO
    score: float
    document_id: uuid.UUID
    document_filename: str | None = None
class SearchResponseDTO(BaseModel):
    """Full search response."""
    query: str
    results: list[SearchResultDTO]
    total: int
