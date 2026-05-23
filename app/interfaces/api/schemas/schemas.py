"""Pydantic v2 request/response schemas for the API layer."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field
from app.domain.entities.document import DocumentStatus
# ── Document schemas ───────────────────────────────────────────────────────────
class DocumentCreateRequest(BaseModel):
    title: str | None = Field(None, max_length=512)
    description: str | None = Field(None, max_length=2048)
    metadata: dict[str, Any] = Field(default_factory=dict)
class DocumentUpdateRequest(BaseModel):
    title: str | None = Field(None, max_length=512)
    description: str | None = Field(None, max_length=2048)
    metadata: dict[str, Any] | None = None
class DocumentResponse(BaseModel):
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
class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    skip: int
    limit: int
class ProcessDocumentResponse(BaseModel):
    document_id: uuid.UUID
    chunk_count: int
    message: str
# ── Chunk schemas ──────────────────────────────────────────────────────────────
class ChunkResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    content: str
    chunk_index: int
    token_count: int
    page_number: int | None
    section_title: str | None
    chunk_type: str
    metadata: dict[str, Any]
    created_at: datetime
# ── Search schemas ─────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
class SearchResultItem(BaseModel):
    chunk: ChunkResponse
    score: float
    document_id: uuid.UUID
    document_filename: str | None = None
class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    total: int
# ── Conversation schemas ───────────────────────────────────────────────────────
class CreateConversationRequest(BaseModel):
    user_id: str
    title: str | None = Field(None, max_length=512)
    document_ids: list[uuid.UUID] = Field(default_factory=list)
class ConversationResponse(BaseModel):
    id: uuid.UUID
    user_id: str
    title: str | None
    document_ids: list[uuid.UUID]
    message_count: int
    created_at: datetime
    updated_at: datetime
class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    total: int
    skip: int
    limit: int
class ChatRequest(BaseModel):
    user_id: str
    content: str = Field(..., min_length=1, max_length=10000)
    document_ids: list[uuid.UUID] = Field(default_factory=list)
class MessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: Literal["user", "assistant", "system"]
    content: str
    source_chunks: list[uuid.UUID]
    metadata: dict[str, Any]
    created_at: datetime
class ChatResponse(BaseModel):
    message: MessageResponse
    sources: list[dict[str, Any]]
# ── System schemas ─────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
class ConfigResponse(BaseModel):
    supported_file_types: list[str]
    max_file_size_mb: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    chat_model: str
    rerank_enabled: bool
class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None