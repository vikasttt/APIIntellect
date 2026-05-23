"""Application-layer DTOs for conversations and chat."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel
from app.domain.entities.conversation import MessageRole
class CreateConversationDTO(BaseModel):
    user_id: str
    title: str | None = None
    document_ids: list[uuid.UUID] = []
class ChatMessageDTO(BaseModel):
    conversation_id: uuid.UUID
    user_id: str
    content: str
    document_ids: list[uuid.UUID] = []
class MessageResponseDTO(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: MessageRole
    content: str
    source_chunks: list[uuid.UUID]
    metadata: dict[str, Any]
    created_at: datetime
    model_config = {"from_attributes": True}
class ConversationResponseDTO(BaseModel):
    id: uuid.UUID
    user_id: str
    title: str | None
    document_ids: list[uuid.UUID]
    message_count: int
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}
class ConversationListDTO(BaseModel):
    items: list[ConversationResponseDTO]
    total: int
    skip: int
    limit: int
class ChatResponseDTO(BaseModel):
    message: MessageResponseDTO
    sources: list[dict[str, Any]]