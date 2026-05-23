"""Conversation and Message domain entities."""
from __future__ import annotations
import uuid
from datetime import datetime
from enum import Enum
from typing import Any
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
class Message:
    """Single chat message in a conversation."""
    def __init__(
        self,
        *,
        id: uuid.UUID | None = None,
        conversation_id: uuid.UUID,
        role: MessageRole,
        content: str,
        source_chunks: list[uuid.UUID] | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.id: uuid.UUID = id or uuid.uuid4()
        self.conversation_id = conversation_id
        self.role = role
        self.content = content
        self.source_chunks: list[uuid.UUID] = source_chunks or []
        self.metadata: dict[str, Any] = metadata or {}
        self.created_at: datetime = created_at or datetime.utcnow()
    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role}>"
class Conversation:
    """Conversation thread linking messages to a user."""
    def __init__(
        self,
        *,
        id: uuid.UUID | None = None,
        user_id: str,
        title: str | None = None,
        document_ids: list[uuid.UUID] | None = None,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self.id: uuid.UUID = id or uuid.uuid4()
        self.user_id = user_id
        self.title = title
        self.document_ids: list[uuid.UUID] = document_ids or []
        self.messages: list[Message] = messages or []
        self.metadata: dict[str, Any] = metadata or {}
        self.created_at: datetime = created_at or datetime.utcnow()
        self.updated_at: datetime = updated_at or datetime.utcnow()
    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    def __repr__(self) -> str:
        return f"<Conversation id={self.id} user={self.user_id}>"