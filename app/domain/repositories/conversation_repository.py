"""Abstract conversation repository interface."""
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from app.domain.entities.conversation import Conversation, Message
class IConversationRepository(ABC):
    """Port: conversation and message persistence."""
    @abstractmethod
    async def create(self, conversation: Conversation) -> Conversation:
        ...
    @abstractmethod
    async def get_by_id(self, conversation_id: uuid.UUID) -> Conversation | None:
        ...
    @abstractmethod
    async def get_by_user(
        self,
        user_id: str,
        *,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[Conversation], int]:
        ...
    @abstractmethod
    async def delete(self, conversation_id: uuid.UUID) -> bool:
        ...
    @abstractmethod
    async def add_message(self, message: Message) -> Message:
        ...
    @abstractmethod
    async def get_messages(
        self,
        conversation_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 50,
    ) -> list[Message]:
        ...
