"""SQLAlchemy async implementation of IConversationRepository."""
from __future__ import annotations
import uuid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.entities.conversation import Conversation, Message, MessageRole
from app.domain.repositories.conversation_repository import IConversationRepository
from app.infrastructure.db.models import ConversationModel, MessageModel
class SqlConversationRepository(IConversationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
    @staticmethod
    def _conv_to_entity(model: ConversationModel, messages: list[Message] | None = None) -> Conversation:
        return Conversation(
            id=model.id,
            user_id=str(model.user_id),
            title=model.title,
            document_ids=[uuid.UUID(d) for d in (model.document_ids or [])],
            messages=messages or [],
            metadata=model.conv_metadata or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
    @staticmethod
    def _msg_to_entity(model: MessageModel) -> Message:
        return Message(
            id=model.id,
            conversation_id=model.conversation_id,
            role=MessageRole(model.role),
            content=model.content,
            source_chunks=[uuid.UUID(c) for c in (model.source_chunks or [])],
            metadata=model.msg_metadata or {},
            created_at=model.created_at,
        )
    async def create(self, conversation: Conversation) -> Conversation:
        model = ConversationModel(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            document_ids=[str(d) for d in conversation.document_ids],
            conv_metadata=conversation.metadata,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._conv_to_entity(model)
    async def get_by_id(self, conversation_id: uuid.UUID) -> Conversation | None:
        result = await self._session.execute(
            select(ConversationModel).where(ConversationModel.id == conversation_id)
        )
        model = result.scalar_one_or_none()
        return self._conv_to_entity(model) if model else None
    async def get_by_user(
        self,
        user_id: str,
        *,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[Conversation], int]:
        query = (
            select(ConversationModel)
            .where(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        count_query = (
            select(func.count())
            .select_from(ConversationModel)
            .where(ConversationModel.user_id == user_id)
        )
        results = await self._session.execute(query)
        total = (await self._session.execute(count_query)).scalar_one()
        return [self._conv_to_entity(m) for m in results.scalars().all()], total
    async def delete(self, conversation_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            select(ConversationModel).where(ConversationModel.id == conversation_id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True
    async def add_message(self, message: Message) -> Message:
        model = MessageModel(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role.value,
            content=message.content,
            source_chunks=[str(c) for c in message.source_chunks],
            msg_metadata=message.metadata,
            created_at=message.created_at,
        )
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._msg_to_entity(model)
    async def get_messages(
        self,
        conversation_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 50,
    ) -> list[Message]:
        result = await self._session.execute(
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at)
            .offset(skip)
            .limit(limit)
        )
        return [self._msg_to_entity(m) for m in result.scalars().all()]
