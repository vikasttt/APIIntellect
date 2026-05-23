"""Use cases: conversation management and chat."""
from __future__ import annotations
import uuid
from typing import Protocol
from app.application.dto.conversation_dto import (
    ChatMessageDTO,
    ChatResponseDTO,
    ConversationListDTO,
    ConversationResponseDTO,
    CreateConversationDTO,
    MessageResponseDTO,
)
from app.config.logging import get_logger
from app.domain.entities.conversation import Conversation, Message, MessageRole
from app.domain.repositories.conversation_repository import IConversationRepository
from app.domain.repositories.chunk_repository import IChunkRepository
logger = get_logger(__name__)
class ILLMPort(Protocol):
    async def chat(self, system_prompt: str, messages: list[dict]) -> str: ...  # type: ignore[type-arg]
class ISearchPort(Protocol):
    async def search(self, query: str, user_id: str, document_ids: list[uuid.UUID]) -> list: ...
def _conv_to_dto(conv: Conversation) -> ConversationResponseDTO:
    return ConversationResponseDTO(
        id=conv.id,
        user_id=conv.user_id,
        title=conv.title,
        document_ids=conv.document_ids,
        message_count=len(conv.messages),
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )
def _msg_to_dto(msg: Message) -> MessageResponseDTO:
    return MessageResponseDTO(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        source_chunks=msg.source_chunks,
        metadata=msg.metadata,
        created_at=msg.created_at,
    )
class CreateConversationUseCase:
    def __init__(self, repo: IConversationRepository) -> None:
        self._repo = repo
    async def execute(self, dto: CreateConversationDTO) -> ConversationResponseDTO:
        conv = Conversation(
            user_id=dto.user_id,
            title=dto.title,
            document_ids=dto.document_ids,
        )
        saved = await self._repo.create(conv)
        return _conv_to_dto(saved)
class GetConversationsUseCase:
    def __init__(self, repo: IConversationRepository) -> None:
        self._repo = repo
    async def execute(
        self, user_id: str, skip: int = 0, limit: int = 20
    ) -> ConversationListDTO:
        convs, total = await self._repo.get_by_user(user_id, skip=skip, limit=limit)
        return ConversationListDTO(
            items=[_conv_to_dto(c) for c in convs],
            total=total,
            skip=skip,
            limit=limit,
        )
class DeleteConversationUseCase:
    def __init__(self, repo: IConversationRepository) -> None:
        self._repo = repo
    async def execute(self, conversation_id: uuid.UUID, user_id: str) -> None:
        conv = await self._repo.get_by_id(conversation_id)
        if conv is None or conv.user_id != user_id:
            raise ValueError(f"Conversation {conversation_id} not found")
        await self._repo.delete(conversation_id)
class GetMessagesUseCase:
    def __init__(self, repo: IConversationRepository) -> None:
        self._repo = repo
    async def execute(
        self, conversation_id: uuid.UUID, user_id: str, skip: int = 0, limit: int = 50
    ) -> list[MessageResponseDTO]:
        conv = await self._repo.get_by_id(conversation_id)
        if conv is None or conv.user_id != user_id:
            raise ValueError(f"Conversation {conversation_id} not found")
        messages = await self._repo.get_messages(
            conversation_id, skip=skip, limit=limit
        )
        return [_msg_to_dto(m) for m in messages]
SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Context:
{context}
"""
class ChatUseCase:
    """RAG chat: retrieve context → build prompt → LLM → persist."""
    def __init__(
        self,
        conversation_repo: IConversationRepository,
        llm: ILLMPort,
        search: ISearchPort,
    ) -> None:
        self._conv_repo = conversation_repo
        self._llm = llm
        self._search = search
    async def execute(self, dto: ChatMessageDTO) -> ChatResponseDTO:
        conv = await self._conv_repo.get_by_id(dto.conversation_id)
        if conv is None or conv.user_id != dto.user_id:
            raise ValueError(f"Conversation {dto.conversation_id} not found")
        # 1. Retrieve relevant chunks
        doc_ids = dto.document_ids or conv.document_ids
        search_results = await self._search.search(dto.content, dto.user_id, doc_ids)
        context_parts = [r.chunk.content for r in search_results]
        context = "\n\n---\n\n".join(context_parts)
        source_chunk_ids = [r.chunk.id for r in search_results]
        # 2. Build chat history for LLM
        history_msgs = await self._conv_repo.get_messages(
            dto.conversation_id, skip=0, limit=20
        )
        lc_messages = [
            {"role": m.role.value, "content": m.content} for m in history_msgs
        ]
        lc_messages.append({"role": "user", "content": dto.content})
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        # 3. Call LLM
        answer = await self._llm.chat(system_prompt, lc_messages)
        # 4. Persist user message
        user_msg = Message(
            conversation_id=dto.conversation_id,
            role=MessageRole.USER,
            content=dto.content,
        )
        await self._conv_repo.add_message(user_msg)
        # 5. Persist assistant message
        assistant_msg = Message(
            conversation_id=dto.conversation_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            source_chunks=source_chunk_ids,
        )
        saved_assistant = await self._conv_repo.add_message(assistant_msg)
        sources = [
            {
                "chunk_id": str(r.chunk.id),
                "document_id": str(r.chunk.document_id),
                "section_title": r.chunk.section_title,
                "score": r.score,
                "content_snippet": r.chunk.content[:200],
            }
            for r in search_results
        ]
        return ChatResponseDTO(
            message=_msg_to_dto(saved_assistant),
            sources=sources,
        )
