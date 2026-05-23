"""Conversation and Chat API routes."""
from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.application.dto.conversation_dto import ChatMessageDTO, CreateConversationDTO
from app.application.use_cases.conversation import (
    ChatUseCase,
    CreateConversationUseCase,
    DeleteConversationUseCase,
    GetConversationsUseCase,
    GetMessagesUseCase,
)
from app.interfaces.api.dependencies.providers import (
    get_chat_use_case,
    get_create_conversation_use_case,
    get_conversations_use_case,
    get_delete_conversation_use_case,
    get_messages_use_case,
)
from app.interfaces.api.schemas.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
)
router = APIRouter(prefix="/conversations", tags=["Conversations"])
@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a conversation",
)
async def create_conversation(
    body: CreateConversationRequest,
    create_uc: CreateConversationUseCase = Depends(get_create_conversation_use_case),
) -> ConversationResponse:
    dto = CreateConversationDTO(
        user_id=body.user_id,
        title=body.title,
        document_ids=body.document_ids,
    )
    result = await create_uc.execute(dto)
    return ConversationResponse(**result.model_dump())

@router.get(
    "",
    response_model=ConversationListResponse,
    summary="Get conversations for a user",
)
async def list_conversations(
    user_id: str = Query(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    list_uc: GetConversationsUseCase = Depends(get_conversations_use_case),
) -> ConversationListResponse:
    result = await list_uc.execute(user_id, skip=skip, limit=limit)
    return ConversationListResponse(
        items=[ConversationResponse(**c.model_dump()) for c in result.items],
        total=result.total,
        skip=result.skip,
        limit=result.limit,
    )
@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    user_id: str = Query(...),
    delete_uc: DeleteConversationUseCase = Depends(get_delete_conversation_use_case),
) -> None:
    try:
        await delete_uc.execute(conversation_id, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
    summary="Get messages in a conversation",
)
async def get_messages(
    conversation_id: uuid.UUID,
    user_id: str = Query(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    messages_uc: GetMessagesUseCase = Depends(get_messages_use_case),
) -> list[MessageResponse]:
    try:
        msgs = await messages_uc.execute(
            conversation_id, user_id, skip=skip, limit=limit
        )
        return [MessageResponse(**m.model_dump()) for m in msgs]
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
@router.post(
    "/{conversation_id}/chat",
    response_model=ChatResponse,
    summary="Chat with documents (RAG)",
)
async def chat(
    conversation_id: uuid.UUID,
    body: ChatRequest,
    chat_uc: ChatUseCase = Depends(get_chat_use_case),
) -> ChatResponse:
    dto = ChatMessageDTO(
        conversation_id=conversation_id,
        user_id=body.user_id,
        content=body.content,
        document_ids=body.document_ids,
    )
    try:
        result = await chat_uc.execute(dto)
        return ChatResponse(
            message=MessageResponse(**result.message.model_dump()),
            sources=result.sources,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(exc)}",
        )
