"""Document API routes."""
from __future__ import annotations
import os
import uuid
from pathlib import Path
import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status, Query
from app.application.dto.document_dto import UpdateDocumentDTO, UploadDocumentDTO
from app.application.use_cases.document_crud import (
    DeleteDocumentUseCase,
    GetDocumentUseCase,
    GetDocumentsUseCase,
    UpdateDocumentUseCase,
)
from app.application.use_cases.process_document import ProcessDocumentUseCase
from app.application.use_cases.upload_document import UploadDocumentUseCase
from app.domain.entities.document import DocumentStatus
from app.infrastructure.db.repositories.chunk_repository import SqlChunkRepository
from app.interfaces.api.dependencies.providers import (
    get_chunk_repo,
    get_delete_use_case,
    get_document_use_case,
    get_documents_use_case,
    get_process_use_case,
    get_settings,
    get_update_use_case,
    get_upload_use_case,
)
from app.interfaces.api.schemas.schemas import (
    ChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdateRequest,
    ProcessDocumentResponse,
)
from app.config.logging import get_logger
from fastapi import Request
logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])
# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    title: str | None = Form(None),
    description: str | None = Form(None),
    auto_process: bool = Form(False),
    upload_uc: UploadDocumentUseCase = Depends(get_upload_use_case),
    process_uc: ProcessDocumentUseCase = Depends(get_process_use_case),
) -> DocumentResponse:
    settings = get_settings(request)
    # Validate file size
    if file.size and file.size > settings.storage.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
        )
    # Validate extension
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' not supported. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )
    # Save file to disk
    upload_dir = Path(settings.UPLOAD_DIR) / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = upload_dir / safe_name
    content = await file.read()
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)
    dto = UploadDocumentDTO(
        user_id=user_id,
        original_filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        file_size=len(content),
        file_path=str(file_path),
        title=title,
        description=description,
    )
    result = await upload_uc.execute(dto)
    if auto_process:
        background_tasks.add_task(_process_in_background, process_uc, result.id)
    return DocumentResponse(**result.model_dump())

async def _process_in_background(
    process_uc: ProcessDocumentUseCase, document_id: uuid.UUID
) -> None:
    try:
        await process_uc.execute(document_id)
    except Exception as exc:
        logger.error("Background processing failed", document_id=str(document_id), error=str(exc))
# ── Process ────────────────────────────────────────────────────────────────────
@router.post(
    "/{document_id}/process",
    response_model=ProcessDocumentResponse,
    summary="Process document (parse → chunk → embed)",
)
async def process_document(
    document_id: uuid.UUID,
    process_uc: ProcessDocumentUseCase = Depends(get_process_use_case),
) -> ProcessDocumentResponse:
    try:
        chunk_count = await process_uc.execute(document_id)
        return ProcessDocumentResponse(
            document_id=document_id,
            chunk_count=chunk_count,
            message="Document processed successfully",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
# ── Reprocess ──────────────────────────────────────────────────────────────────
@router.post(
    "/{document_id}/reprocess",
    response_model=ProcessDocumentResponse,
    summary="Force re-chunk and re-embed a document",
)
async def reprocess_document(
    document_id: uuid.UUID,
    process_uc: ProcessDocumentUseCase = Depends(get_process_use_case),
) -> ProcessDocumentResponse:
    try:
        chunk_count = await process_uc.execute(document_id)
        return ProcessDocumentResponse(
            document_id=document_id,
            chunk_count=chunk_count,
            message="Document reprocessed successfully",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
# ── List ───────────────────────────────────────────────────────────────────────
@router.get(
    "",
    response_model=DocumentListResponse,
    summary="Get all documents (paginated)",
)
async def list_documents(
    user_id: str = Query(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status_filter: DocumentStatus | None = Query(None, alias="status"),
    list_uc: GetDocumentsUseCase = Depends(get_documents_use_case),
) -> DocumentListResponse:
    result = await list_uc.execute(user_id, skip=skip, limit=limit, status=status_filter)
    return DocumentListResponse(
        items=[DocumentResponse(**d.model_dump()) for d in result.items],
        total=result.total,
        skip=result.skip,
        limit=result.limit,
    )
# ── Get single ─────────────────────────────────────────────────────────────────
@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get single document",
)
async def get_document(
    document_id: uuid.UUID,
    user_id: str = Query(...),
    get_uc: GetDocumentUseCase = Depends(get_document_use_case),
) -> DocumentResponse:
    try:
        result = await get_uc.execute(document_id, user_id)
        return DocumentResponse(**result.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
# ── Update ─────────────────────────────────────────────────────────────────────
@router.put(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document metadata",
)
async def update_document(
    document_id: uuid.UUID,
    body: DocumentUpdateRequest,
    user_id: str = Query(...),
    update_uc: UpdateDocumentUseCase = Depends(get_update_use_case),
) -> DocumentResponse:
    try:
        dto = UpdateDocumentDTO(
            title=body.title,
            description=body.description,
            metadata=body.metadata,
        )
        result = await update_uc.execute(document_id, user_id, dto)
        return DocumentResponse(**result.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
# ── Delete ─────────────────────────────────────────────────────────────────────
@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document (and its chunks)",
)
async def delete_document(
    document_id: uuid.UUID,
    user_id: str = Query(...),
    delete_uc: DeleteDocumentUseCase = Depends(get_delete_use_case),
) -> None:
    try:
        await delete_uc.execute(document_id, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
# ── Chunks ─────────────────────────────────────────────────────────────────────
@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkResponse],
    summary="Get chunks of a document",
)
async def get_document_chunks(
    document_id: uuid.UUID,
    chunk_repo: SqlChunkRepository = Depends(get_chunk_repo),
) -> list[ChunkResponse]:
    chunks = await chunk_repo.get_by_document(document_id)
    return [
        ChunkResponse(
            id=c.id,
            document_id=c.document_id,
            content=c.content,
            chunk_index=c.chunk_index,
            token_count=c.token_count,
            page_number=c.page_number,
            section_title=c.section_title,
            chunk_type=c.chunk_type,
            metadata=c.metadata,
            created_at=c.created_at,
        )
        for c in chunks
    ]
