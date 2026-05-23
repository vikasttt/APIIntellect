"""Use cases: CRUD operations for documents."""
from __future__ import annotations
import uuid
from app.application.dto.document_dto import (
    DocumentListDTO,
    DocumentResponseDTO,
    UpdateDocumentDTO,
)
from app.config.logging import get_logger
from app.domain.entities.document import Document, DocumentStatus
from app.domain.repositories.document_repository import IDocumentRepository
logger = get_logger(__name__)
def _to_dto(doc: Document) -> DocumentResponseDTO:
    return DocumentResponseDTO(
        id=doc.id,
        user_id=doc.user_id,
        filename=doc.filename,
        original_filename=doc.original_filename,
        file_size=doc.file_size,
        content_type=doc.content_type,
        status=doc.status,
        title=doc.title,
        description=doc.description,
        metadata=doc.metadata,
        chunk_count=doc.chunk_count,
        error_message=doc.error_message,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        processed_at=doc.processed_at,
    )
class GetDocumentsUseCase:
    def __init__(self, document_repo: IDocumentRepository) -> None:
        self._repo = document_repo
    async def execute(
        self,
        user_id: str,
        *,
        skip: int = 0,
        limit: int = 20,
        status: DocumentStatus | None = None,
    ) -> DocumentListDTO:
        docs, total = await self._repo.get_by_user(
            user_id, skip=skip, limit=limit, status=status
        )
        return DocumentListDTO(
            items=[_to_dto(d) for d in docs],
            total=total,
            skip=skip,
            limit=limit,
        )
class GetDocumentUseCase:
    def __init__(self, document_repo: IDocumentRepository) -> None:
        self._repo = document_repo
    async def execute(self, document_id: uuid.UUID, user_id: str) -> DocumentResponseDTO:
        doc = await self._repo.get_by_id(document_id)
        if doc is None or doc.user_id != user_id:
            raise ValueError(f"Document {document_id} not found")
        return _to_dto(doc)
class UpdateDocumentUseCase:
    def __init__(self, document_repo: IDocumentRepository) -> None:
        self._repo = document_repo
    async def execute(
        self, document_id: uuid.UUID, user_id: str, dto: UpdateDocumentDTO
    ) -> DocumentResponseDTO:
        doc = await self._repo.get_by_id(document_id)
        if doc is None or doc.user_id != user_id:
            raise ValueError(f"Document {document_id} not found")
        if dto.title is not None:
            doc.title = dto.title
        if dto.description is not None:
            doc.description = dto.description
        if dto.metadata is not None:
            doc.metadata.update(dto.metadata)
        updated = await self._repo.update(doc)
        logger.info("Document updated", document_id=str(document_id))
        return _to_dto(updated)
class DeleteDocumentUseCase:
    def __init__(
        self,
        document_repo: IDocumentRepository,
    ) -> None:
        self._doc_repo = document_repo
    async def execute(self, document_id: uuid.UUID, user_id: str) -> None:
        doc = await self._doc_repo.get_by_id(document_id)
        if doc is None or doc.user_id != user_id:
            raise ValueError(f"Document {document_id} not found")
        deleted = await self._doc_repo.delete(document_id)
        if not deleted:
            raise ValueError(f"Could not delete document {document_id}")
        logger.info("Document deleted", document_id=str(document_id))
