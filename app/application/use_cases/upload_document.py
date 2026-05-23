"""Use case: Upload a document."""
from __future__ import annotations
import uuid
from typing import Protocol
from app.application.dto.document_dto import DocumentResponseDTO, UploadDocumentDTO
from app.config.logging import get_logger
from app.domain.entities.document import Document
from app.domain.repositories.document_repository import IDocumentRepository
from app.domain.services.document_service import DocumentDomainService
logger = get_logger(__name__)
class UploadDocumentUseCase:
    """Validates and persists a newly uploaded document."""
    def __init__(
        self,
        document_repo: IDocumentRepository,
        domain_service: DocumentDomainService,
    ) -> None:
        self._repo = document_repo
        self._svc = domain_service
    async def execute(self, dto: UploadDocumentDTO) -> DocumentResponseDTO:
        logger.info("Uploading document", filename=dto.original_filename, user=dto.user_id)
        if not self._svc.is_extension_allowed(dto.original_filename.rsplit(".", 1)[-1]):
            raise ValueError(
                f"File type not supported: {dto.original_filename.rsplit('.', 1)[-1]}"
            )
        document = Document(
            user_id=dto.user_id,
            filename=f"{uuid.uuid4()}_{dto.original_filename}",
            original_filename=dto.original_filename,
            file_path=dto.file_path,
            file_size=dto.file_size,
            content_type=dto.content_type,
            title=dto.title,
            description=dto.description,
            metadata=dto.metadata,
        )
        saved = await self._repo.create(document)
        logger.info("Document created", document_id=str(saved.id))
        return _to_dto(saved)
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
