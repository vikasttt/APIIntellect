"""SQLAlchemy async implementation of IDocumentRepository."""
from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.entities.document import Document, DocumentStatus
from app.domain.repositories.document_repository import IDocumentRepository
from app.infrastructure.db.models import DocumentModel
class SqlDocumentRepository(IDocumentRepository):
    """Postgres-backed document repository."""
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
    # ─── Mapping helpers ────────────────────────────────────────────────────
    @staticmethod
    def _to_entity(model: DocumentModel) -> Document:
        return Document(
            id=model.id,
            user_id=str(model.user_id),
            filename=model.filename,
            original_filename=model.original_filename,
            file_path=model.file_path,
            file_size=model.file_size,
            content_type=model.content_type,
            status=DocumentStatus(model.status),
            title=model.title,
            description=model.description,
            metadata=model.doc_metadata or {},
            chunk_count=model.chunk_count,
            error_message=model.error_message,
            created_at=model.created_at,
            updated_at=model.updated_at,
            processed_at=model.processed_at,
        )
    @staticmethod
    def _to_model(entity: Document) -> DocumentModel:
        return DocumentModel(
            id=entity.id,
            user_id=entity.user_id,
            filename=entity.filename,
            original_filename=entity.original_filename,
            file_path=entity.file_path,
            file_size=entity.file_size,
            content_type=entity.content_type,
            status=entity.status.value,
            title=entity.title,
            description=entity.description,
            doc_metadata=entity.metadata,
            chunk_count=entity.chunk_count,
            error_message=entity.error_message,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            processed_at=entity.processed_at,
        )
    # ─── IDocumentRepository ────────────────────────────────────────────────
    async def create(self, document: Document) -> Document:
        model = self._to_model(document)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._to_entity(model)
    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None
    async def get_by_user(
        self,
        user_id: str,
        *,
        skip: int = 0,
        limit: int = 20,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        query = select(DocumentModel).where(DocumentModel.user_id == user_id)
        count_query = select(func.count()).select_from(DocumentModel).where(
            DocumentModel.user_id == user_id
        )
        if status is not None:
            query = query.where(DocumentModel.status == status.value)
            count_query = count_query.where(DocumentModel.status == status.value)
        query = query.order_by(DocumentModel.created_at.desc()).offset(skip).limit(limit)
        results = await self._session.execute(query)
        total_result = await self._session.execute(count_query)
        models = results.scalars().all()
        total = total_result.scalar_one()
        return [self._to_entity(m) for m in models], total
    async def update(self, document: Document) -> Document:
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.id == document.id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            raise ValueError(f"Document {document.id} not found")
        model.title = document.title
        model.description = document.description
        model.doc_metadata = document.metadata
        model.status = document.status.value
        model.chunk_count = document.chunk_count
        model.error_message = document.error_message
        model.updated_at = datetime.utcnow()
        model.processed_at = document.processed_at
        await self._session.flush()
        return self._to_entity(model)
    async def delete(self, document_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True
    async def update_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        values: dict = {
            "status": status.value,
            "updated_at": datetime.utcnow(),
        }
        if error_message is not None:
            values["error_message"] = error_message
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if status == DocumentStatus.PROCESSED:
            values["processed_at"] = datetime.utcnow()
        await self._session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(**values)
        )
        await self._session.flush()