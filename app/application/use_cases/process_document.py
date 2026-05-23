"""Use case: Process a document (parse → chunk → embed → store)."""
from __future__ import annotations
import uuid
from typing import Protocol
from app.config.logging import get_logger
from app.domain.entities.document import DocumentStatus
from app.domain.repositories.chunk_repository import IChunkRepository
from app.domain.repositories.document_repository import IDocumentRepository
from app.domain.services.document_service import DocumentDomainService
logger = get_logger(__name__)
class IDocumentParser(Protocol):
    """Port: parse raw file bytes into structured text chunks."""
    async def parse(self, file_path: str, content_type: str) -> list[dict]:  # type: ignore[type-arg]
        ...
class IChunkingService(Protocol):
    """Port: split parsed content into Chunk domain entities."""
    async def chunk(
        self,
        parsed_sections: list[dict],  # type: ignore[type-arg]
        document_id: uuid.UUID,
        user_id: str,
    ) -> list:
        ...
class IEmbeddingService(Protocol):
    """Port: generate embeddings for chunks."""
    async def embed_chunks(self, chunks: list) -> list:
        ...
class ProcessDocumentUseCase:
    """Orchestrates the parse → chunk → embed → store pipeline."""
    def __init__(
        self,
        document_repo: IDocumentRepository,
        chunk_repo: IChunkRepository,
        parser: IDocumentParser,
        chunker: IChunkingService,
        embedder: IEmbeddingService,
        domain_service: DocumentDomainService,
    ) -> None:
        self._doc_repo = document_repo
        self._chunk_repo = chunk_repo
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._svc = domain_service
    async def execute(self, document_id: uuid.UUID) -> int:
        """Process a document; return number of chunks created."""
        document = await self._doc_repo.get_by_id(document_id)
        if document is None:
            raise ValueError(f"Document {document_id} not found")
        if not self._svc.can_be_processed(document):
            raise ValueError(
                f"Document {document_id} cannot be processed (status={document.status})"
            )
        logger.info("Starting document processing", document_id=str(document_id))
        await self._doc_repo.update_status(document_id, DocumentStatus.PROCESSING)
        try:
            # 1. Parse file → structured sections
            parsed_sections = await self._parser.parse(
                document.file_path, document.content_type
            )
            logger.info(
                "Document parsed",
                document_id=str(document_id),
                section_count=len(parsed_sections),
            )
            # 2. Chunk sections → Chunk entities
            chunks = await self._chunker.chunk(
                parsed_sections, document.id, document.user_id
            )
            logger.info(
                "Document chunked",
                document_id=str(document_id),
                chunk_count=len(chunks),
            )
            # 3. Delete old chunks if reprocessing
            await self._chunk_repo.delete_by_document(document_id)
            # 4. Embed chunks
            chunks_with_embeddings = await self._embedder.embed_chunks(chunks)
            # 5. Store chunks
            saved_chunks = await self._chunk_repo.bulk_create(chunks_with_embeddings)
            # 6. Update document status
            await self._doc_repo.update_status(
                document_id,
                DocumentStatus.PROCESSED,
                chunk_count=len(saved_chunks),
            )
            logger.info(
                "Document processing complete",
                document_id=str(document_id),
                chunk_count=len(saved_chunks),
            )
            return len(saved_chunks)
        except Exception as exc:
            logger.error(
                "Document processing failed",
                document_id=str(document_id),
                error=str(exc),
                exc_info=True,
            )
            await self._doc_repo.update_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(exc),
            )
            raise
