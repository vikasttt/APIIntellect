"""Domain services pure logic, no external dependencies."""
from __future__ import annotations
import uuid
from typing import Any
from app.domain.entities.document import Document, DocumentStatus
class DocumentDomainService:
    """Business rules that span the Document entity."""
    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"pdf", "docx"})
    def is_extension_allowed(self, extension: str) -> bool:
        return extension.lower() in self.SUPPORTED_EXTENSIONS
    def can_be_processed(self, document: Document) -> bool:
        """A document can be processed only if it's in UPLOADED or FAILED state."""
        return document.status in (DocumentStatus.UPLOADED, DocumentStatus.FAILED)
    def can_be_reprocessed(self, document: Document) -> bool:
        """Any non-processing document can be reprocessed."""
        return document.status != DocumentStatus.PROCESSING
    def validate_file_size(self, size_bytes: int, max_bytes: int) -> None:
        if size_bytes > max_bytes:
            raise ValueError(
                f"File size {size_bytes} bytes exceeds limit of {max_bytes} bytes"
            )
