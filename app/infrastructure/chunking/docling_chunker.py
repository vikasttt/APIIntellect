"""Title-based chunking using LangChain's RecursiveCharacterTextSplitter."""
from __future__ import annotations
import uuid
from typing import Any
from app.config.logging import get_logger
from app.domain.entities.chunk import Chunk
logger = get_logger(__name__)
class DoclingChunkingService:
    """
    Splits docling-parsed sections into Chunk entities.
    Strategy:
    - Each section (split by title) is kept intact if small enough.
    - Large sections are further split using RecursiveCharacterTextSplitter.
    - Image description and table sections are kept as single chunks.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
    async def chunk(
        self,
        parsed_sections: list[dict[str, Any]],
        document_id: uuid.UUID,
        user_id: str,
    ) -> list[Chunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[Chunk] = []
        chunk_index = 0
        for section in parsed_sections:
            content: str = section.get("content", "").strip()
            section_title: str | None = section.get("section_title")
            page_number: int | None = section.get("page_number")
            chunk_type: str = section.get("chunk_type", "text")
            metadata: dict[str, Any] = section.get("metadata", {})
            if not content:
                continue
            # Non-text chunks are kept as single units
            if chunk_type in ("image_description", "table"):
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        user_id=user_id,
                        content=content,
                        chunk_index=chunk_index,
                        token_count=len(content.split()),
                        section_title=section_title,
                        page_number=page_number,
                        chunk_type=chunk_type,
                        metadata={**metadata, "source_section": section_title or ""},
                    )
                )
                chunk_index += 1
                continue
            # Text sections: split if necessary
            if len(content) <= self._chunk_size:
                sub_texts = [content]
            else:
                sub_texts = splitter.split_text(content)
            for sub in sub_texts:
                if sub.strip():
                    chunks.append(
                        Chunk(
                            document_id=document_id,
                            user_id=user_id,
                            content=sub.strip(),
                            chunk_index=chunk_index,
                            token_count=len(sub.split()),
                            section_title=section_title,
                            page_number=page_number,
                            chunk_type=chunk_type,
                            metadata={**metadata, "source_section": section_title or ""},
                        )
                    )
                    chunk_index += 1
        logger.info(
            "Chunking complete",
            document_id=str(document_id),
            chunk_count=len(chunks),
        )
        return chunks