"""Docling-based document parser with image description and title-based chunking."""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any
from app.config.logging import get_logger
logger = get_logger(__name__)
class DoclingParser:
    """
    Parse PDF and DOCX files using Docling.
    Returns a list of structured sections, each containing:
    - content: str
    - section_title: str | None
    - page_number: int | None
    - chunk_type: "text" | "image_description" | "table"
    - metadata: dict
    """
    def __init__(self, openai_api_key: str, vision_model: str = "gpt-4o") -> None:
        self._openai_api_key = openai_api_key
        self._vision_model = vision_model
    async def parse(self, file_path: str, content_type: str) -> list[dict[str, Any]]:
        """Async wrapper around synchronous Docling parse."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._parse_sync, file_path, content_type
        )
    def _parse_sync(self, file_path: str, content_type: str) -> list[dict[str, Any]]:
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError:
            raise RuntimeError("docling is not installed. Run: uv add docling")
        logger.info("Parsing document with Docling", file_path=file_path)
        path = Path(file_path)
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(path))
        doc = result.document
        sections: list[dict[str, Any]] = []
        current_title: str | None = None
        current_page: int | None = None
        for element, _level in doc.iterate_items():
            element_type = type(element).__name__
            if element_type == "SectionHeaderItem":
                current_title = str(element.text) if hasattr(element, "text") else None
                continue
            if element_type == "TextItem":
                text = str(element.text) if hasattr(element, "text") else ""
                page = (
                    element.prov[0].page_no
                    if hasattr(element, "prov") and element.prov
                    else current_page
                )
                if text.strip():
                    sections.append(
                        {
                            "content": text,
                            "section_title": current_title,
                            "page_number": page,
                            "chunk_type": "text",
                            "metadata": {"element_type": element_type},
                        }
                    )
                current_page = page
            elif element_type == "TableItem":
                try:
                    table_md = element.export_to_markdown()
                except Exception:
                    table_md = str(element)
                page = (
                    element.prov[0].page_no
                    if hasattr(element, "prov") and element.prov
                    else current_page
                )
                if table_md.strip():
                    sections.append(
                        {
                            "content": table_md,
                            "section_title": current_title,
                            "page_number": page,
                            "chunk_type": "table",
                            "metadata": {"element_type": "table"},
                        }
                    )
            elif element_type == "PictureItem":
                # Attempt vision description via OpenAI
                description = self._describe_image(element, current_title)
                if description:
                    page = (
                        element.prov[0].page_no
                        if hasattr(element, "prov") and element.prov
                        else current_page
                    )
                    sections.append(
                        {
                            "content": description,
                            "section_title": current_title,
                            "page_number": page,
                            "chunk_type": "image_description",
                            "metadata": {"element_type": "image"},
                        }
                    )
        logger.info("Docling parse complete", section_count=len(sections))
        return sections
    def _describe_image(self, element: Any, section_title: str | None) -> str | None:
        """Send image to OpenAI vision model for description."""
        try:
            import base64
            from openai import OpenAI
            # Export image bytes
            image_bytes: bytes | None = None
            if hasattr(element, "get_image"):
                img = element.get_image(scale=1.0)
                if img is not None:
                    import io
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
            if image_bytes is None:
                return None
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            client = OpenAI(api_key=self._openai_api_key)
            context = f"This image appears in the section: '{section_title}'." if section_title else ""
            response = client.chat.completions.create(
                model=self._vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"{context} Please describe this image in detail, "
                                    "including any text, charts, diagrams, or visual elements."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.warning("Image description failed", error=str(exc))
            return None
