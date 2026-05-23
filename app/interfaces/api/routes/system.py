"""Health and Config system endpoints."""
from __future__ import annotations
from fastapi import APIRouter, Depends, Request
from app.interfaces.api.dependencies.providers import get_settings
from app.interfaces.api.schemas.schemas import ConfigResponse, HealthResponse
from app.config.settings import Settings
router = APIRouter(tags=["System"])
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check(request: Request) -> HealthResponse:
    settings = get_settings(request)
    return HealthResponse(
        status="ok",
        version=settings.OTEL_SERVICE_VERSION,
        environment=settings.APP_ENV,
    )
@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Get supported file types and system limits",
)
async def get_config(request: Request) -> ConfigResponse:
    settings = get_settings(request)
    return ConfigResponse(
        supported_file_types=settings.ALLOWED_EXTENSIONS,
        max_file_size_mb=settings.MAX_FILE_SIZE_MB,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
        chat_model=settings.OPENAI_CHAT_MODEL,
        rerank_enabled=settings.RERANK_ENABLED,
    )
