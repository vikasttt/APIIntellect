"""FastAPI application factory with lifespan management."""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config.logging import configure_logging, configure_telemetry, get_logger
from app.config.settings import Settings, get_settings
from app.interfaces.api.dependencies.container import build_container, set_container
from app.interfaces.api.routes.agent import router as agent_router
from app.interfaces.api.routes.conversations import router as conversations_router
from app.interfaces.api.routes.documents import router as documents_router
from app.interfaces.api.routes.search import router as search_router
from app.interfaces.api.routes.system import router as system_router
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown lifecycle."""
    settings: Settings = app.state.settings  # type: ignore[attr-defined]

    # ── Logging & telemetry ───────────────────────────────────────────────
    configure_logging(debug=settings.APP_DEBUG)
    configure_telemetry(
        connection_string=settings.APPLICATIONINSIGHTS_CONNECTION_STRING,
        service_name=settings.OTEL_SERVICE_NAME,
        service_version=settings.OTEL_SERVICE_VERSION,
    )

    # ── DI container ──────────────────────────────────────────────────────
    container = build_container(settings)
    set_container(container)
    logger.info(
        "Application starting",
        env=settings.APP_ENV,
        version=settings.OTEL_SERVICE_VERSION,
    )

    # ── LangGraph: init PostgreSQL-backed checkpointer ────────────────────
    from app.agent.graph import close_graph, init_graph
    from app.agent.rag_services import RagServices, set_rag_services
    from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddingService
    from app.infrastructure.llm.cohere_rerank import CohereRerankService
    from app.infrastructure.llm.multi_query import MultiQueryService
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    dsn = settings.get_checkpointer_dsn()
    await init_graph(dsn)
    logger.info("LangGraph initialised — DB host: %s", dsn.split("@")[-1])

    # ── Register RAG services so rag_node uses real vector search ─────────
    set_rag_services(
        RagServices(
            embedder=container.resolve(OpenAIEmbeddingService),
            multi_query=container.resolve(MultiQueryService),
            reranker=container.resolve(CohereRerankService),
            session_factory=container.resolve(async_sessionmaker[AsyncSession]),
            top_k=settings.AGENT_RAG_TOP_K,
            multi_query_count=settings.AGENT_RAG_MULTI_QUERY_COUNT,
            rerank_top_n=settings.AGENT_RAG_RERANK_TOP_N,
            rerank_enabled=settings.AGENT_RAG_RERANK_ENABLED,
        )
    )
    logger.info("RagServices registered for agent rag_node ✓")

    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────
    await close_graph()
    logger.info("Application shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory."""
    if settings is None:
        settings = get_settings()
    app = FastAPI(
        title="Smart Client — RAG + Agentic Chat API",
        description=(
            "Production-ready Retrieval-Augmented Generation backend with a "
            "LangGraph-powered agentic chat layer. Upload documents, search "
            "semantically, and interact via a dynamic chat widget."
        ),
        version=settings.OTEL_SERVICE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings  # type: ignore[attr-defined]

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.APP_ENV != "production" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handlers ─────────────────────────────────────────
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("Unhandled exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    # ── Routers ───────────────────────────────────────────────────────────
    api_prefix = "/api/v1"
    app.include_router(system_router, prefix=api_prefix)
    app.include_router(documents_router, prefix=api_prefix)
    app.include_router(search_router, prefix=api_prefix)
    app.include_router(conversations_router, prefix=api_prefix)
    app.include_router(agent_router, prefix=api_prefix)
    return app
