"""SQLAlchemy async engine and session factory."""
from __future__ import annotations
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.config.settings import Settings
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass
def create_engine(settings: Settings) -> AsyncEngine:
    """Create and configure the async SQLAlchemy engine."""
    return create_async_engine(
        settings.DATABASE_URL,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_timeout=settings.DATABASE_POOL_TIMEOUT,
        echo=settings.DATABASE_ECHO,
        pool_pre_ping=True,
    )
def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the given engine."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
