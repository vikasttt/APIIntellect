"""SQLAlchemy ORM models.

Table order (FK dependency):
  1. users
  2. documents  → users.id
  3. chunks     → documents.id, users.id
  4. conversations → users.id
  5. messages   → conversations.id
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.db.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# 1. Users
# ─────────────────────────────────────────────────────────────────────────────

class UserModel(Base):
    """Persistent user account."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String(320), nullable=False, unique=True, index=True
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="user"
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active"
    )
    avatar_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    user_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships (back-references for convenience)
    documents: Mapped[list[DocumentModel]] = relationship(
        "DocumentModel",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="noload",
    )
    conversations: Mapped[list[ConversationModel]] = relationship(
        "ConversationModel",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="noload",
    )
    tenants: Mapped[list[TenantModel]] = relationship(
        "TenantModel",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    __table_args__ = (
        Index("ix_users_email", "email", unique=True),
        Index("ix_users_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<UserModel id={self.id} email={self.email} role={self.role}>"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Documents
# ─────────────────────────────────────────────────────────────────────────────

class DocumentModel(Base):
    """Persistent document metadata."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # FK → users.id (CASCADE so documents are removed when user is deleted)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="uploaded"
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped[UserModel] = relationship(
        "UserModel", back_populates="documents", lazy="noload"
    )
    chunks: Mapped[list[ChunkModel]] = relationship(
        "ChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="noload",
    )
    tenant: Mapped[TenantModel | None] = relationship(
        "TenantModel", back_populates="documents", lazy="noload"
    )

    __table_args__ = (
        Index("ix_documents_user_status", "user_id", "status"),
        Index("ix_documents_tenant_id", "tenant_id"),
        Index("ix_documents_created_at", "created_at"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chunks
# ─────────────────────────────────────────────────────────────────────────────

class ChunkModel(Base):
    """A single text chunk derived from a document."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalised FK for fast per-user vector search without joining documents
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(3072), nullable=True
    )
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="text"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[DocumentModel] = relationship(
        "DocumentModel", back_populates="chunks", lazy="noload"
    )

    __table_args__ = (
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
        Index("ix_chunks_user_id", "user_id"),
        # HNSW vector index and GIN FTS index created via Alembic migration
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conversations
# ─────────────────────────────────────────────────────────────────────────────

class ConversationModel(Base):
    """A conversation thread."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    document_ids: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    conv_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped[UserModel] = relationship(
        "UserModel", back_populates="conversations", lazy="noload"
    )
    messages: Mapped[list[MessageModel]] = relationship(
        "MessageModel",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="noload",
        order_by="MessageModel.created_at",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Messages
# ─────────────────────────────────────────────────────────────────────────────

class MessageModel(Base):
    """A single message in a conversation."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_chunks: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    msg_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped[ConversationModel] = relationship(
        "ConversationModel", back_populates="messages", lazy="noload"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tenants
# ─────────────────────────────────────────────────────────────────────────────

class TenantModel(Base):
    """A registered tenant."""
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_key: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company_name: Mapped[str] = mapped_column(String(512), nullable=False)
    company_info: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped[UserModel] = relationship(
        "UserModel", back_populates="tenants", lazy="noload"
    )
    api_specs: Mapped[list[ApiSpecModel]] = relationship(
        "ApiSpecModel",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="noload",
    )
    documents: Mapped[list[DocumentModel]] = relationship(
        "DocumentModel", back_populates="tenant", lazy="noload"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tenant API Specs
# ─────────────────────────────────────────────────────────────────────────────

class ApiSpecModel(Base):
    """A registered API endpoint for a tenant."""
    __tablename__ = "tenant_api_specs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False, default="GET")
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    headers: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    query_params: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    request_body: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    path_params: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    lookup_fields: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    tenant: Mapped[TenantModel] = relationship(
        "TenantModel", back_populates="api_specs", lazy="noload"
    )

