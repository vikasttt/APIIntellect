"""User domain entity."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class User:
    """Core user domain entity (plain Python, no ORM dependency)."""

    def __init__(
        self,
        *,
        id: uuid.UUID | None = None,
        email: str,
        full_name: str,
        role: UserRole = UserRole.USER,
        status: UserStatus = UserStatus.ACTIVE,
        hashed_password: str | None = None,
        avatar_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        last_login_at: datetime | None = None,
    ) -> None:
        self.id: uuid.UUID = id or uuid.uuid4()
        self.email = email
        self.full_name = full_name
        self.role = role
        self.status = status
        self.hashed_password = hashed_password
        self.avatar_url = avatar_url
        self.metadata: dict[str, Any] = metadata or {}
        self.created_at: datetime = created_at or datetime.utcnow()
        self.updated_at: datetime = updated_at or datetime.utcnow()
        self.last_login_at = last_login_at

    # ── Business rules ─────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.status == UserStatus.ACTIVE

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN

    def deactivate(self) -> None:
        self.status = UserStatus.INACTIVE
        self.updated_at = datetime.utcnow()

    def suspend(self) -> None:
        self.status = UserStatus.SUSPENDED
        self.updated_at = datetime.utcnow()

    def activate(self) -> None:
        self.status = UserStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def record_login(self) -> None:
        self.last_login_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"
