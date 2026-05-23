"""Abstract user repository interface."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from app.domain.entities.user import User, UserRole, UserStatus


class IUserRepository(ABC):
    """Port: user persistence operations."""

    @abstractmethod
    async def create(self, user: User) -> User:
        """Persist a new user and return it with DB-assigned fields."""
        ...

    @abstractmethod
    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        """Retrieve a user by UUID, or None."""
        ...

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        """Retrieve a user by email address, or None."""
        ...

    @abstractmethod
    async def list_users(
        self,
        *,
        skip: int = 0,
        limit: int = 20,
        role: UserRole | None = None,
        status: UserStatus | None = None,
    ) -> tuple[list[User], int]:
        """Return paginated users plus total count."""
        ...

    @abstractmethod
    async def update(self, user: User) -> User:
        """Persist changes to an existing user."""
        ...

    @abstractmethod
    async def delete(self, user_id: uuid.UUID) -> bool:
        """Hard-delete a user; return True if it existed."""
        ...

    @abstractmethod
    async def exists_by_email(self, email: str) -> bool:
        """Return True if a user with the given email already exists."""
        ...
