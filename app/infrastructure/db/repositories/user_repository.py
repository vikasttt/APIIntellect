"""SQLAlchemy async implementation of IUserRepository."""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.user import User, UserRole, UserStatus
from app.domain.repositories.user_repository import IUserRepository
from app.infrastructure.db.models import UserModel


class SqlUserRepository(IUserRepository):
    """PostgreSQL-backed user repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Mapping ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_entity(model: UserModel) -> User:
        return User(
            id=model.id,
            email=model.email,
            full_name=model.full_name,
            hashed_password=model.hashed_password,
            role=UserRole(model.role),
            status=UserStatus(model.status),
            avatar_url=model.avatar_url,
            metadata=model.user_metadata or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_login_at=model.last_login_at,
        )

    @staticmethod
    def _to_model(entity: User) -> UserModel:
        return UserModel(
            id=entity.id,
            email=entity.email,
            full_name=entity.full_name,
            hashed_password=entity.hashed_password,
            role=entity.role.value,
            status=entity.status.value,
            avatar_url=entity.avatar_url,
            user_metadata=entity.metadata,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            last_login_at=entity.last_login_at,
        )

    # ── IUserRepository ────────────────────────────────────────────────────

    async def create(self, user: User) -> User:
        model = self._to_model(user)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._to_entity(model)

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(
            select(UserModel).where(UserModel.email == email.lower())
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def list_users(
        self,
        *,
        skip: int = 0,
        limit: int = 20,
        role: UserRole | None = None,
        status: UserStatus | None = None,
    ) -> tuple[list[User], int]:
        query = select(UserModel)
        count_query = select(func.count()).select_from(UserModel)

        if role is not None:
            query = query.where(UserModel.role == role.value)
            count_query = count_query.where(UserModel.role == role.value)
        if status is not None:
            query = query.where(UserModel.status == status.value)
            count_query = count_query.where(UserModel.status == status.value)

        query = query.order_by(UserModel.created_at.desc()).offset(skip).limit(limit)

        results = await self._session.execute(query)
        total = (await self._session.execute(count_query)).scalar_one()
        return [self._to_entity(m) for m in results.scalars().all()], total

    async def update(self, user: User) -> User:
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == user.id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            raise ValueError(f"User {user.id} not found")

        model.email = user.email.lower()
        model.full_name = user.full_name
        model.role = user.role.value
        model.status = user.status.value
        model.avatar_url = user.avatar_url
        model.user_metadata = user.metadata
        model.last_login_at = user.last_login_at

        await self._session.flush()
        return self._to_entity(model)

    async def delete(self, user_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    async def exists_by_email(self, email: str) -> bool:
        result = await self._session.execute(
            select(func.count())
            .select_from(UserModel)
            .where(UserModel.email == email.lower())
        )
        return result.scalar_one() > 0
