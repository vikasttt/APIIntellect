"""SQLAlchemy async implementation of ITenantRepository."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.entities.tenant import ApiSpec, Tenant
from app.domain.repositories.tenant_repository import ITenantRepository
from app.infrastructure.db.models import ApiSpecModel, TenantModel


class SqlTenantRepository(ITenantRepository):
    """Postgres-backed tenant and api spec repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ─── Mapping helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_entity(model: TenantModel) -> Tenant:
        return Tenant(
            id=model.id,
            tenant_key=model.tenant_key,
            user_id=model.user_id,
            company_name=model.company_name,
            company_info=model.company_info,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at,
            api_specs=[
                ApiSpec(
                    id=s.id,
                    tenant_id=s.tenant_id,
                    name=s.name,
                    description=s.description,
                    method=s.method,
                    url=s.url,
                    headers=s.headers,
                    query_params=s.query_params,
                    request_body=s.request_body,
                    path_params=s.path_params,
                    lookup_fields=s.lookup_fields,
                    is_active=s.is_active,
                    created_at=s.created_at,
                )
                for s in model.api_specs
            ]
            if model.api_specs is not None
            else [],
        )

    @staticmethod
    def _to_model(entity: Tenant) -> TenantModel:
        return TenantModel(
            id=entity.id,
            tenant_key=entity.tenant_key,
            user_id=entity.user_id,
            company_name=entity.company_name,
            company_info=entity.company_info,
            is_active=entity.is_active,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    # ─── ITenantRepository ──────────────────────────────────────────────────

    async def create(self, tenant: Tenant) -> Tenant:
        model = self._to_model(tenant)
        self._session.add(model)
        await self._session.flush()
        # Refresh to get DB-assigned fields if needed, but not strictly required
        return self._to_entity(model)

    async def get_by_key(self, tenant_key: str) -> Tenant | None:
        result = await self._session.execute(
            select(TenantModel)
            .options(selectinload(TenantModel.api_specs))
            .where(TenantModel.tenant_key == tenant_key)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        result = await self._session.execute(
            select(TenantModel)
            .options(selectinload(TenantModel.api_specs))
            .where(TenantModel.id == tenant_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_by_user(self, user_id: uuid.UUID) -> list[Tenant]:
        result = await self._session.execute(
            select(TenantModel)
            .options(selectinload(TenantModel.api_specs))
            .where(TenantModel.user_id == user_id)
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def update(self, tenant: Tenant) -> Tenant:
        result = await self._session.execute(
            select(TenantModel).where(TenantModel.id == tenant.id)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise ValueError(f"Tenant {tenant.id} not found")

        model.company_name = tenant.company_name
        model.company_info = tenant.company_info
        model.is_active = tenant.is_active
        model.updated_at = datetime.utcnow()
        await self._session.flush()
        return self._to_entity(model)

    async def deactivate(self, tenant_id: uuid.UUID) -> None:
        await self._session.execute(
            update(TenantModel)
            .where(TenantModel.id == tenant_id)
            .values(is_active=False, updated_at=datetime.utcnow())
        )
        await self._session.flush()

    async def delete(self, tenant_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            delete(TenantModel).where(TenantModel.id == tenant_id)
        )
        await self._session.flush()
        return result.rowcount > 0

    # ─── API Spec CRUD ──────────────────────────────────────────────────────

    async def upsert_api_specs(
        self, tenant_id: uuid.UUID, specs: list[ApiSpec]
    ) -> list[ApiSpec]:
        # Delete existing
        await self._session.execute(
            delete(ApiSpecModel).where(ApiSpecModel.tenant_id == tenant_id)
        )
        
        if not specs:
            return []

        # Insert new
        models = [
            ApiSpecModel(
                id=s.id,
                tenant_id=tenant_id,
                name=s.name,
                description=s.description,
                method=s.method,
                url=s.url,
                headers=s.headers,
                query_params=s.query_params,
                request_body=s.request_body,
                path_params=s.path_params,
                lookup_fields=s.lookup_fields,
                is_active=s.is_active,
                created_at=s.created_at,
            )
            for s in specs
        ]
        self._session.add_all(models)
        await self._session.flush()
        
        # Return updated specs
        return [
            ApiSpec(
                id=m.id,
                tenant_id=m.tenant_id,
                name=m.name,
                description=m.description,
                method=m.method,
                url=m.url,
                headers=m.headers,
                query_params=m.query_params,
                request_body=m.request_body,
                path_params=m.path_params,
                lookup_fields=m.lookup_fields,
                is_active=m.is_active,
                created_at=m.created_at,
            )
            for m in models
        ]

    async def get_api_specs(self, tenant_id: uuid.UUID) -> list[ApiSpec]:
        result = await self._session.execute(
            select(ApiSpecModel).where(ApiSpecModel.tenant_id == tenant_id)
        )
        models = result.scalars().all()
        return [
            ApiSpec(
                id=m.id,
                tenant_id=m.tenant_id,
                name=m.name,
                description=m.description,
                method=m.method,
                url=m.url,
                headers=m.headers,
                query_params=m.query_params,
                request_body=m.request_body,
                path_params=m.path_params,
                lookup_fields=m.lookup_fields,
                is_active=m.is_active,
                created_at=m.created_at,
            )
            for m in models
        ]
