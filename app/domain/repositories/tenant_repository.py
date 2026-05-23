"""Repository interface for Tenant aggregate."""
from __future__ import annotations

import uuid
from typing import Protocol

from app.domain.entities.tenant import ApiSpec, Tenant


class ITenantRepository(Protocol):
    """Port: persistence operations for the Tenant aggregate."""

    # ── Tenant CRUD ──────────────────────────────────────────────────────────

    async def create(self, tenant: Tenant) -> Tenant: ...

    async def get_by_key(self, tenant_key: str) -> Tenant | None:
        """Fetch tenant + api_specs by the short slug key."""
        ...

    async def get_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        """Fetch tenant + api_specs by primary key."""
        ...

    async def get_by_user(self, user_id: uuid.UUID) -> list[Tenant]: ...

    async def update(self, tenant: Tenant) -> Tenant: ...

    async def deactivate(self, tenant_id: uuid.UUID) -> None: ...

    async def delete(self, tenant_id: uuid.UUID) -> bool: ...

    # ── API Spec CRUD ────────────────────────────────────────────────────────

    async def upsert_api_specs(
        self, tenant_id: uuid.UUID, specs: list[ApiSpec]
    ) -> list[ApiSpec]:
        """Replace all API specs for a tenant (delete-and-insert)."""
        ...

    async def get_api_specs(self, tenant_id: uuid.UUID) -> list[ApiSpec]: ...
