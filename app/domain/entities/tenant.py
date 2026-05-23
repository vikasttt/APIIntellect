"""Domain entity: Tenant and ApiSpec."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ApiSpec:
    """A single API endpoint spec owned by a tenant."""

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    tenant_id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str = ""
    description: str = ""
    method: str = "GET"                 # GET | POST | PUT | PATCH | DELETE
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    request_body: Dict[str, str] = field(default_factory=dict)
    path_params: List[str] = field(default_factory=list)
    lookup_fields: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the flat dict expected by the LangGraph agent tools."""
        return {
            "name":          self.name,
            "description":   self.description,
            "method":        self.method,
            "url":           self.url,
            "headers":       self.headers,
            "query_params":  self.query_params,
            "request_body":  self.request_body,
            "path_params":   self.path_params,
            "lookup_fields": self.lookup_fields,
        }


@dataclass
class Tenant:
    """A registered tenant (company) that owns API specs and documents."""

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    tenant_key: str = ""          # short slug used in API calls, e.g. "acme-corp"
    user_id: uuid.UUID = field(default_factory=uuid.uuid4)   # FK → users.id
    company_name: str = ""
    company_info: str = ""        # free-text RAG knowledge base
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Eagerly loaded when needed (not always populated)
    api_specs: List[ApiSpec] = field(default_factory=list)
