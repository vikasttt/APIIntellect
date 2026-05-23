"""
State definition for the Smart API Agent system.

Uses TypedDict + Annotated reducers so concurrent branches merge safely.
company_name / company_info are kept here so each tenant's session carries
its own identity without hitting a shared registry on every node.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence

from langchain_core.messages import BaseMessage


# ── Workflow-level state ─────────────────────────────────────────────────────

class AgentState(dict):
    """
    Central state object threaded through every LangGraph node.
    All list / message fields use operator.add so parallel branches merge safely.
    """

    # ── Conversation messages (append-only reducer) ──────────────────────────
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── Tenant context ───────────────────────────────────────────────────────
    company_name: str                   # e.g. "Acme Corp"
    company_info: str                   # Free-text RAG knowledge base
    api_specs: List[Dict[str, Any]]     # User-provided API spec list

    # ── Routing / intent ─────────────────────────────────────────────────────
    current_intent: Literal["rag", "api"]

    # ── API orchestration ────────────────────────────────────────────────────
    api_type: Literal[
        "get", "post", "delete", "put", "patch", "complex_coder", "unknown"
    ]
    target_api_name: str                # Which spec to call
    missing_information: List[str]      # Fields still needed from user
    collected_fields: Dict[str, Any]    # Fields gathered via HITL so far
    lookup_needed: List[Dict[str, Any]] # Pending name → id resolution tasks

    # ── Supervisor / validation ───────────────────────────────────────────────
    validation_comments: str
    supervisor_decision: Literal[
        "get_agent", "create_agent", "delete_agent", "coder_agent", "end"
    ]
    retry_count: int
    error_context: Optional[str]        # Last error message for coder agent

    # ── Guards ───────────────────────────────────────────────────────────────
    guardrail_blocked: bool

    # ── Session isolation ─────────────────────────────────────────────────────
    conversation_id: str                # Used as LangGraph thread_id
