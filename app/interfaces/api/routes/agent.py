"""
FastAPI routes for the Smart API Agent.

Endpoints:
  POST   /api/v1/agent/register          — Register a tenant (company info + API specs)
  DELETE /api/v1/agent/register/{id}     — Deregister a tenant
  GET    /api/v1/agent/tenants           — List registered tenant IDs
  POST   /api/v1/agent/chat/{session_id} — Send a user message (returns SSE stream)
  POST   /api/v1/agent/resume/{session_id} — Resume after human-in-the-loop pause

CORS is handled globally in app.py — all origins allowed in non-production.

SSE event types emitted by /chat and /resume:
  {"type": "token",              "content": "..."}   — streaming token
  {"type": "node_start",         "node": "..."}       — which agent is active
  {"type": "clarification_needed","content": "..."}   — HITL question for user
  {"type": "done"}                                     — stream finished
  {"type": "error",              "detail": "..."}     — fatal error
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent.graph import get_graph
from app.domain.entities.tenant import ApiSpec, Tenant
from app.domain.repositories.tenant_repository import ITenantRepository
from app.interfaces.api.dependencies.providers import get_tenant_repo
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

# ── Nodes whose activation is reported to the frontend ───────────────────────
_REPORTABLE_NODES = {
    "guardrail", "decider", "enhancer", "supervisor",
    "get_agent", "create_agent", "delete_agent", "coder_agent", "validatory",
}


# ════════════════════════════════════════════════════════════════════════════
#  Request / Response models
# ════════════════════════════════════════════════════════════════════════════

class ApiSpecModel(BaseModel):
    """Shape of a single API endpoint spec provided by the tenant."""
    name:         str
    description:  str
    method:       str                         # GET | POST | PUT | PATCH | DELETE
    url:          str
    headers:      Dict[str, str]   = Field(default_factory=dict)
    query_params: Dict[str, str]   = Field(default_factory=dict)
    request_body: Dict[str, str]   = Field(default_factory=dict)
    path_params:  List[str]        = Field(default_factory=list)
    lookup_fields: Dict[str, Any]  = Field(default_factory=dict)


class RegisterRequest(BaseModel):
    user_id:      uuid.UUID                   # Required to link to a user
    tenant_key:   str                         # Slug identifier, e.g. "acme"
    company_name: str
    company_info: str                         # Free-text product / company knowledge
    api_specs:    List[ApiSpecModel]


class RegisterResponse(BaseModel):
    status:    str
    tenant_id: str
    spec_count: int
    script_tag: str                           # Ready-to-embed <script> tag


class ChatRequest(BaseModel):
    tenant_id: str
    message:   str


class ResumeRequest(BaseModel):
    tenant_id: str
    message:   str                            # User's reply to the clarification question


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════


def _build_initial_state(
    tenant: Tenant,
    message: str,
    session_id: str,
) -> Dict[str, Any]:
    return {
        "messages":            [HumanMessage(content=message)],
        "company_name":        tenant.company_name,
        "company_info":        tenant.company_info,
        "api_specs":           [s.to_dict() for s in tenant.api_specs],
        "current_intent":      "api",
        "api_type":            "unknown",
        "target_api_name":     "",
        "missing_information": [],
        "collected_fields":    {},
        "lookup_needed":       [],
        "validation_comments": "",
        "supervisor_decision": "end",
        "retry_count":         0,
        "guardrail_blocked":   False,
        "error_context":       None,
        "conversation_id":     session_id,
    }


async def _stream_graph(
    session_id: str,
    initial_state: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Run the graph from the beginning and yield SSE-formatted events."""
    graph  = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")
            data = event.get("data", {})

            # Token-by-token streaming
            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

            # Node-level progress events
            elif kind == "on_chain_start" and name in _REPORTABLE_NODES:
                yield f"data: {json.dumps({'type': 'node_start', 'node': name})}\n\n"

            # HITL: graph paused to ask the user for information
            elif kind == "on_chain_end" and name == "human_clarification":
                output = data.get("output", {})
                msgs   = output.get("messages", [])
                if msgs:
                    content = (
                        msgs[-1].content
                        if hasattr(msgs[-1], "content")
                        else str(msgs[-1])
                    )
                    yield (
                        f"data: {json.dumps({'type': 'clarification_needed', 'content': content})}\n\n"
                    )

    except Exception as exc:
        logger.exception("Graph execution error for session %s", session_id)
        yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _resume_stream(session_id: str) -> AsyncGenerator[str, None]:
    """Continue graph execution after a human-in-the-loop pause."""
    graph  = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        async for event in graph.astream_events(None, config=config, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")
            data = event.get("data", {})

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

            elif kind == "on_chain_start" and name in _REPORTABLE_NODES | {"human_clarification"}:
                yield f"data: {json.dumps({'type': 'node_start', 'node': name})}\n\n"

            elif kind == "on_chain_end" and name == "human_clarification":
                output = data.get("output", {})
                msgs   = output.get("messages", [])
                if msgs:
                    content = (
                        msgs[-1].content
                        if hasattr(msgs[-1], "content")
                        else str(msgs[-1])
                    )
                    yield (
                        f"data: {json.dumps({'type': 'clarification_needed', 'content': content})}\n\n"
                    )

    except Exception as exc:
        logger.exception("Graph resume error for session %s", session_id)
        yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


_SSE_HEADERS = {
    "Cache-Control":   "no-cache",
    "X-Accel-Buffering": "no",
}


# ════════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════════

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a tenant with API specs and company knowledge",
)
async def register_tenant(
    body: RegisterRequest,
    repo: ITenantRepository = Depends(get_tenant_repo)
) -> RegisterResponse:
    """
    Accepts the company/product knowledge base (free text) and a list of API specs.
    Returns a ready-to-embed <script> tag that clients can drop into any webpage.

    The script tag embeds the chat widget and wires it to this server + tenant.
    """
    # Create the tenant
    tenant = Tenant(
        tenant_key=body.tenant_key,
        user_id=body.user_id,
        company_name=body.company_name,
        company_info=body.company_info,
        is_active=True,
    )
    tenant = await repo.create(tenant)
    
    # Add the API specs
    specs = [
        ApiSpec(
            tenant_id=tenant.id,
            name=s.name,
            description=s.description,
            method=s.method,
            url=s.url,
            headers=s.headers,
            query_params=s.query_params,
            request_body=s.request_body,
            path_params=s.path_params,
            lookup_fields=s.lookup_fields,
            is_active=True,
        )
        for s in body.api_specs
    ]
    await repo.upsert_api_specs(tenant.id, specs)

    logger.info(
        "Registered tenant '%s' with %d API spec(s)", tenant.id, len(specs)
    )

    # Build the embeddable script tag
    # The frontend widget JS is served from /static/chat-widget.js
    script_tag = (
        f'<script '
        f'src="/static/chat-widget.js" '
        f'data-tenant-id="{tenant.id}" '
        f'data-api-base="/api/v1/agent" '
        f'defer>'
        f'</script>'
    )

    return RegisterResponse(
        status="ok",
        tenant_id=str(tenant.id),
        spec_count=len(specs),
        script_tag=script_tag,
    )


@router.delete(
    "/register/{tenant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deregister a tenant",
)
async def deregister_tenant(
    tenant_id: uuid.UUID,
    repo: ITenantRepository = Depends(get_tenant_repo)
) -> None:
    deleted = await repo.delete(tenant_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found.",
        )
    logger.info("Deregistered tenant '%s'", tenant_id)


@router.get(
    "/tenants/{user_id}",
    summary="List all registered tenant IDs for a user",
)
async def list_tenants(
    user_id: uuid.UUID,
    repo: ITenantRepository = Depends(get_tenant_repo)
) -> Dict[str, Any]:
    tenants = await repo.get_by_user(user_id)
    return {"tenants": [{"id": str(t.id), "name": t.company_name} for t in tenants]}


@router.post(
    "/chat/{session_id}",
    summary="Send a user message and receive a streamed agent response (SSE)",
)
async def chat(
    session_id: str, 
    body: ChatRequest,
    repo: ITenantRepository = Depends(get_tenant_repo)
) -> StreamingResponse:
    """
    Streams the agent response as Server-Sent Events.

    SSE event types:
    - `token`               — incremental text token
    - `node_start`          — which LangGraph node is now active
    - `clarification_needed`— graph paused, user must reply via /resume
    - `done`                — stream complete
    - `error`               — unrecoverable error
    """
    try:
        tenant_id_uuid = uuid.UUID(body.tenant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")

    tenant = await repo.get_by_id(tenant_id_uuid)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{body.tenant_id}' not found.",
        )

    initial_state = _build_initial_state(tenant, body.message, session_id)

    return StreamingResponse(
        _stream_graph(session_id, initial_state),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post(
    "/resume/{session_id}",
    summary="Resume graph execution after a human-in-the-loop clarification pause",
)
async def resume(
    session_id: str, 
    body: ResumeRequest,
    repo: ITenantRepository = Depends(get_tenant_repo)
) -> StreamingResponse:
    """
    After the graph emits a `clarification_needed` event, the frontend collects
    the user's reply and POSTs it here. We inject it into the checkpoint and
    resume graph execution from the human_clarification node.
    """
    try:
        tenant_id_uuid = uuid.UUID(body.tenant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid tenant_id format")

    tenant = await repo.get_by_id(tenant_id_uuid)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{body.tenant_id}' not found.",
        )

    graph  = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Inject the user reply into the existing checkpoint
    graph.update_state(
        config,
        {"messages": [HumanMessage(content=body.message)]},
        as_node="human_clarification",
    )

    return StreamingResponse(
        _resume_stream(session_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get(
    "/session/{session_id}",
    summary="Get the current checkpoint state for a session",
)
async def get_session_state(session_id: str) -> Dict[str, Any]:
    """
    Returns the current LangGraph checkpoint state for debugging / inspection.
    Useful for frontends that need to know if a session is paused or complete.
    """
    graph  = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No state found for session '{session_id}'.",
            )
        # Return a safe subset — omit raw messages for brevity
        values = state.values if hasattr(state, "values") else {}
        return {
            "session_id":          session_id,
            "current_intent":      values.get("current_intent"),
            "api_type":            values.get("api_type"),
            "missing_information": values.get("missing_information", []),
            "supervisor_decision": values.get("supervisor_decision"),
            "retry_count":         values.get("retry_count", 0),
            "guardrail_blocked":   values.get("guardrail_blocked", False),
            "next_nodes":          list(state.next) if hasattr(state, "next") else [],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session state: {str(exc)}",
        )
