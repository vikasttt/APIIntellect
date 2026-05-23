"""
FastAPI server exposing the Smart API Agent as an HTTP + SSE endpoint.

Endpoints:
  POST /register          → register company info + API specs for a tenant
  POST /chat/{session_id} → send a user message (returns SSE stream)
  POST /resume/{session_id} → resume after human-in-the-loop pause

CORS is configured to allow the embedded chat widget from any origin.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from graph import build_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App initialisation ───────────────────────────────────────────────────────
app = FastAPI(
    title="Smart API Agent",
    description="Dynamic agentic chat backend powered by LangGraph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory tenant registry (replace with DB in production) ────────────────
_tenant_registry: Dict[str, Dict[str, Any]] = {}

# ── Single compiled graph (shared across all tenants) ────────────────────────
graph = build_graph()


# ════════════════════════════════════════════════════════════════════════════
#  Request / Response models
# ════════════════════════════════════════════════════════════════════════════

class ApiSpecModel(BaseModel):
    name:        str
    description: str
    method:      str
    url:         str
    headers:     Dict[str, str]   = Field(default_factory=dict)
    query_params: Dict[str, str]  = Field(default_factory=dict)
    request_body: Dict[str, str]  = Field(default_factory=dict)
    path_params:  List[str]       = Field(default_factory=list)
    lookup_fields: Dict[str, Any] = Field(default_factory=dict)


class RegisterRequest(BaseModel):
    tenant_id:    str
    company_name: str
    company_info: str             # RAG knowledge base (free text)
    api_specs:    List[ApiSpecModel]


class ChatRequest(BaseModel):
    tenant_id: str
    message:   str


class ResumeRequest(BaseModel):
    tenant_id: str
    message:   str                # user's reply to the clarification question


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _get_tenant(tenant_id: str) -> Dict[str, Any]:
    tenant = _tenant_registry.get(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not registered.")
    return tenant


async def _stream_graph(
    session_id: str,
    initial_state: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Run the graph and yield SSE-formatted events."""
    config = {"configurable": {"thread_id": session_id}}

    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        kind = event.get("event")
        name = event.get("name", "")
        data = event.get("data", {})

        # Stream token-by-token for AI messages
        if kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        # Emit node-level events so the frontend knows which agent is active
        elif kind == "on_chain_start" and name in {
            "guardrail", "decider", "enhancer", "supervisor",
            "get_agent", "create_agent", "delete_agent", "coder_agent", "validatory",
        }:
            yield f"data: {json.dumps({'type': 'node_start', 'node': name})}\n\n"

        # Human-in-the-loop pause: the graph emitted the clarification question
        elif kind == "on_chain_end" and name == "human_clarification":
            output = data.get("output", {})
            msgs   = output.get("messages", [])
            if msgs:
                content = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
                yield f"data: {json.dumps({'type': 'clarification_needed', 'content': content})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ════════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════════

@app.post("/register", summary="Register tenant with API specs and company info")
async def register_tenant(body: RegisterRequest) -> Dict[str, str]:
    _tenant_registry[body.tenant_id] = {
        "company_name": body.company_name,
        "company_info": body.company_info,
        "api_specs":    [s.model_dump() for s in body.api_specs],
    }
    logger.info("Registered tenant: %s with %d API specs", body.tenant_id, len(body.api_specs))
    return {"status": "ok", "tenant_id": body.tenant_id}


@app.post("/chat/{session_id}", summary="Send a user message and stream the agent response")
async def chat(session_id: str, body: ChatRequest) -> StreamingResponse:
    tenant = _get_tenant(body.tenant_id)

    initial_state: Dict[str, Any] = {
        "messages":            [HumanMessage(content=body.message)],
        "company_name":        tenant["company_name"],
        "company_info":        tenant["company_info"],
        "api_specs":           tenant["api_specs"],
        "current_intent":      "api",
        "api_type":            "unknown",
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

    return StreamingResponse(
        _stream_graph(session_id, initial_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/resume/{session_id}", summary="Resume graph after human-in-the-loop pause")
async def resume(session_id: str, body: ResumeRequest) -> StreamingResponse:
    """
    After the graph pauses at human_clarification, the frontend POSTs the user's
    reply here. We inject it into the checkpointed state and resume execution.
    """
    _get_tenant(body.tenant_id)  # validate tenant exists

    config = {"configurable": {"thread_id": session_id}}

    # Inject the user reply into the existing checkpoint
    graph.update_state(
        config,
        {"messages": [HumanMessage(content=body.message)]},
        as_node="human_clarification",
    )

    async def _resume_stream() -> AsyncGenerator[str, None]:
        async for event in graph.astream_events(None, config=config, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")
            data = event.get("data", {})

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

            elif kind == "on_chain_start" and name in {
                "human_clarification", "supervisor", "get_agent",
                "create_agent", "delete_agent", "coder_agent", "validatory",
            }:
                yield f"data: {json.dumps({'type': 'node_start', 'node': name})}\n\n"

            elif kind == "on_chain_end" and name == "human_clarification":
                output = data.get("output", {})
                msgs   = output.get("messages", [])
                if msgs:
                    content = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
                    yield f"data: {json.dumps({'type': 'clarification_needed', 'content': content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        _resume_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# ── Dev runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
