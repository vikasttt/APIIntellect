"""
LangGraph graph definition — Smart API Agent.

Topology:
  guardrail → decider → [rag | enhancer]
  enhancer  → [human_clarification* | supervisor]
  supervisor → [get_agent | create_agent | delete_agent | coder_agent]
  *_agent   → validatory → [END | supervisor (retry)]

Checkpointing:
  Uses AsyncPostgresSaver (langgraph-checkpoint-postgres) backed by a
  psycopg AsyncConnectionPool.  This enables:
    - Durable persistence across server restarts
    - Human-in-the-loop resume across multiple HTTP requests
    - Multi-tenant session isolation via thread_id
    - Horizontal scaling (any pod can resume any session)

  The pool is created once during FastAPI lifespan (init_graph) and
  closed gracefully on shutdown (close_graph).

*human_clarification is compiled with interrupt_before so the graph pauses
 there, allowing the frontend to inject user input via graph.update_state()
 and then resume.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from app.agent.state import AgentState
from app.agent.agents import (
    coder_agent_node,
    create_agent_node,
    decider_node,
    delete_agent_node,
    enhancer_node,
    get_agent_node,
    guardrail_node,
    human_clarification_node,
    rag_node,
    supervisor_node,
    validatory_node,
)

logger = logging.getLogger(__name__)


# ── Edge condition functions ──────────────────────────────────────────────────

def _route_after_guardrail(state: AgentState) -> str:
    if state.get("guardrail_blocked"):
        return END  # type: ignore[return-value]
    return "decider"


def _route_after_decider(state: AgentState) -> str:
    return "rag" if state.get("current_intent") == "rag" else "enhancer"


def _route_after_enhancer(state: AgentState) -> str:
    if state.get("missing_information"):
        return "human_clarification"
    return "supervisor"


def _route_after_human(state: AgentState) -> str:
    if state.get("missing_information"):
        return "human_clarification"
    return "supervisor"


def _route_after_supervisor(state: AgentState) -> str:
    decision = state.get("supervisor_decision", "end")
    if decision == "end":
        return END  # type: ignore[return-value]
    return decision


def _route_after_validatory(state: AgentState) -> str:
    if not state.get("validation_comments"):
        return END  # type: ignore[return-value]
    return "supervisor"


# ── Graph topology builder (pure — no checkpointer wired here) ────────────────

def _build_workflow() -> StateGraph:
    """Construct and return the compiled StateGraph without a checkpointer."""
    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("guardrail",           guardrail_node)
    workflow.add_node("decider",             decider_node)
    workflow.add_node("rag",                 rag_node)
    workflow.add_node("enhancer",            enhancer_node)
    workflow.add_node("human_clarification", human_clarification_node)
    workflow.add_node("supervisor",          supervisor_node)
    workflow.add_node("get_agent",           get_agent_node)
    workflow.add_node("create_agent",        create_agent_node)
    workflow.add_node("delete_agent",        delete_agent_node)
    workflow.add_node("coder_agent",         coder_agent_node)
    workflow.add_node("validatory",          validatory_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("guardrail")

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {END: END, "decider": "decider"},
    )
    workflow.add_conditional_edges(
        "decider",
        _route_after_decider,
        {"rag": "rag", "enhancer": "enhancer"},
    )
    workflow.add_edge("rag", END)
    workflow.add_conditional_edges(
        "enhancer",
        _route_after_enhancer,
        {
            "human_clarification": "human_clarification",
            "supervisor":          "supervisor",
        },
    )
    workflow.add_conditional_edges(
        "human_clarification",
        _route_after_human,
        {
            "human_clarification": "human_clarification",
            "supervisor":          "supervisor",
        },
    )
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            END:            END,
            "get_agent":    "get_agent",
            "create_agent": "create_agent",
            "delete_agent": "delete_agent",
            "coder_agent":  "coder_agent",
        },
    )
    for agent_node_name in ["get_agent", "create_agent", "delete_agent", "coder_agent"]:
        workflow.add_edge(agent_node_name, "validatory")
    workflow.add_conditional_edges(
        "validatory",
        _route_after_validatory,
        {END: END, "supervisor": "supervisor"},
    )

    return workflow


# ── Module-level state for the compiled graph + connection pool ───────────────

_graph: Optional[Any] = None
_pool:  Optional[Any] = None   # psycopg AsyncConnectionPool


async def init_graph(conn_string: str) -> None:
    """
    Build the LangGraph application with AsyncPostgresSaver.

    Must be called once during FastAPI lifespan (startup).

    Parameters
    ----------
    conn_string
        A plain psycopg-style connection string, e.g.
        ``"postgresql://user:pass@host:5432/dbname"``
        (NOT the asyncpg ``+asyncpg`` variant — psycopg3 is used here).
    """
    global _graph, _pool

    from psycopg_pool import AsyncConnectionPool  # type: ignore[import]
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore[import]

    logger.info("Initialising LangGraph PostgreSQL checkpointer …")

    # Open a managed async connection pool
    _pool = AsyncConnectionPool(
        conninfo=conn_string,
        min_size=2,
        max_size=10,
        open=False,          # we open explicitly below
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    await _pool.open()

    checkpointer = AsyncPostgresSaver(_pool)

    # Ensure the langgraph_checkpoints schema/tables exist
    await checkpointer.setup()

    workflow = _build_workflow()
    _graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_clarification"],
    )

    logger.info("LangGraph graph compiled with AsyncPostgresSaver ✓")


async def close_graph() -> None:
    """Gracefully close the connection pool. Call from FastAPI lifespan shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("LangGraph PostgreSQL connection pool closed.")


def get_graph() -> Any:
    """
    Return the compiled LangGraph application.

    Raises RuntimeError if init_graph() has not been called yet.
    """
    if _graph is None:
        raise RuntimeError(
            "LangGraph graph is not initialised. "
            "Ensure init_graph() is called during application startup."
        )
    return _graph
