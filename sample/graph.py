"""
LangGraph graph definition for the Smart API Agent system.

Graph topology:
  guardrail → decider → [rag | enhancer]
  enhancer  → [human_clarification* | supervisor]
  supervisor → [get_agent | create_agent | delete_agent | coder_agent]
  *_agent   → validatory → [END | supervisor (retry)]

*human_clarification is an interrupt_before node – the graph pauses here
 so the frontend can collect user input and resume.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from agents import (
    guardrail_node,
    decider_node,
    rag_node,
    enhancer_node,
    human_clarification_node,
    supervisor_node,
    get_agent_node,
    create_agent_node,
    delete_agent_node,
    coder_agent_node,
    validatory_node,
)


# ── Edge condition functions ─────────────────────────────────────────────────

def _route_after_guardrail(state: AgentState) -> str:
    if state.get("guardrail_blocked"):
        return END  # type: ignore[return-value]
    return "decider"


def _route_after_decider(state: AgentState) -> str:
    return "rag" if state.get("current_intent") == "rag" else "enhancer"


def _route_after_enhancer(state: AgentState) -> str:
    """
    Go to human_clarification if information is still missing,
    otherwise hand off to supervisor.
    """
    if state.get("missing_information"):
        return "human_clarification"
    return "supervisor"


def _route_after_human(state: AgentState) -> str:
    """
    After the user replies, check if there is still missing information.
    Loop back to human_clarification until satisfied, then go to supervisor.
    """
    if state.get("missing_information"):
        return "human_clarification"
    return "supervisor"


def _route_after_supervisor(state: AgentState) -> str:
    decision = state.get("supervisor_decision", "end")
    if decision == "end":
        return END  # type: ignore[return-value]
    return decision  # one of: get_agent | create_agent | delete_agent | coder_agent


def _route_after_validatory(state: AgentState) -> str:
    """
    If validation passes → end.
    If validation fails  → re-route to supervisor (which may pick a different agent).
    """
    if not state.get("validation_comments"):
        return END  # type: ignore[return-value]
    return "supervisor"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph() -> object:
    """
    Compile and return the LangGraph application with MemorySaver checkpoint.
    The graph is compiled with interrupt_before=["human_clarification"] so the
    workflow pauses there and waits for external user input before resuming.
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────────
    workflow.add_node("guardrail",          guardrail_node)
    workflow.add_node("decider",            decider_node)
    workflow.add_node("rag",                rag_node)
    workflow.add_node("enhancer",           enhancer_node)
    workflow.add_node("human_clarification",human_clarification_node) 
    workflow.add_node("supervisor",         supervisor_node)
    workflow.add_node("get_agent",          get_agent_node)
    workflow.add_node("create_agent",       create_agent_node)
    workflow.add_node("delete_agent",       delete_agent_node)
    workflow.add_node("coder_agent",        coder_agent_node)
    workflow.add_node("validatory",         validatory_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    workflow.set_entry_point("guardrail")

    # ── Edges ───────────────────────────────────────────────────────────────

    # Guardrail → blocked (END) or continue
    workflow.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {END: END, "decider": "decider"},
    )

    # Decider → RAG or API pipeline
    workflow.add_conditional_edges(
        "decider",
        _route_after_decider,
        {"rag": "rag", "enhancer": "enhancer"},
    )

    # RAG → done
    workflow.add_edge("rag", END)

    # Enhancer → ask user for missing info OR hand off to supervisor
    workflow.add_conditional_edges(
        "enhancer",
        _route_after_enhancer,
        {"human_clarification": "human_clarification", "supervisor": "supervisor"},
    )

    # Human clarification → loop back if still missing, else supervisor
    workflow.add_conditional_edges(
        "human_clarification",
        _route_after_human,
        {"human_clarification": "human_clarification", "supervisor": "supervisor"},
    )

    # Supervisor → specialist agents OR end (max-retries)
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            END:             END,
            "get_agent":     "get_agent",
            "create_agent":  "create_agent",
            "delete_agent":  "delete_agent",
            "coder_agent":   "coder_agent",
        },
    )

    # All specialist agents → validatory
    for agent_node in ["get_agent", "create_agent", "delete_agent", "coder_agent"]:
        workflow.add_edge(agent_node, "validatory")

    # Validatory → done or retry via supervisor
    workflow.add_conditional_edges(
        "validatory",
        _route_after_validatory,
        {END: END, "supervisor": "supervisor"},
    )

    # ── Compile with persistent memory + human-in-the-loop interrupt ────────
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_clarification"],
    )

    return app


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()
    print("✅ Graph compiled successfully.")
    # Optionally draw the graph
    try:
        img_bytes = app.get_graph().draw_mermaid_png()
        with open("graph_diagram.png", "wb") as f:
            f.write(img_bytes)
        print("✅ Graph diagram saved as graph_diagram.png")
    except Exception as e:
        print(f"ℹ️  Could not draw graph (install pygraphviz): {e}")
