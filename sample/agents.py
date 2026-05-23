"""
All LangGraph node implementations for the Smart API Agent system.

Every node is an async function: (AgentState) -> partial AgentState dict.
This keeps the graph composable, testable, and side-effect-free at the node level.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from state import AgentState
from tools import create_dynamic_tools, create_lookup_tool, get_coder_tool
from prompts import (
    DeciderOutput, EnhancerOutput, SupervisorDecision, ValidatoryOutput, GuardrailOutput,
    GUARDRAIL_PROMPT, DECIDER_PROMPT, RAG_PROMPT, ENHANCER_PROMPT,
    SUPERVISOR_PROMPT, GET_AGENT_SYSTEM_PROMPT, CREATE_AGENT_SYSTEM_PROMPT,
    DELETE_AGENT_SYSTEM_PROMPT, CODER_AGENT_SYSTEM_PROMPT, VALIDATORY_PROMPT,
)
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

# ── Shared LLM instance (swap model in one place) ───────────────────────────
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

MAX_RETRIES = 3


# ════════════════════════════════════════════════════════════════════════════
#  MIDDLEWARE / GUARDRAIL
# ════════════════════════════════════════════════════════════════════════════

async def guardrail_node(state: AgentState) -> Dict[str, Any]:
    """
    First node in the graph. Screens the incoming query for:
      - Prompt injection
      - Off-topic / abusive content
      - Attempts to exfiltrate data outside defined APIs
    Blocks execution early if unsafe.
    """
    structured_llm = llm.with_structured_output(GuardrailOutput)
    response: GuardrailOutput = await structured_llm.ainvoke(
        [SystemMessage(content=GUARDRAIL_PROMPT)] + list(state["messages"])
    )

    if not response.safe:
        logger.warning("Guardrail blocked query: %s", response.reason)
        return {
            "guardrail_blocked": True,
            "messages": [AIMessage(content=f"🚫 Request blocked: {response.reason}")],
        }

    return {"guardrail_blocked": False}


# ════════════════════════════════════════════════════════════════════════════
#  DECIDER
# ════════════════════════════════════════════════════════════════════════════

async def decider_node(state: AgentState) -> Dict[str, Any]:
    """Routes the query to RAG or API pipeline."""
    api_specs = state.get("api_specs", [])
    api_specs_summary = ", ".join(s.get("name", "") for s in api_specs) or "none"

    structured_llm = llm.with_structured_output(DeciderOutput)
    response: DeciderOutput = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=DECIDER_PROMPT.format(api_specs_summary=api_specs_summary)
            )
        ]
        + list(state["messages"])
    )

    logger.info("Decider → intent=%s  reason=%s", response.intent, response.reasoning)
    return {"current_intent": response.intent}


# ════════════════════════════════════════════════════════════════════════════
#  RAG
# ════════════════════════════════════════════════════════════════════════════

async def rag_node(state: AgentState) -> Dict[str, Any]:
    """Answers using static company/product knowledge."""
    company_info = state.get("company_info", "No company information available.")
    company_name = state.get("company_name", "our company")

    formatted = RAG_PROMPT.format(
        company_name=company_name, company_info=company_info
    )
    response = await llm.ainvoke(
        [SystemMessage(content=formatted)] + list(state["messages"])
    )
    return {"messages": [response]}


# ════════════════════════════════════════════════════════════════════════════
#  ENHANCER
# ════════════════════════════════════════════════════════════════════════════

async def enhancer_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the query against API specs to determine:
      - Which API to call (api_type + target_api_name)
      - Missing information (fields the user hasn't supplied)
      - Lookups needed (e.g., department name → department_id)
      - An enhanced, unambiguous query string
    """
    api_specs = state.get("api_specs", [])
    collected  = state.get("collected_fields", {})

    # Build context string including already collected fields
    collected_context = (
        f"Already collected: {json.dumps(collected)}"
        if collected
        else "No fields collected yet."
    )

    structured_llm = llm.with_structured_output(EnhancerOutput)
    response: EnhancerOutput = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=ENHANCER_PROMPT.format(
                    api_specs=json.dumps(api_specs, indent=2),
                    collected_context=collected_context,
                )
            )
        ]
        + list(state["messages"])
    )

    logger.info(
        "Enhancer → api_type=%s  missing=%s  lookups=%s",
        response.api_type,
        response.missing_information,
        response.lookup_needed,
    )

    return {
        "api_type": response.api_type,
        "missing_information": response.missing_information,
        "lookup_needed": response.lookup_needed,
        # Store enhanced query as an assistant message so agents can see it
        "messages": [AIMessage(content=f"[Enhanced]: {response.enhanced_query}")],
    }


# ════════════════════════════════════════════════════════════════════════════
#  HUMAN-IN-THE-LOOP  (interrupt_before this node)
# ════════════════════════════════════════════════════════════════════════════

async def human_clarification_node(state: AgentState) -> Dict[str, Any]:
    """
    Asks the user for missing fields.
    The graph is interrupted *before* this node so the frontend can inject the
    user's reply via graph.update_state(), then resume from here.
    When resumed, this node reads the latest HumanMessage, merges its content
    into collected_fields, and clears missing_information.
    """
    missing = state.get("missing_information", [])

    # On initial interrupt: ask the question, then the graph pauses.
    # On resume (after user reply): the new HumanMessage is already in state.
    last_message = state["messages"][-1]

    if isinstance(last_message, HumanMessage) and missing:
        # User replied – parse their answer into collected_fields
        # Simple strategy: ask the LLM to extract key-value pairs
        extract_prompt = (
            f"Extract the following fields from the user reply as JSON: {missing}.\n"
            f"User reply: \"{last_message.content}\"\n"
            "Return ONLY valid JSON, e.g. {{\"severity\": \"high\", \"department\": \"IT\"}}"
        )
        raw = await llm.ainvoke([HumanMessage(content=extract_prompt)])
        try:
            extracted = json.loads(raw.content)
        except Exception:
            extracted = {}

        current_collected = state.get("collected_fields", {})
        merged = {**current_collected, **extracted}

        # Remove fields that are now provided
        still_missing = [f for f in missing if f not in extracted]

        return {
            "collected_fields": merged,
            "missing_information": still_missing,
        }

    # First time hitting this node – emit the clarification question
    fields_str = "\n".join(f"  • {f}" for f in missing)
    question = f"To proceed, I need the following information:\n{fields_str}\n\nPlease provide these details."
    return {"messages": [AIMessage(content=question)]}


# ════════════════════════════════════════════════════════════════════════════
#  SUPERVISOR
# ════════════════════════════════════════════════════════════════════════════

async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Orchestration hub. Decides which specialist agent runs next based on:
      - api_type from enhancer
      - validation_comments from validatory
      - retry_count guard-rail
    """
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRIES:
        logger.warning("Max retries (%d) reached – ending workflow.", MAX_RETRIES)
        return {
            "supervisor_decision": "end",
            "messages": [
                AIMessage(
                    content=(
                        "I wasn't able to complete your request after several attempts. "
                        "Please contact support or rephrase your query."
                    )
                )
            ],
        }

    state_summary = json.dumps(
        {
            "api_type":            state.get("api_type"),
            "validation_comments": state.get("validation_comments"),
            "retry_count":         retry_count,
            "lookup_needed":       state.get("lookup_needed", []),
        },
        indent=2,
    )

    structured_llm = llm.with_structured_output(SupervisorDecision)
    decision: SupervisorDecision = await structured_llm.ainvoke(
        [SystemMessage(content=SUPERVISOR_PROMPT.format(state_summary=state_summary))]
        + list(state["messages"])
    )

    logger.info("Supervisor → next=%s  reason=%s", decision.next_node, decision.reasoning)

    return {
        "supervisor_decision": decision.next_node,
        "retry_count": retry_count + 1,
    }


# ════════════════════════════════════════════════════════════════════════════
#  SPECIALIST AGENTS  (GET / CREATE / DELETE / CODER)
# ════════════════════════════════════════════════════════════════════════════

def _filter_specs(api_specs: List[Dict], methods: List[str]) -> List[Dict]:
    return [s for s in api_specs if s.get("method", "").upper() in methods]


async def _run_react_agent(
    state: AgentState,
    tools,
    system_prompt: str,
) -> Dict[str, Any]:
    """Helper: builds a ReAct agent on-the-fly and runs it asynchronously."""
    agent = create_react_agent(llm, tools=tools)
    result = await agent.ainvoke(
        {
            "messages": [SystemMessage(content=system_prompt)]
            + list(state["messages"])
        }
    )
    # Return only the final AI response
    return {"messages": result["messages"][-1:]}


async def get_agent_node(state: AgentState) -> Dict[str, Any]:
    """Handles all read (GET) operations."""
    specs = _filter_specs(state.get("api_specs", []), ["GET"])
    # Also add any needed lookup tools
    lookup_tools = _build_lookup_tools(state)
    tools = create_dynamic_tools(specs) + lookup_tools

    system = GET_AGENT_SYSTEM_PROMPT.format(
        tool_names=", ".join(t.name for t in tools)
    )
    return await _run_react_agent(state, tools, system)


async def create_agent_node(state: AgentState) -> Dict[str, Any]:
    """Handles all write (POST/PUT/PATCH) operations."""
    specs = _filter_specs(state.get("api_specs", []), ["POST", "PUT", "PATCH"])
    lookup_tools = _build_lookup_tools(state)
    tools = create_dynamic_tools(specs) + lookup_tools

    system = CREATE_AGENT_SYSTEM_PROMPT.format(
        tool_names=", ".join(t.name for t in tools)
    )
    return await _run_react_agent(state, tools, system)


async def delete_agent_node(state: AgentState) -> Dict[str, Any]:
    """Handles all delete (DELETE) operations."""
    specs = _filter_specs(state.get("api_specs", []), ["DELETE"])
    lookup_tools = _build_lookup_tools(state)
    tools = create_dynamic_tools(specs) + lookup_tools

    system = DELETE_AGENT_SYSTEM_PROMPT.format(
        tool_names=", ".join(t.name for t in tools)
    )
    return await _run_react_agent(state, tools, system)


async def coder_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Handles complex analytical queries by combining API tools + Python REPL.
    Typical use-case: 'how many cases were created in September grouped by severity?'
    when only a list-cases API exists.
    """
    api_specs = state.get("api_specs", [])
    all_api_tools = create_dynamic_tools(api_specs)
    repl_tool    = get_coder_tool()
    tools        = all_api_tools + [repl_tool]

    error_context = state.get("error_context", "None")
    system = CODER_AGENT_SYSTEM_PROMPT.format(error_context=error_context)
    return await _run_react_agent(state, tools, system)


# ════════════════════════════════════════════════════════════════════════════
#  VALIDATORY
# ════════════════════════════════════════════════════════════════════════════

async def validatory_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the specialist agent's response.
    If INVALID, returns validation_comments so supervisor can retry.
    """
    # Original user query is the first HumanMessage
    original_query = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "Unknown query",
    )

    structured_llm = llm.with_structured_output(ValidatoryOutput)
    result: ValidatoryOutput = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=VALIDATORY_PROMPT.format(original_query=original_query)
            )
        ]
        + list(state["messages"])
    )

    logger.info("Validatory → status=%s", result.status)

    if result.status == "VALID":
        return {"validation_comments": "", "error_context": None}

    # Store error for potential coder agent retry
    return {
        "validation_comments": result.validation_comments,
        "error_context": result.validation_comments,
    }


# ════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _build_lookup_tools(state: AgentState) -> list:
    """Build lookup tools for any pending lookup tasks in state."""
    lookup_needed = state.get("lookup_needed", [])
    api_specs     = state.get("api_specs", [])
    tools = []
    for lookup_spec in lookup_needed:
        # LookupTask may be a Pydantic model (from EnhancerOutput) or a plain dict
        spec_dict = (
            lookup_spec.model_dump() if hasattr(lookup_spec, "model_dump") else lookup_spec
        )
        tool = create_lookup_tool(spec_dict, api_specs)
        if tool:
            tools.append(tool)
    return tools
