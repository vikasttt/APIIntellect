"""
All LangGraph node implementations for the Smart API Agent.

Every node is an async function: (AgentState) → partial AgentState dict.

Key changes vs. the sample:
- rag_node uses the real production hybrid vector-search pipeline
  (multi-query → pgvector + full-text fusion → Cohere rerank) rather than
  answering from a raw company_info string.
- The tenant's company_info string is kept as a fallback when no document
  chunks are found (e.g. the tenant hasn't uploaded any files yet).
- Model: gpt-4o (one constant — swap in one place).
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agent.state import AgentState
from app.agent.tools import create_dynamic_tools, create_lookup_tool, get_coder_tool
from app.agent.rag_services import get_rag_services
from app.agent.prompts import (
    DeciderOutput,
    EnhancerOutput,
    SupervisorDecision,
    ValidatoryOutput,
    GuardrailOutput,
    GUARDRAIL_PROMPT,
    DECIDER_PROMPT,
    RAG_PROMPT,
    ENHANCER_PROMPT,
    SUPERVISOR_PROMPT,
    GET_AGENT_SYSTEM_PROMPT,
    CREATE_AGENT_SYSTEM_PROMPT,
    DELETE_AGENT_SYSTEM_PROMPT,
    CODER_AGENT_SYSTEM_PROMPT,
    VALIDATORY_PROMPT,
)

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────
_LLM_MODEL   = "gpt-5.4-mini"   # Swap to "gpt-4o-mini" for cost savings
_TEMPERATURE = 0.0
MAX_RETRIES  = 3

def get_llm() -> ChatOpenAI:
    from app.interfaces.api.dependencies.container import get_container
    from app.config.settings import Settings

    container = get_container()
    settings = container.resolve(Settings)
    return ChatOpenAI(
        model=_LLM_MODEL,
        temperature=_TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )


# ════════════════════════════════════════════════════════════════════════════
#  GUARDRAIL
# ════════════════════════════════════════════════════════════════════════════

async def guardrail_node(state: AgentState) -> Dict[str, Any]:
    """
    Security middleware — screens for prompt injection, off-topic content,
    and data-exfiltration attempts.  Blocks early if unsafe.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(GuardrailOutput)
    response: GuardrailOutput = await structured_llm.ainvoke(
        [SystemMessage(content=GUARDRAIL_PROMPT)] + list(state["messages"])
    )

    if not response.safe:
        logger.warning("Guardrail blocked: %s", response.reason)
        return {
            "guardrail_blocked": True,
            "messages": [AIMessage(content=f"🚫 Request blocked: {response.reason}")],
        }

    return {"guardrail_blocked": False}


# ════════════════════════════════════════════════════════════════════════════
#  DECIDER
# ════════════════════════════════════════════════════════════════════════════

async def decider_node(state: AgentState) -> Dict[str, Any]:
    """Routes the query to RAG (knowledge) or the API execution pipeline."""
    api_specs         = state.get("api_specs", [])
    api_specs_summary = ", ".join(s.get("name", "") for s in api_specs) or "none"

    llm = get_llm()
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
#  RAG  — uses the real production vector-search pipeline
# ════════════════════════════════════════════════════════════════════════════

async def rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Answers company / product questions via the existing RAG infrastructure.

    Pipeline:
      1. Multi-query expansion (generate N alternative phrasings).
      2. For each phrasing: embed → pgvector cosine similarity search.
      3. Deduplicate and merge candidates.
      4. Cohere rerank → keep top-n.
      5. Build context from chunk text → call gpt-4o with system prompt.

    Fallback: if no RagServices are configured (tenant hasn't uploaded docs),
    answers from the tenant's company_info free-text string.
    """
    company_info = state.get("company_info", "No company information available.")
    company_name = state.get("company_name", "our company")
    tenant_id    = state.get("conversation_id", "unknown")  # used as user_id proxy

    svc = get_rag_services()

    # ── Attempt real vector search ────────────────────────────────────────────
    context: str = company_info   # default fallback
    sources: List[Dict[str, Any]] = []

    if svc is not None:
        try:
            # Extract the latest user query
            query = next(
                (m.content for m in reversed(list(state["messages"]))
                 if isinstance(m, HumanMessage)),
                "",
            )

            if query:
                # 1. Generate query variants
                sub_queries = await svc.multi_query.generate_queries(
                    query, svc.multi_query_count
                )
                all_queries = [query] + sub_queries

                # 2. Embed + similarity search across all variants
                from app.infrastructure.db.repositories.chunk_repository import (
                    SqlChunkRepository,
                )

                seen: Dict[str, tuple] = {}
                async with svc.session_factory() as session:
                    repo = SqlChunkRepository(session)
                    for q in all_queries:
                        embedding = await svc.embedder.embed_query(q)
                        results = await repo.similarity_search(
                            embedding,
                            user_id=tenant_id,
                            top_k=svc.top_k,
                        )
                        for chunk, score in results:
                            chunk_key = str(chunk.id)
                            if chunk_key not in seen or seen[chunk_key][1] < score:
                                seen[chunk_key] = (chunk, score)

                candidates = list(seen.values())

                # 3. Cohere rerank
                if svc.rerank_enabled and candidates:
                    chunks_only = [c for c, _ in candidates]
                    reranked = await svc.reranker.rerank(
                        query, chunks_only, svc.rerank_top_n
                    )
                    final = reranked
                else:
                    final = sorted(candidates, key=lambda x: x[1], reverse=True)[
                        : svc.rerank_top_n
                    ]

                if final:
                    context_parts = [c.content for c, _ in final]
                    # Prepend company_info so tenant knowledge is always available
                    context = (
                        f"=== Company/Product Knowledge ===\n{company_info}\n\n"
                        f"=== Retrieved Document Chunks ===\n"
                        + "\n\n---\n\n".join(context_parts)
                    )
                    sources = [
                        {
                            "chunk_id":       str(c.id),
                            "document_id":    str(c.document_id),
                            "section_title":  c.section_title,
                            "score":          round(float(s), 4),
                            "content_snippet": c.content[:200],
                        }
                        for c, s in final
                    ]
                    logger.info(
                        "RAG node retrieved %d chunks for tenant=%s", len(final), tenant_id
                    )

        except Exception as exc:
            # Non-fatal: fall back to company_info only
            logger.warning(
                "RAG vector search failed, falling back to company_info. error=%s", exc
            )

    # ── Build prompt + call LLM ───────────────────────────────────────────────
    formatted = RAG_PROMPT.format(
        company_name=company_name, company_info=context
    )
    llm = get_llm()
    response = await llm.ainvoke(
        [SystemMessage(content=formatted)] + list(state["messages"])
    )

    # Attach source metadata to the AI message so the route layer can surface it
    if sources:
        response.additional_kwargs["rag_sources"] = sources  # type: ignore[assignment]

    return {"messages": [response]}


# ════════════════════════════════════════════════════════════════════════════
#  ENHANCER
# ════════════════════════════════════════════════════════════════════════════

async def enhancer_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyses the query against API specs to determine:
      - Which API to call (api_type + target_api_name)
      - Missing information (fields not yet supplied by user)
      - Lookups needed (e.g. department name → department_id)
      - An enhanced, unambiguous query string
    """
    api_specs  = state.get("api_specs", [])
    collected  = state.get("collected_fields", {})
    collected_context = (
        f"Already collected: {json.dumps(collected)}"
        if collected
        else "No fields collected yet."
    )

    llm = get_llm()
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
        "Enhancer → api_type=%s  missing=%s  lookups=%d",
        response.api_type,
        response.missing_information,
        len(response.lookup_needed),
    )

    # Serialise LookupTask Pydantic models to plain dicts (state must be JSON-safe)
    lookup_needed_dicts = [
        lt.model_dump() if hasattr(lt, "model_dump") else lt
        for lt in response.lookup_needed
    ]

    return {
        "api_type":            response.api_type,
        "target_api_name":     response.target_api_name,
        "missing_information": response.missing_information,
        "lookup_needed":       lookup_needed_dicts,
        "messages": [
            AIMessage(content=f"[Enhanced]: {response.enhanced_query}")
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
#  HUMAN-IN-THE-LOOP  (graph is interrupted *before* this node)
# ════════════════════════════════════════════════════════════════════════════

async def human_clarification_node(state: AgentState) -> Dict[str, Any]:
    """
    Asks the user for missing fields.

    Flow:
      1. Graph interrupted before this node (LangGraph interrupt_before).
      2. Frontend shows the clarification question.
      3. User replies via POST /api/v1/agent/resume/{session_id}.
      4. graph.update_state() injects the reply as a HumanMessage.
      5. Graph resumes here — we parse the reply, merge into collected_fields,
         and clear any fields that are now provided.
    """
    missing      = state.get("missing_information", [])
    last_message = state["messages"][-1]

    if isinstance(last_message, HumanMessage) and missing:
        # User replied — extract key-value pairs from natural language
        extract_prompt = (
            f"Extract the following fields from the user reply as JSON: {missing}.\n"
            f'User reply: "{last_message.content}"\n'
            'Return ONLY valid JSON, e.g. {"severity": "high", "department": "IT"}'
        )
        llm = get_llm()
        raw = await llm.ainvoke([HumanMessage(content=extract_prompt)])
        try:
            extracted: dict = json.loads(raw.content)
        except Exception:
            extracted = {}

        merged       = {**state.get("collected_fields", {}), **extracted}
        still_missing = [f for f in missing if f not in extracted]

        return {
            "collected_fields":    merged,
            "missing_information": still_missing,
        }

    # First arrival — emit the clarification question
    fields_str = "\n".join(f"  • {f}" for f in missing)
    question = (
        f"To proceed, I need a bit more information:\n{fields_str}\n\n"
        "Please provide these details and I'll continue right away."
    )
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
        logger.warning("Max retries (%d) reached — ending workflow.", MAX_RETRIES)
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
            "target_api_name":     state.get("target_api_name"),
            "validation_comments": state.get("validation_comments"),
            "retry_count":         retry_count,
            "lookup_needed":       state.get("lookup_needed", []),
        },
        indent=2,
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(SupervisorDecision)
    decision: SupervisorDecision = await structured_llm.ainvoke(
        [SystemMessage(content=SUPERVISOR_PROMPT.format(state_summary=state_summary))]
        + list(state["messages"])
    )

    logger.info("Supervisor → next=%s  reason=%s", decision.next_node, decision.reasoning)
    return {
        "supervisor_decision": decision.next_node,
        "retry_count":         retry_count + 1,
    }


# ════════════════════════════════════════════════════════════════════════════
#  SPECIALIST AGENTS  (GET / CREATE / DELETE / CODER)
# ════════════════════════════════════════════════════════════════════════════

def _filter_specs(api_specs: List[Dict], methods: List[str]) -> List[Dict]:
    return [s for s in api_specs if s.get("method", "").upper() in methods]


def _build_lookup_tools(state: AgentState) -> list:
    """Build lookup tools for any pending lookup tasks recorded in state."""
    lookup_needed = state.get("lookup_needed", [])
    api_specs     = state.get("api_specs", [])
    tools = []
    for lookup_spec in lookup_needed:
        spec_dict = (
            lookup_spec.model_dump()
            if hasattr(lookup_spec, "model_dump")
            else lookup_spec
        )
        tool = create_lookup_tool(spec_dict, api_specs)
        if tool:
            tools.append(tool)
    return tools


async def _run_react_agent(
    state: AgentState,
    tools: list,
    system_prompt: str,
) -> Dict[str, Any]:
    """Build a ReAct agent on-the-fly and run it asynchronously."""
    llm = get_llm()
    agent = create_react_agent(llm, tools=tools)
    result = await agent.ainvoke(
        {
            "messages": [SystemMessage(content=system_prompt)]
            + list(state["messages"])
        }
    )
    return {"messages": result["messages"][-1:]}


async def get_agent_node(state: AgentState) -> Dict[str, Any]:
    """Handles all read (GET) operations."""
    specs        = _filter_specs(state.get("api_specs", []), ["GET"])
    lookup_tools = _build_lookup_tools(state)
    tools        = create_dynamic_tools(specs) + lookup_tools
    system       = GET_AGENT_SYSTEM_PROMPT.format(
        tool_names=", ".join(t.name for t in tools)
    )
    return await _run_react_agent(state, tools, system)


async def create_agent_node(state: AgentState) -> Dict[str, Any]:
    """Handles all write (POST / PUT / PATCH) operations."""
    specs        = _filter_specs(state.get("api_specs", []), ["POST", "PUT", "PATCH"])
    lookup_tools = _build_lookup_tools(state)
    tools        = create_dynamic_tools(specs) + lookup_tools
    system       = CREATE_AGENT_SYSTEM_PROMPT.format(
        tool_names=", ".join(t.name for t in tools)
    )
    return await _run_react_agent(state, tools, system)


async def delete_agent_node(state: AgentState) -> Dict[str, Any]:
    """Handles all DELETE operations."""
    specs        = _filter_specs(state.get("api_specs", []), ["DELETE"])
    lookup_tools = _build_lookup_tools(state)
    tools        = create_dynamic_tools(specs) + lookup_tools
    system       = DELETE_AGENT_SYSTEM_PROMPT.format(
        tool_names=", ".join(t.name for t in tools)
    )
    return await _run_react_agent(state, tools, system)


async def coder_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Handles complex analytical queries by combining all API tools + Python REPL.
    Typical: 'how many cases were created in September grouped by severity?'
    when only a list-cases API exists.
    """
    api_specs     = state.get("api_specs", [])
    all_api_tools = create_dynamic_tools(api_specs)
    repl_tool     = get_coder_tool()
    tools         = all_api_tools + [repl_tool]

    error_context = state.get("error_context") or "None"
    system = CODER_AGENT_SYSTEM_PROMPT.format(error_context=error_context)
    return await _run_react_agent(state, tools, system)


# ════════════════════════════════════════════════════════════════════════════
#  VALIDATORY
# ════════════════════════════════════════════════════════════════════════════

async def validatory_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the specialist agent's response.
    If INVALID, stores validation_comments so the supervisor can retry.
    """
    original_query = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "Unknown query",
    )

    llm = get_llm()
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

    return {
        "validation_comments": result.validation_comments,
        "error_context":       result.validation_comments,
    }
