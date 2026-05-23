"""
State definition for the Smart API Agent system.
Uses TypedDict with Annotated reducers for safe concurrent state updates.
"""
import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence
from langchain_core.messages import BaseMessage


# ── API Specification contract ──────────────────────────────────────────────

class ApiSpec(Dict[str, Any]):
    """
    Shape of a user-provided API spec entry:
    {
        "name":        str,          # unique tool name
        "description": str,          # human-readable purpose
        "method":      "GET"|"POST"|"DELETE"|"PUT"|"PATCH",
        "url":         str,          # full URL (may contain {path_params})
        "headers":     dict,         # static headers (e.g. auth tokens)
        "query_params": { field: type_hint },   # optional query params
        "request_body": { field: type_hint },   # optional body fields
        "path_params":  [str],       # names of path-level params
        "lookup_fields": {           # fields that need a lookup before calling
            "department_id": {
                "lookup_api": "get_departments",
                "match_field": "name",
                "return_field": "id"
            }
        }
    }
    """


# ── Workflow-level state ─────────────────────────────────────────────────────

class AgentState(dict):
    """
    Central state object threaded through every LangGraph node.
    All list/message fields use operator.add so parallel branches merge safely.
    """
    messages:            Annotated[Sequence[BaseMessage], operator.add]
    company_info:        str                          # RAG knowledge base
    api_specs:           List[Dict[str, Any]]         # user-provided specs
    current_intent:      Literal["rag", "api"]        # decider output
    api_type:            Literal["get", "post", "delete", "put", "patch",
                                 "complex_coder", "unknown"]
    missing_information: List[str]                    # fields still needed
    collected_fields:    Dict[str, Any]               # fields gathered so far
    lookup_needed:       List[Dict[str, Any]]         # pending lookup tasks
    validation_comments: str                          # validatory feedback
    supervisor_decision: Literal[                     # supervisor routing
        "get_agent", "create_agent", "delete_agent",
        "coder_agent", "end"
    ]
    retry_count:         int                          # guard-rail retry counter
    guardrail_blocked:   bool                         # hard-stop flag
    error_context:       Optional[str]                # last error for coder
    conversation_id:     str                          # for checkpoint isolation
