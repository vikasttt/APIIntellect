"""
Dynamic tool factory for the Smart API Agent system.

Key features:
- Fully async HTTP calls via httpx
- Headers forwarded from API spec (e.g. auth tokens)
- Path-parameter substitution
- Lookup tool for ID resolution (name → id)
- PythonREPLTool wrapped for safe coder usage
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool
from langchain_experimental.tools import PythonREPLTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

# ── Async HTTP helper ────────────────────────────────────────────────────────

async def _call_api(
    method: str,
    url: str,
    headers: Dict[str, str],
    query_params: Optional[Dict[str, Any]] = None,
    request_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Execute an HTTP request and return text response or error string."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=query_params or {},
                json=request_body,
            )
            response.raise_for_status()
            # Try to pretty-print JSON, fall back to raw text
            try:
                return json.dumps(response.json(), indent=2)
            except Exception:
                return response.text
        except httpx.HTTPStatusError as e:
            return (
                f"API_ERROR status={e.response.status_code} "
                f"body={e.response.text[:500]}"
            )
        except Exception as e:
            return f"API_ERROR {type(e).__name__}: {str(e)}"


# ── Pydantic input schema builder ────────────────────────────────────────────

_TYPE_MAP: Dict[str, Any] = {
    "string": (str, ...),
    "str":    (str, ...),
    "int":    (int, ...),
    "integer":(int, ...),
    "float":  (float, ...),
    "bool":   (bool, ...),
    "boolean":(bool, ...),
    "dict":   (Dict[str, Any], ...),
    "list":   (List[Any], ...),
}

def _build_input_model(
    name: str,
    query_params: Dict[str, str],
    request_body: Dict[str, str],
    path_params: List[str],
) -> type[BaseModel]:
    """Build a typed Pydantic model for tool inputs from the spec."""
    fields: Dict[str, Any] = {}

    for param in path_params:
        fields[param] = (str, Field(description=f"Path parameter: {param}"))

    for field, ftype in query_params.items():
        py_type, _ = _TYPE_MAP.get(str(ftype).lower(), (str, ...))
        fields[field] = (Optional[py_type], Field(default=None, description=f"Query param: {field}"))

    for field, ftype in request_body.items():
        py_type, _ = _TYPE_MAP.get(str(ftype).lower(), (str, ...))
        fields[field] = (Optional[py_type], Field(default=None, description=f"Body field: {field}"))

    return create_model(f"{name}Input", **fields)


# ── Tool factory ─────────────────────────────────────────────────────────────

def create_dynamic_tools(api_specs: List[Dict[str, Any]]) -> List[StructuredTool]:
    """
    Generate a list of async LangChain StructuredTools from API spec dicts.

    Each spec can contain:
      name, description, method, url, headers,
      query_params, request_body, path_params, lookup_fields
    """
    tools: List[StructuredTool] = []

    for spec in api_specs:
        name        = spec.get("name", "unnamed_api")
        description = spec.get("description", "No description provided.")
        method      = spec.get("method", "GET").upper()
        base_url    = spec.get("url", "")
        headers     = spec.get("headers", {})
        q_params    = spec.get("query_params", {})
        body_fields = spec.get("request_body", {})
        path_params = spec.get("path_params", [])

        # Build input schema
        InputModel = _build_input_model(name, q_params, body_fields, path_params)

        # Close over spec values so each lambda is independent
        def _make_func(
            _method=method,
            _base_url=base_url,
            _headers=headers,
            _path_params=path_params,
            _q_fields=list(q_params.keys()),
            _body_fields=list(body_fields.keys()),
        ):
            async def api_func(**kwargs) -> str:
                # Substitute path params
                url = _base_url
                for pp in _path_params:
                    val = kwargs.pop(pp, None)
                    if val is not None:
                        url = url.replace(f"{{{pp}}}", str(val))

                query = {k: v for k in _q_fields if (v := kwargs.get(k)) is not None}
                body  = {k: v for k in _body_fields if (v := kwargs.get(k)) is not None}

                return await _call_api(_method, url, _headers, query, body or None)

            return api_func

        tool_func = _make_func()
        tool_func.__name__ = name

        tool = StructuredTool.from_function(
            coroutine=tool_func,
            name=name,
            description=description,
            args_schema=InputModel,
        )
        tools.append(tool)
        logger.debug("Registered tool: %s [%s]", name, method)

    return tools


# ── Lookup tool ──────────────────────────────────────────────────────────────

def create_lookup_tool(
    lookup_spec: Dict[str, Any],
    api_specs: List[Dict[str, Any]],
) -> Optional[StructuredTool]:
    """
    Build a tool that fetches a list from 'lookup_api' and resolves
    display_name → id using match_field / return_field.
    """
    lookup_api_name = lookup_spec.get("lookup_api")
    match_field     = lookup_spec.get("match_field", "name")
    return_field    = lookup_spec.get("return_field", "id")

    target_spec = next(
        (s for s in api_specs if s.get("name") == lookup_api_name), None
    )
    if not target_spec:
        return None

    base_url = target_spec.get("url", "")
    headers  = target_spec.get("headers", {})

    class LookupInput(BaseModel):
        display_value: str = Field(description=f"The {match_field} to search for.")

    async def lookup_func(display_value: str) -> str:
        raw = await _call_api("GET", base_url, headers)
        try:
            data = json.loads(raw)
            items = data if isinstance(data, list) else data.get("results", data.get("items", []))
            for item in items:
                if str(item.get(match_field, "")).lower() == display_value.lower():
                    return str(item.get(return_field, ""))
            return f"LOOKUP_NOT_FOUND: No match for '{display_value}' in {lookup_api_name}"
        except Exception as e:
            return f"LOOKUP_ERROR: {e}"

    lookup_func.__name__ = f"lookup_{lookup_api_name}"

    return StructuredTool.from_function(
        coroutine=lookup_func,
        name=f"lookup_{lookup_api_name}",
        description=(
            f"Look up the {return_field} of a record in {lookup_api_name} "
            f"by matching its {match_field}."
        ),
        args_schema=LookupInput,
    )


# ── Coder tool ───────────────────────────────────────────────────────────────

def get_coder_tool() -> PythonREPLTool:
    """Return a sandboxed Python REPL for data aggregation tasks."""
    return PythonREPLTool(
        description=(
            "Execute Python code to aggregate, filter, group, or compute values "
            "from data already fetched by API tools. "
            "No file I/O. No additional network calls."
        )
    )
