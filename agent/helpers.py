"""Utility functions shared across agent nodes.

Responsibilities
----------------
- Token estimation  – fallback when the LLM client omits usage metadata.
- Metrics recording – appends a telemetry snapshot after each node runs.
- Update finaliser  – attaches metrics and emits a structured log entry.
- Region extraction – discovers function boundaries via AST walking.
- Line-range helpers – parse region labels and slice file lines safely.
- Syntax validation – lightweight AST check used by the Critic node.
- LLM client        – constructs the Gemini client used by all nodes.
- LLM JSON invoke   – calls the LLM and parses JSON with fallback handling.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState

logger = logging.getLogger("phd_demo_agent")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(*texts: str) -> int:
    """
    Approximate token count using the ~4 characters-per-token heuristic.

    Used only as a fallback when the LLM response does not include
    ``usage_metadata``.  Not intended to be precise.
    """
    joined = " ".join(t for t in texts if t)
    return max(1, len(joined) // 4) if joined else 0


# ---------------------------------------------------------------------------
# Telemetry / metrics
# ---------------------------------------------------------------------------

def record_metrics(state: AgentState, node_name: str) -> List[Dict[str, Any]]:
    """Append a telemetry snapshot for the current step and return the full history."""
    return state.metrics_history + [
        {
            "step": len(state.metrics_history) + 1,
            "node": node_name,
            "loop_count": state.loop_count,
            "total_tokens": state.total_tokens,
            "accepted_fixes": len(state.accepted_fixes),
            "analyzed_regions": len(state.analyzed_regions),
            "budget_used_ratio": state.total_tokens / max(1, state.max_tokens_budget),
        }
    ]


def finalize_update(
    state: AgentState, update: Dict[str, Any], node_name: str
) -> Dict[str, Any]:
    """
    Attach updated metrics to a node's return dict and emit a structured log
    snapshot.  Every node must call this before returning so that telemetry
    is consistent regardless of which code path was taken.
    """
    temp_state = state.model_copy(update=update)
    update["metrics_history"] = record_metrics(temp_state, node_name)

    snapshot = {
        "node": node_name,
        "loop_count": temp_state.loop_count,
        "total_tokens": temp_state.total_tokens,
        "accepted_fixes": len(temp_state.accepted_fixes),
        "analyzed_regions": temp_state.analyzed_regions,
        "next_step": temp_state.next_step,
    }
    logger.info(json.dumps(snapshot, indent=2))
    return update


# ---------------------------------------------------------------------------
# Code region extraction
# ---------------------------------------------------------------------------

def extract_candidate_regions(file_content: str) -> List[str]:
    """
    Walk the AST to discover all function definitions and return them as
    labelled line ranges, e.g. ``"divide:lines 1-2"``.

    Falls back to ``"module:lines 1-end"`` if parsing fails (syntax errors in
    the input file) or no functions are found (e.g. pure script files).
    """
    regions: List[str] = []
    try:
        tree = ast.parse(file_content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", start)
                if start and end:
                    regions.append(f"{node.name}:lines {start}-{end}")
    except SyntaxError:
        regions.append("module:lines 1-end")

    return regions or ["module:lines 1-end"]


def parse_region_to_line_range(region: str, total_lines: int) -> Tuple[int, int]:
    """
    Parse a region label of the form ``"name:lines A-B"`` into ``(A, B)``.

    ``B`` is replaced with ``total_lines`` when the label contains ``"end"``.
    Falls back to ``(1, 50)`` on any parsing error.
    """
    try:
        tail = region.split("lines ", 1)[1]
        a, b = tail.split("-", 1)
        return int(a), total_lines if b == "end" else int(b)
    except Exception:
        return 1, min(total_lines, 50)


def safe_slice(lines: List[str], start: int, end: int) -> str:
    """Return lines ``start``–``end`` (1-indexed, inclusive) joined as a string."""
    start = max(1, start)
    end = min(len(lines), end)
    return "\n".join(lines[start - 1 : end])


# ---------------------------------------------------------------------------
# Syntax validation
# ---------------------------------------------------------------------------

def try_ast_check(source: str) -> Tuple[bool, str]:
    """
    Attempt to parse ``source`` with the standard AST.

    Returns ``(True, message)`` on success or ``(False, error)`` on failure.
    This gives the Critic a local, deterministic signal before making an LLM
    call, which can save tokens when the proposed fix is syntactically broken.
    """
    try:
        ast.parse(source)
        return True, "AST parse succeeded."
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def build_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini LLM client used by all agent nodes."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ---------------------------------------------------------------------------
# LLM invocation with JSON parsing
# ---------------------------------------------------------------------------

def llm_json_invoke(
    model: ChatGoogleGenerativeAI,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[Dict[str, Any], int]:
    """
    Invoke the LLM and attempt to parse the response as JSON.

    Handles two common LLM response formats:
    - Clean JSON string.
    - JSON wrapped in a markdown code fence (``` or ```json).

    Falls back to ``{"raw_text": <content>}`` if both parses fail.

    Also extracts token usage from ``usage_metadata`` when present,
    falling back to the character-count heuristic.

    Returns
    -------
    data : dict
        Parsed JSON payload.
    total_tokens : int
        Tokens consumed by this call.
    """
    response = model.invoke([("system", system_prompt), ("human", user_prompt)])
    content = response.content if isinstance(response.content, str) else str(response.content)

    usage = getattr(response, "usage_metadata", None) or {}
    total_tokens = usage.get("total_tokens")
    if not isinstance(total_tokens, int):
        total_tokens = estimate_tokens(system_prompt, user_prompt, content)

    try:
        data = json.loads(content)
    except Exception:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json\n", "", 1).strip()
        try:
            data = json.loads(cleaned)
        except Exception:
            data = {"raw_text": content}

    return data, total_tokens
