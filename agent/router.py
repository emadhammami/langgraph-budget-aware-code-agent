"""Guardrail router for the LangGraph conditional edges.

This single function is attached to every conditional edge in the graph.
Before honouring the ``next_step`` field set by the preceding node it checks
whether either hard budget limit has been breached.  If so, it overrides the
routing destination with ``"early_exit"``, giving the graph a safe and
observable stopping condition regardless of how far the loop has progressed.
"""

from __future__ import annotations

from .state import AgentState


def guardrail_router(state: AgentState) -> str:
    """
    Return the next node name, enforcing token-budget and loop-count limits.

    Hard limits (checked in order)
    --------------------------------
    1. ``loop_count > max_loops``        – too many planner–executor–critic cycles.
    2. ``total_tokens > max_tokens_budget`` – cumulative LLM token spend exceeded.

    If either limit is exceeded the router returns ``"early_exit"`` instead of
    the node's requested ``next_step``, short-circuiting the loop safely.
    """
    if state.loop_count > state.max_loops or state.total_tokens > state.max_tokens_budget:
        return "early_exit"
    return state.next_step
