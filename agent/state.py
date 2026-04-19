"""Pydantic state models shared across every node in the agent graph.

The ``AgentState`` class is the single source of truth for the graph's
mutable data.  Every node receives the current state and returns a partial
dict; LangGraph merges the returned dict back into the state before routing
to the next node.

Budget guardrails
-----------------
``loop_count`` is incremented once per planner call.
``total_tokens`` is incremented by the token cost reported (or estimated) for
each LLM invocation.  The ``guardrail_router`` checks both values against
``max_loops`` and ``max_tokens_budget`` before every transition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class AnalysisPlan(BaseModel):
    """Output produced by the Planner node for a single iteration."""

    target_region: str = ""
    rationale: str = ""
    bug_hypothesis: str = ""


class FixProposal(BaseModel):
    """Output produced by the Executor node for a single region."""

    summary: str = ""
    fixed_snippet: str = ""
    explanation: str = ""


class CritiqueResult(BaseModel):
    """Output produced by the Critic node for a single fix proposal."""

    valid: bool = False
    reason: str = ""


class AgentState(BaseModel):
    """Shared state that flows through every node in the LangGraph graph."""

    # ---- input ---------------------------------------------------------------
    file_path: str

    # ---- parsed file ---------------------------------------------------------
    file_content: str = ""
    file_lines: List[str] = Field(default_factory=list)

    # ---- analysis bookkeeping ------------------------------------------------
    candidate_regions: List[str] = Field(default_factory=list)
    analyzed_regions: List[str] = Field(default_factory=list)
    accepted_fixes: List[str] = Field(default_factory=list)

    # ---- latest node outputs -------------------------------------------------
    planner_output: AnalysisPlan = Field(default_factory=AnalysisPlan)
    executor_output: FixProposal = Field(default_factory=FixProposal)
    critic_output: CritiqueResult = Field(default_factory=CritiqueResult)

    # ---- telemetry -----------------------------------------------------------
    metrics_history: List[Dict[str, Any]] = Field(default_factory=list)

    # ---- routing control -----------------------------------------------------
    next_step: Literal["planner", "executor", "critic", "early_exit", "finish"] = "planner"
    final_message: str = ""
    last_node: str = ""

    # ---- budget guardrails ---------------------------------------------------
    total_tokens: int = 0
    loop_count: int = 0
    max_loops: int = 4
    max_tokens_budget: int = 8000
