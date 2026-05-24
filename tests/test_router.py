"""Tests for guardrail_router — the central routing function.

guardrail_router must:
  - pass through state.next_step when both budgets are within limits
  - override to "early_exit" when loop_count > max_loops
  - override to "early_exit" when total_tokens > max_tokens_budget
  - allow transitions at the *exact* limit (> not >=)
"""

from __future__ import annotations

from agent.router import guardrail_router
from agent.state import AgentState


def _state(**kwargs) -> AgentState:
    """Helper: build an AgentState with sane defaults, overriding with kwargs."""
    return AgentState(file_path="x.py", **kwargs)


class TestGuardrailRouter:
    def test_routes_to_executor_normally(self) -> None:
        state = _state(next_step="executor", loop_count=1, total_tokens=100)
        assert guardrail_router(state) == "executor"

    def test_routes_to_planner_normally(self) -> None:
        state = _state(next_step="planner", loop_count=0, total_tokens=0)
        assert guardrail_router(state) == "planner"

    def test_routes_to_critic_normally(self) -> None:
        state = _state(next_step="critic", loop_count=2, total_tokens=500)
        assert guardrail_router(state) == "critic"

    def test_routes_to_finish_normally(self) -> None:
        state = _state(next_step="finish", loop_count=3, total_tokens=1000)
        assert guardrail_router(state) == "finish"

    # -- loop_count guard ---------------------------------------------------

    def test_overrides_when_loop_count_exceeded(self) -> None:
        state = _state(next_step="executor", loop_count=5, max_loops=4)
        assert guardrail_router(state) == "early_exit"

    def test_allows_at_exact_loop_limit(self) -> None:
        """loop_count == max_loops must NOT trigger early exit (> not >=)."""
        state = _state(next_step="critic", loop_count=4, max_loops=4)
        assert guardrail_router(state) == "critic"

    def test_overrides_when_loop_count_far_exceeded(self) -> None:
        state = _state(next_step="planner", loop_count=100, max_loops=4)
        assert guardrail_router(state) == "early_exit"

    # -- total_tokens guard -------------------------------------------------

    def test_overrides_when_tokens_exceeded(self) -> None:
        state = _state(next_step="planner", total_tokens=9000, max_tokens_budget=8000)
        assert guardrail_router(state) == "early_exit"

    def test_allows_at_exact_token_limit(self) -> None:
        """total_tokens == max_tokens_budget must NOT trigger early exit (> not >=)."""
        state = _state(next_step="executor", total_tokens=8000, max_tokens_budget=8000)
        assert guardrail_router(state) == "executor"

    # -- both guards exceeded simultaneously --------------------------------

    def test_early_exit_when_both_limits_exceeded(self) -> None:
        state = _state(
            next_step="finish",
            loop_count=10,
            max_loops=4,
            total_tokens=10_000,
            max_tokens_budget=8000,
        )
        assert guardrail_router(state) == "early_exit"

    # -- custom budget parameters ------------------------------------------

    def test_respects_custom_max_loops(self) -> None:
        state = _state(next_step="planner", loop_count=3, max_loops=2)
        assert guardrail_router(state) == "early_exit"

    def test_respects_custom_max_tokens(self) -> None:
        state = _state(next_step="critic", total_tokens=501, max_tokens_budget=500)
        assert guardrail_router(state) == "early_exit"
