"""Smoke tests for graph assembly and full-pipeline execution.

No real LLM calls are made here. The LLM-facing internals of every node
(build_llm, llm_json_invoke) are replaced with deterministic stubs so the
graph wiring, routing logic, and state transitions can be verified in CI
without requiring a GOOGLE_API_KEY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.graph import build_graph, run_agent


# ---------------------------------------------------------------------------
# Deterministic stubs
# ---------------------------------------------------------------------------

_PLANNER_RESPONSE = (
    {
        "target_region": "add:lines 1-2",
        "rationale": "simple arithmetic — chosen for demo",
        "bug_hypothesis": "no guard against non-numeric input",
    },
    50,
)

_EXECUTOR_RESPONSE = (
    {
        "summary": "Add type check",
        "fixed_snippet": "def add(a, b):\n    if not isinstance(a, (int, float)):\n        raise TypeError\n    return a + b",
        "explanation": "Validates input types before arithmetic.",
    },
    60,
)

_CRITIC_RESPONSE = (
    {
        "valid": True,
        "reason": "Fix is syntactically correct and addresses the hypothesis.",
    },
    40,
)


def _mock_llm_json_invoke(model: Any, system_prompt: str, user_prompt: str) -> tuple:  # noqa: ANN401
    """Rotate through canned responses based on node hint in system_prompt."""
    prompt_lower = system_prompt.lower()
    if "planner" in prompt_lower:
        return _PLANNER_RESPONSE
    if "executor" in prompt_lower:
        return _EXECUTOR_RESPONSE
    if "critic" in prompt_lower:
        return _CRITIC_RESPONSE
    # Fallback — should not be reached in normal flow
    return ({"raw_text": "unexpected"}, 10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_returns_compiled_graph(self) -> None:
        """build_graph() must succeed without touching any LLM."""
        graph = build_graph()
        assert graph is not None

    def test_compiled_graph_is_reusable(self) -> None:
        """Calling build_graph() twice must return independent instances."""
        g1 = build_graph()
        g2 = build_graph()
        assert g1 is not g2


class TestRunAgentFileNotFound:
    def test_missing_file_returns_error_message(self) -> None:
        """run_agent on a missing file must terminate gracefully."""
        result = run_agent("__nonexistent_file_xyz__.py")
        assert "final_message" in result
        assert "not found" in result["final_message"].lower()

    def test_missing_file_has_metrics(self) -> None:
        """Even the error path must produce at least one metrics entry."""
        result = run_agent("__nonexistent__.py")
        assert isinstance(result["metrics_history"], list)
        assert len(result["metrics_history"]) >= 1


class TestRunAgentWithMockedLLM:
    """End-to-end graph execution with LLM calls stubbed out."""

    @pytest.fixture
    def python_file(self, tmp_path: Path) -> Path:
        code = "def add(a, b):\n    return a + b\n"
        p = tmp_path / "target.py"
        p.write_text(code, encoding="utf-8")
        return p

    def test_full_run_produces_final_message(self, python_file: Path) -> None:
        with (
            patch("agent.nodes.llm_json_invoke", side_effect=_mock_llm_json_invoke),
            patch("agent.nodes.build_llm", return_value=MagicMock()),
        ):
            result = run_agent(str(python_file))

        assert "final_message" in result
        assert isinstance(result["final_message"], str)
        assert len(result["final_message"]) > 0

    def test_full_run_produces_metrics_history(self, python_file: Path) -> None:
        with (
            patch("agent.nodes.llm_json_invoke", side_effect=_mock_llm_json_invoke),
            patch("agent.nodes.build_llm", return_value=MagicMock()),
        ):
            result = run_agent(str(python_file))

        assert isinstance(result["metrics_history"], list)
        assert len(result["metrics_history"]) > 0

    def test_metrics_history_entries_have_required_keys(self, python_file: Path) -> None:
        with (
            patch("agent.nodes.llm_json_invoke", side_effect=_mock_llm_json_invoke),
            patch("agent.nodes.build_llm", return_value=MagicMock()),
        ):
            result = run_agent(str(python_file))

        required_keys = {"step", "node", "loop_count", "total_tokens", "budget_used_ratio"}
        for entry in result["metrics_history"]:
            assert required_keys.issubset(entry.keys()), f"Missing keys in: {entry}"

    def test_guardrail_budget_enforcement(self, tmp_path: Path) -> None:
        """A tight token budget must trigger early_exit before all regions are analysed."""
        code = "def a():\n    pass\n\ndef b():\n    pass\n\ndef c():\n    pass\n"
        p = tmp_path / "multi.py"
        p.write_text(code, encoding="utf-8")

        # Very low token budget — should trigger early_exit after first planner call
        with (
            patch("agent.nodes.llm_json_invoke", side_effect=_mock_llm_json_invoke),
            patch("agent.nodes.build_llm", return_value=MagicMock()),
        ):
            from agent.state import AgentState
            from agent.graph import build_graph

            app = build_graph()
            initial = AgentState(file_path=str(p), max_tokens_budget=10)  # tiny budget
            result = app.invoke(initial)

        assert "early exit" in result["final_message"].lower()
