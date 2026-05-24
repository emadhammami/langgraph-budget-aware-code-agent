"""Tests for AgentState and its sub-models (AnalysisPlan, FixProposal, CritiqueResult).

All tests are purely in-memory — no LLM calls, no file I/O.
"""

from __future__ import annotations

import pytest

from agent.state import AgentState, AnalysisPlan, CritiqueResult, FixProposal


class TestAnalysisPlan:
    def test_defaults(self) -> None:
        plan = AnalysisPlan()
        assert plan.target_region == ""
        assert plan.rationale == ""
        assert plan.bug_hypothesis == ""

    def test_set_fields(self) -> None:
        plan = AnalysisPlan(
            target_region="foo:lines 1-5",
            rationale="suspicious null check",
            bug_hypothesis="possible NoneType error",
        )
        assert plan.target_region == "foo:lines 1-5"
        assert plan.rationale == "suspicious null check"
        assert plan.bug_hypothesis == "possible NoneType error"


class TestFixProposal:
    def test_defaults(self) -> None:
        fp = FixProposal()
        assert fp.summary == ""
        assert fp.fixed_snippet == ""
        assert fp.explanation == ""

    def test_set_fields(self) -> None:
        fp = FixProposal(summary="s", fixed_snippet="x = 1", explanation="e")
        assert fp.summary == "s"
        assert fp.fixed_snippet == "x = 1"
        assert fp.explanation == "e"


class TestCritiqueResult:
    def test_defaults(self) -> None:
        cr = CritiqueResult()
        assert cr.valid is False
        assert cr.reason == ""

    def test_accepted(self) -> None:
        cr = CritiqueResult(valid=True, reason="looks correct")
        assert cr.valid is True
        assert cr.reason == "looks correct"


class TestAgentState:
    def test_requires_file_path(self) -> None:
        with pytest.raises(Exception):
            AgentState()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        state = AgentState(file_path="some/file.py")
        assert state.loop_count == 0
        assert state.total_tokens == 0
        assert state.max_loops == 4
        assert state.max_tokens_budget == 8000
        assert state.next_step == "planner"
        assert state.candidate_regions == []
        assert state.analyzed_regions == []
        assert state.accepted_fixes == []
        assert state.metrics_history == []
        assert state.file_content == ""
        assert state.final_message == ""

    def test_file_path_stored(self) -> None:
        state = AgentState(file_path="foo.py")
        assert state.file_path == "foo.py"

    def test_custom_budget_params(self) -> None:
        state = AgentState(file_path="x.py", max_loops=10, max_tokens_budget=20_000)
        assert state.max_loops == 10
        assert state.max_tokens_budget == 20_000

    def test_model_copy_with_update_is_immutable(self) -> None:
        """model_copy(update=...) must not mutate the original."""
        state = AgentState(file_path="x.py")
        updated = state.model_copy(update={"loop_count": 3})
        assert updated.loop_count == 3
        assert state.loop_count == 0  # original must be unchanged

    def test_next_step_literal_values(self) -> None:
        """next_step must only accept the documented literals."""
        for valid in ("planner", "executor", "critic", "early_exit", "finish"):
            s = AgentState(file_path="x.py", next_step=valid)  # type: ignore[arg-type]
            assert s.next_step == valid

    def test_invalid_next_step_raises(self) -> None:
        with pytest.raises(Exception):
            AgentState(file_path="x.py", next_step="unknown_node")  # type: ignore[arg-type]
