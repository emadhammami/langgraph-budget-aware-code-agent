"""Tests for the validation_node LangGraph node.

The validation node executes candidate code via the sandbox and attaches
results to AgentState.  These tests mock ``run_code`` to avoid real subprocess
overhead and to exercise the node's state-management logic in isolation.

All tests require no external APIs or network access.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from agent.nodes import validation_node
from agent.state import AgentState, FixProposal, ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(**kwargs: Any) -> AgentState:
    """Return a minimal AgentState suitable for calling validation_node."""
    defaults: Dict[str, Any] = {
        "file_path": "dummy.py",
        "executor_output": FixProposal(
            summary="Fix divide",
            fixed_snippet="def divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
            explanation="Guard against division by zero.",
        ),
        "validation_failures": 0,
    }
    defaults.update(kwargs)
    return AgentState(**defaults)


def _mock_success() -> Dict[str, Any]:
    return {
        "success": True,
        "stdout": "All tests passed\n",
        "stderr": "",
        "runtime_seconds": 0.12,
        "timed_out": False,
        "error_category": "none",
    }


def _mock_failure(error_category: str = "runtime_error") -> Dict[str, Any]:
    return {
        "success": False,
        "stdout": "",
        "stderr": "ZeroDivisionError: division by zero\n",
        "runtime_seconds": 0.08,
        "timed_out": False,
        "error_category": error_category,
    }


def _mock_timeout() -> Dict[str, Any]:
    return {
        "success": False,
        "stdout": "",
        "stderr": "",
        "runtime_seconds": 1.01,
        "timed_out": True,
        "error_category": "timeout",
    }


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestValidationNodeSuccess:
    def test_routes_to_critic(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        assert result["next_step"] == "critic"

    def test_last_node_is_validation(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        assert result["last_node"] == "validation"

    def test_validation_result_success_true(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        vr: ValidationResult = result["validation_result"]
        assert vr.success is True

    def test_stdout_captured(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        assert "All tests passed" in result["validation_result"].stdout

    def test_error_category_none_on_success(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        assert result["validation_result"].error_category == "none"

    def test_validation_failures_not_incremented_on_success(self) -> None:
        state = _make_state(validation_failures=2)
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        assert result["validation_failures"] == 2

    def test_runtime_seconds_stored(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        assert result["validation_result"].runtime_seconds == pytest.approx(0.12)

    def test_metrics_history_appended(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        assert len(result["metrics_history"]) == 1
        assert result["metrics_history"][0]["node"] == "validation"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestValidationNodeFailure:
    def test_validation_result_success_false(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_failure()):
            result = validation_node(state)
        assert result["validation_result"].success is False

    def test_stderr_captured(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_failure()):
            result = validation_node(state)
        assert "ZeroDivisionError" in result["validation_result"].stderr

    def test_validation_failures_incremented(self) -> None:
        state = _make_state(validation_failures=0)
        with patch("agent.nodes.run_code", return_value=_mock_failure()):
            result = validation_node(state)
        assert result["validation_failures"] == 1

    def test_validation_failures_cumulative(self) -> None:
        state = _make_state(validation_failures=3)
        with patch("agent.nodes.run_code", return_value=_mock_failure()):
            result = validation_node(state)
        assert result["validation_failures"] == 4

    def test_error_category_runtime_error(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_failure("runtime_error")):
            result = validation_node(state)
        assert result["validation_result"].error_category == "runtime_error"

    def test_still_routes_to_critic_on_failure(self) -> None:
        """Failures are evidence for the Critic, not a reason to skip it."""
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_failure()):
            result = validation_node(state)
        assert result["next_step"] == "critic"


# ---------------------------------------------------------------------------
# Timeout path
# ---------------------------------------------------------------------------


class TestValidationNodeTimeout:
    def test_timed_out_flag_propagated(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_timeout()):
            result = validation_node(state)
        assert result["validation_result"].timed_out is True

    def test_timeout_error_category_propagated(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_timeout()):
            result = validation_node(state)
        assert result["validation_result"].error_category == "timeout"

    def test_validation_failures_incremented_on_timeout(self) -> None:
        state = _make_state(validation_failures=1)
        with patch("agent.nodes.run_code", return_value=_mock_timeout()):
            result = validation_node(state)
        assert result["validation_failures"] == 2

    def test_routes_to_critic_on_timeout(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_timeout()):
            result = validation_node(state)
        assert result["next_step"] == "critic"


# ---------------------------------------------------------------------------
# State update shape
# ---------------------------------------------------------------------------


class TestValidationNodeStateShape:
    def test_all_expected_keys_present(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        for key in ("validation_result", "validation_failures", "next_step", "last_node", "metrics_history"):
            assert key in result, f"Missing key: {key}"

    def test_validation_result_is_pydantic_model(self) -> None:
        state = _make_state()
        with patch("agent.nodes.run_code", return_value=_mock_success()):
            result = validation_node(state)
        assert isinstance(result["validation_result"], ValidationResult)
