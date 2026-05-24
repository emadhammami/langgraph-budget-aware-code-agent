"""Tests for deterministic helper functions in agent.helpers.

Only functions that do NOT require a live LLM are tested here:
  - estimate_tokens
  - extract_candidate_regions
  - parse_region_to_line_range
  - safe_slice
  - try_ast_check
  - record_metrics
  - finalize_update (partial — log emission is not verified, only return shape)

LLM-dependent helpers (build_llm, llm_json_invoke) are covered via mocks
in test_graph_smoke.py.
"""

from __future__ import annotations

import textwrap

from agent.helpers import (
    estimate_tokens,
    extract_candidate_regions,
    finalize_update,
    parse_region_to_line_range,
    record_metrics,
    safe_slice,
    try_ast_check,
)
from agent.state import AgentState


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_returns_zero(self) -> None:
        assert estimate_tokens() == 0

    def test_empty_string_returns_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_four_chars_is_one_token(self) -> None:
        # 4 characters → ceiling(4/4) = 1 token
        assert estimate_tokens("abcd") == 1

    def test_eight_chars_is_two_tokens(self) -> None:
        assert estimate_tokens("abcdefgh") == 2

    def test_multiple_strings_are_joined(self) -> None:
        # "aaaa" + " " + "bbbb" = 9 chars → at least 2 tokens
        result = estimate_tokens("aaaa", "bbbb")
        assert result >= 2

    def test_nonempty_returns_at_least_one(self) -> None:
        assert estimate_tokens("x") >= 1

    def test_filters_empty_strings(self) -> None:
        # Empty strings must be ignored, not contribute a separator space
        result_with = estimate_tokens("abcdefgh", "")
        result_without = estimate_tokens("abcdefgh")
        assert result_with == result_without


# ---------------------------------------------------------------------------
# extract_candidate_regions
# ---------------------------------------------------------------------------


class TestExtractCandidateRegions:
    def test_single_function(self) -> None:
        code = "def foo():\n    pass\n"
        regions = extract_candidate_regions(code)
        assert len(regions) == 1
        assert regions[0].startswith("foo:lines")

    def test_multiple_functions_all_detected(self) -> None:
        code = textwrap.dedent("""\
            def add(a, b):
                return a + b

            def subtract(a, b):
                return a - b

            def multiply(a, b):
                return a * b
        """)
        regions = extract_candidate_regions(code)
        names = {r.split(":")[0] for r in regions}
        assert names == {"add", "subtract", "multiply"}

    def test_async_function_detected(self) -> None:
        code = "async def fetch():\n    pass\n"
        regions = extract_candidate_regions(code)
        assert any(r.startswith("fetch:") for r in regions)

    def test_syntax_error_falls_back_to_module(self) -> None:
        regions = extract_candidate_regions("def bad(:\n  pass")
        assert regions == ["module:lines 1-end"]

    def test_no_functions_falls_back_to_module(self) -> None:
        regions = extract_candidate_regions("x = 1\ny = 2\n")
        assert regions == ["module:lines 1-end"]

    def test_empty_source_falls_back(self) -> None:
        regions = extract_candidate_regions("")
        assert regions == ["module:lines 1-end"]

    def test_region_format_contains_line_numbers(self) -> None:
        code = "def alpha():\n    pass\n"
        regions = extract_candidate_regions(code)
        # format: "name:lines A-B"
        assert ":lines " in regions[0]
        tail = regions[0].split(":lines ")[1]
        start, end = tail.split("-")
        assert int(start) >= 1
        assert int(end) >= int(start)


# ---------------------------------------------------------------------------
# parse_region_to_line_range
# ---------------------------------------------------------------------------


class TestParseRegionToLineRange:
    def test_normal_range(self) -> None:
        assert parse_region_to_line_range("foo:lines 3-8", 20) == (3, 8)

    def test_first_line(self) -> None:
        assert parse_region_to_line_range("bar:lines 1-1", 10) == (1, 1)

    def test_end_keyword_substituted_with_total(self) -> None:
        assert parse_region_to_line_range("module:lines 1-end", 15) == (1, 15)

    def test_bad_format_falls_back(self) -> None:
        start, end = parse_region_to_line_range("garbage", 100)
        assert start == 1
        assert end == 50

    def test_bad_format_caps_at_total_lines_when_small(self) -> None:
        start, end = parse_region_to_line_range("garbage", 30)
        assert end <= 30

    def test_missing_lines_keyword_falls_back(self) -> None:
        start, end = parse_region_to_line_range("foo:5-10", 50)
        assert start == 1  # fallback


# ---------------------------------------------------------------------------
# safe_slice
# ---------------------------------------------------------------------------


class TestSafeSlice:
    LINES = ["line1", "line2", "line3", "line4", "line5"]

    def test_normal_slice(self) -> None:
        assert safe_slice(self.LINES, 2, 4) == "line2\nline3\nline4"

    def test_single_line(self) -> None:
        assert safe_slice(self.LINES, 1, 1) == "line1"

    def test_full_range(self) -> None:
        assert safe_slice(self.LINES, 1, 5) == "\n".join(self.LINES)

    def test_clamps_start_below_one(self) -> None:
        result = safe_slice(self.LINES, 0, 2)
        assert result.startswith("line1")

    def test_clamps_end_beyond_length(self) -> None:
        result = safe_slice(self.LINES, 4, 100)
        assert result == "line4\nline5"

    def test_empty_list(self) -> None:
        assert safe_slice([], 1, 5) == ""


# ---------------------------------------------------------------------------
# try_ast_check
# ---------------------------------------------------------------------------


class TestTryAstCheck:
    def test_valid_code_returns_true(self) -> None:
        ok, msg = try_ast_check("x = 1\n")
        assert ok is True
        assert "succeeded" in msg.lower()

    def test_invalid_code_returns_false(self) -> None:
        ok, msg = try_ast_check("def broken(:\n    pass")
        assert ok is False
        assert "SyntaxError" in msg

    def test_empty_string_is_valid(self) -> None:
        ok, _ = try_ast_check("")
        assert ok is True

    def test_multiline_valid(self) -> None:
        code = textwrap.dedent("""\
            def greet(name):
                return f"Hello, {name}"
        """)
        ok, _ = try_ast_check(code)
        assert ok is True

    def test_indentation_error_returns_false(self) -> None:
        ok, msg = try_ast_check("def f():\npass\n")
        assert ok is False
        assert "SyntaxError" in msg


# ---------------------------------------------------------------------------
# record_metrics
# ---------------------------------------------------------------------------


class TestRecordMetrics:
    def test_returns_list_with_one_entry_when_history_empty(self) -> None:
        state = AgentState(file_path="x.py", loop_count=1, total_tokens=200)
        history = record_metrics(state, "planner")
        assert len(history) == 1

    def test_entry_has_correct_fields(self) -> None:
        state = AgentState(file_path="x.py", loop_count=1, total_tokens=200)
        entry = record_metrics(state, "planner")[0]
        assert entry["node"] == "planner"
        assert entry["loop_count"] == 1
        assert entry["total_tokens"] == 200
        assert entry["step"] == 1
        assert "budget_used_ratio" in entry
        assert "accepted_fixes" in entry
        assert "analyzed_regions" in entry

    def test_step_increments_with_history(self) -> None:
        existing = [
            {
                "step": 1, "node": "load_file", "loop_count": 0,
                "total_tokens": 0, "accepted_fixes": 0,
                "analyzed_regions": 0, "budget_used_ratio": 0.0,
            }
        ]
        state = AgentState(file_path="x.py", metrics_history=existing)
        history = record_metrics(state, "planner")
        assert history[-1]["step"] == 2

    def test_budget_ratio_calculation(self) -> None:
        state = AgentState(file_path="x.py", total_tokens=4000, max_tokens_budget=8000)
        entry = record_metrics(state, "executor")[0]
        assert abs(entry["budget_used_ratio"] - 0.5) < 1e-9

    def test_does_not_mutate_existing_history(self) -> None:
        state = AgentState(file_path="x.py")
        _ = record_metrics(state, "planner")
        assert state.metrics_history == []  # original state untouched


# ---------------------------------------------------------------------------
# finalize_update
# ---------------------------------------------------------------------------


class TestFinalizeUpdate:
    def test_returns_dict_with_metrics_history(self) -> None:
        state = AgentState(file_path="x.py")
        update = {"next_step": "executor", "last_node": "planner"}
        result = finalize_update(state, update, "planner")
        assert "metrics_history" in result
        assert len(result["metrics_history"]) == 1

    def test_preserves_caller_keys(self) -> None:
        state = AgentState(file_path="x.py")
        update = {"next_step": "executor", "last_node": "planner"}
        result = finalize_update(state, update, "planner")
        assert result["next_step"] == "executor"
        assert result["last_node"] == "planner"

    def test_uses_updated_state_for_metrics(self) -> None:
        """Metrics snapshot must reflect the updated token count, not the old one."""
        state = AgentState(file_path="x.py", total_tokens=100)
        update = {"total_tokens": 300, "last_node": "executor"}
        result = finalize_update(state, update, "executor")
        assert result["metrics_history"][0]["total_tokens"] == 300
