"""Tests for agent.tools.sandbox_executor.

All tests are self-contained and require no external APIs, network access,
or Docker.  They exercise the subprocess-based sandbox against known Python
snippets to verify:

    - Success case       – valid code that runs cleanly.
    - Failure case       – code with a deliberate runtime error.
    - Timeout case       – code that deliberately sleeps beyond the limit.
    - Syntax error case  – syntactically invalid Python.
    - stdout capture     – printed output is returned.
    - stderr capture     – error output is returned.
    - result shape       – all required keys are present and typed correctly.
"""

from __future__ import annotations

from agent.tools.sandbox_executor import SandboxResult, run_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_result_shape(result: SandboxResult) -> None:
    """Assert that every required key is present with the correct type."""
    assert isinstance(result["success"], bool)
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
    assert isinstance(result["return_code"], int)
    assert isinstance(result["runtime_seconds"], float)
    assert isinstance(result["timed_out"], bool)
    assert result["error_category"] in {
        "none", "timeout", "syntax_error", "memory_error", "runtime_error"
    }


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


class TestSuccessCase:
    def test_empty_code_succeeds(self) -> None:
        result = run_code("")
        assert result["success"] is True
        assert result["timed_out"] is False
        assert result["error_category"] == "none"

    def test_simple_print_succeeds(self) -> None:
        result = run_code('print("hello sandbox")')
        assert result["success"] is True
        assert result["return_code"] == 0
        assert "hello sandbox" in result["stdout"]
        assert result["error_category"] == "none"

    def test_arithmetic_succeeds(self) -> None:
        result = run_code("x = 2 + 2\nprint(x)")
        assert result["success"] is True
        assert "4" in result["stdout"]

    def test_result_has_correct_shape(self) -> None:
        result = run_code('print("ok")')
        assert_result_shape(result)

    def test_runtime_seconds_is_non_negative(self) -> None:
        result = run_code("pass")
        assert result["runtime_seconds"] >= 0.0

    def test_stderr_is_empty_on_success(self) -> None:
        result = run_code("x = 1")
        assert result["stderr"] == ""


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------


class TestFailureCase:
    def test_runtime_error_returns_failure(self) -> None:
        result = run_code("raise ValueError('deliberate')")
        assert result["success"] is False
        assert result["return_code"] != 0
        assert result["timed_out"] is False

    def test_runtime_error_category(self) -> None:
        result = run_code("raise RuntimeError('oops')")
        assert result["error_category"] == "runtime_error"

    def test_stderr_contains_traceback(self) -> None:
        result = run_code("raise ValueError('deliberate')")
        assert "ValueError" in result["stderr"]

    def test_divide_by_zero_captured(self) -> None:
        result = run_code("print(1 / 0)")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["stderr"]

    def test_system_exit_nonzero_is_failure(self) -> None:
        result = run_code("import sys; sys.exit(1)")
        assert result["success"] is False
        assert result["return_code"] == 1


# ---------------------------------------------------------------------------
# Syntax error cases
# ---------------------------------------------------------------------------


class TestSyntaxErrorCase:
    def test_syntax_error_returns_failure(self) -> None:
        result = run_code("def broken(:\n    pass")
        assert result["success"] is False

    def test_syntax_error_category(self) -> None:
        result = run_code("def broken(:\n    pass")
        assert result["error_category"] == "syntax_error"

    def test_syntax_error_in_stderr(self) -> None:
        result = run_code("if True\n    pass")
        assert "SyntaxError" in result["stderr"]


# ---------------------------------------------------------------------------
# Timeout cases
# ---------------------------------------------------------------------------


class TestTimeoutCase:
    def test_timeout_sets_timed_out_flag(self) -> None:
        # Sleep 5 s with a 1 s timeout — must be killed.
        result = run_code("import time; time.sleep(5)", timeout=1.0)
        assert result["timed_out"] is True

    def test_timeout_returns_failure(self) -> None:
        result = run_code("import time; time.sleep(5)", timeout=1.0)
        assert result["success"] is False
        assert result["return_code"] == -1

    def test_timeout_error_category(self) -> None:
        result = run_code("import time; time.sleep(5)", timeout=1.0)
        assert result["error_category"] == "timeout"

    def test_runtime_within_timeout_does_not_trigger(self) -> None:
        result = run_code("pass", timeout=5.0)
        assert result["timed_out"] is False

    def test_runtime_seconds_recorded_on_timeout(self) -> None:
        result = run_code("import time; time.sleep(5)", timeout=1.0)
        # Runtime should be approximately the timeout value (1 s), not 0.
        assert result["runtime_seconds"] >= 0.5


# ---------------------------------------------------------------------------
# stdout / stderr capture
# ---------------------------------------------------------------------------


class TestOutputCapture:
    def test_multiline_stdout(self) -> None:
        code = "for i in range(3):\n    print(i)"
        result = run_code(code)
        assert result["success"] is True
        assert "0" in result["stdout"]
        assert "1" in result["stdout"]
        assert "2" in result["stdout"]

    def test_stderr_written_directly(self) -> None:
        code = "import sys\nsys.stderr.write('err_signal\\n')"
        result = run_code(code)
        # Writing to stderr alone does not cause a non-zero exit
        assert "err_signal" in result["stderr"]
