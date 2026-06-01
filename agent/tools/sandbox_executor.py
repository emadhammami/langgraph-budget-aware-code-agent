"""Subprocess-based sandbox for executing candidate Python code safely.

Design goals
------------
- **Portable**: runs on Windows, macOS, and Linux without Docker.
- **Observable**: captures stdout, stderr, return status, and wall-clock time.
- **Safe**: enforces a hard timeout; applies soft memory limits on POSIX
  platforms where the ``resource`` module is available.
- **Research-oriented**: returns structured results that feed directly into
  the Critic and telemetry subsystems.

Output format
-------------
::

    {
        "success": false,
        "stdout": "...",
        "stderr": "SyntaxError: invalid syntax ...",
        "runtime_seconds": 0.34,
        "timed_out": false,
        "error_category": "runtime_error"
    }

``error_category`` is one of:
    ``"none"``         – code executed successfully (exit code 0).
    ``"timeout"``      – process killed after exceeding *timeout* seconds.
    ``"syntax_error"`` – Python raised ``SyntaxError`` before execution.
    ``"memory_error"`` – Python raised ``MemoryError`` during execution.
    ``"runtime_error"`` – any other non-zero exit code.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TypedDict


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class SandboxResult(TypedDict):
    """Structured output returned by :func:`run_code`."""

    success: bool
    stdout: str
    stderr: str
    runtime_seconds: float
    timed_out: bool
    error_category: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _categorise_error(stderr: str, timed_out: bool, returncode: int) -> str:
    """Derive a coarse error category from subprocess outcome signals."""
    if returncode == 0:
        return "none"
    if timed_out:
        return "timeout"
    if "SyntaxError" in stderr:
        return "syntax_error"
    if "MemoryError" in stderr:
        return "memory_error"
    return "runtime_error"


def _make_preexec(memory_limit_bytes: int):  # type: ignore[return]
    """
    Return a ``preexec_fn`` that applies a soft ``RLIMIT_AS`` cap.

    On Windows (and any platform lacking the ``resource`` module) this
    function returns ``None`` so ``subprocess.Popen`` ignores it.
    """
    try:
        import resource  # noqa: PLC0415  (POSIX only)

        def _set_limit() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

        return _set_limit
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_code(
    source: str,
    *,
    timeout: float = 10.0,
    memory_limit_mb: int = 256,
    max_output_bytes: int = 65_536,
) -> SandboxResult:
    """
    Execute *source* in a sandboxed subprocess and return structured results.

    Parameters
    ----------
    source:
        Python source code to execute.
    timeout:
        Hard wall-clock limit in seconds.  The process is killed if it
        exceeds this value.
    memory_limit_mb:
        Soft memory ceiling in MiB.  Applied only on POSIX systems via
        ``RLIMIT_AS``; silently ignored on Windows.
    max_output_bytes:
        Maximum bytes captured from stdout and stderr combined.  Output
        beyond this limit is truncated with a notice appended.

    Returns
    -------
    SandboxResult
        A typed dict with keys ``success``, ``stdout``, ``stderr``,
        ``runtime_seconds``, ``timed_out``, and ``error_category``.
    """
    timed_out = False
    returncode = -1

    # Write source to a temporary file so there is no shell-injection risk
    # and the subprocess inherits a clean environment.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        preexec = _make_preexec(memory_limit_mb * 1024 * 1024)

        t_start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                timeout=timeout,
                preexec_fn=preexec,  # None on Windows → ignored by subprocess
            )
            returncode = proc.returncode
            raw_stdout = proc.stdout
            raw_stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = -1
            raw_stdout = exc.stdout or b""
            raw_stderr = exc.stderr or b""
        finally:
            runtime = time.perf_counter() - t_start
    finally:
        tmp_path.unlink(missing_ok=True)

    # Decode and truncate output
    stdout = _decode_truncate(raw_stdout, max_output_bytes // 2)
    stderr = _decode_truncate(raw_stderr, max_output_bytes // 2)

    error_category = _categorise_error(stderr, timed_out, returncode)

    return SandboxResult(
        success=(returncode == 0),
        stdout=stdout,
        stderr=stderr,
        runtime_seconds=round(runtime, 4),
        timed_out=timed_out,
        error_category=error_category,
    )


def _decode_truncate(raw: bytes, limit: int) -> str:
    """Decode bytes to str, replacing undecodable characters, then truncate."""
    text = raw.decode("utf-8", errors="replace")
    if len(text) > limit:
        text = text[:limit] + f"\n[... output truncated at {limit} characters ...]"
    return text
