"""Shared pytest fixtures for the agent test suite."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent.state import AgentState


@pytest.fixture
def simple_python_file(tmp_path: Path) -> Path:
    """Write a minimal valid Python file to a temp dir and return its path."""
    code = textwrap.dedent("""\
        def add(a, b):
            return a + b

        def divide(a, b):
            return a / b
    """)
    p = tmp_path / "sample.py"
    p.write_text(code, encoding="utf-8")
    return p


@pytest.fixture
def base_state(simple_python_file: Path) -> AgentState:
    """Return a default AgentState pointing at the simple Python fixture file."""
    return AgentState(file_path=str(simple_python_file))
