"""Graph assembly and public entry point.

``build_graph`` wires together all nodes and conditional edges into a compiled
LangGraph state machine.  ``run_agent`` is the single public function callers
need: it accepts a file path, compiles the graph, and returns the final
``AgentState`` as a plain dict.

Graph topology
--------------

    START
      │
    load_file ──(guardrail_router)──► planner ──(guardrail_router)──► executor
                                         ▲                                │
                                         │                (guardrail_router)
                                         │                                ▼
                                         └────────────────────────── critic
                                                    (if regions remain)

At any conditional edge the guardrail_router may redirect to:

    early_exit → finish → END
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from .nodes import (
    critic_node,
    early_exit_node,
    executor_node,
    finish_node,
    load_file_node,
    planner_node,
)
from .router import guardrail_router
from .state import AgentState

# All possible routing destinations exposed to every conditional edge.
_ALL_TARGETS = {
    "planner": "planner",
    "executor": "executor",
    "critic": "critic",
    "early_exit": "early_exit",
    "finish": "finish",
}


def build_graph():
    """
    Assemble and compile the LangGraph state machine.

    Each conditional edge passes through ``guardrail_router`` so that budget
    violations are caught between *every* pair of nodes, not only after the
    critic.
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_file", load_file_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("early_exit", early_exit_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "load_file")

    graph.add_conditional_edges("load_file", guardrail_router, _ALL_TARGETS)
    graph.add_conditional_edges("planner",   guardrail_router, _ALL_TARGETS)
    graph.add_conditional_edges("executor",  guardrail_router, _ALL_TARGETS)
    graph.add_conditional_edges("critic",    guardrail_router, _ALL_TARGETS)

    graph.add_edge("early_exit", "finish")
    graph.add_edge("finish", END)

    return graph.compile()


def run_agent(file_path: str) -> Dict[str, Any]:
    """
    Run the budget-aware code-review agent on a Python source file.

    Parameters
    ----------
    file_path : str
        Path to the Python file to analyse.

    Returns
    -------
    dict
        Final ``AgentState`` as a plain dict, including ``final_message``
        and ``metrics_history``.
    """
    app = build_graph()
    initial_state = AgentState(file_path=file_path)
    return app.invoke(initial_state)
