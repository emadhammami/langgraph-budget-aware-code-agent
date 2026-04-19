"""Telemetry visualisation for the budget-aware code-review agent.

``plot_demo_metrics`` produces two matplotlib charts from the
``metrics_history`` list accumulated during a run:

1. Accepted fixes vs. step number  – tracks how many validated fixes the
   agent accumulates as it works through candidate regions.
2. Token budget consumption ratio vs. step number  – shows how quickly the
   agent approaches its token ceiling, making the guardrail visible.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_demo_metrics(result: Dict[str, Any]) -> None:
    """
    Plot per-step telemetry from a completed agent run.

    Parameters
    ----------
    result : dict
        The final AgentState dict returned by ``run_agent``.
        Must contain a non-empty ``metrics_history`` key.
    """
    history = result.get("metrics_history", [])
    if not history:
        print("No metrics to plot.")
        return

    df = pd.DataFrame(history)

    # --- Chart 1: accepted fixes over time ---
    plt.figure(figsize=(7, 4))
    plt.plot(df["step"], df["accepted_fixes"], marker="o")
    plt.xlabel("Step")
    plt.ylabel("Accepted Fixes")
    plt.title("Accepted Fixes Over Time")
    plt.tight_layout()
    plt.show()

    # --- Chart 2: token budget consumption over time ---
    plt.figure(figsize=(7, 4))
    plt.plot(df["step"], df["budget_used_ratio"], marker="o", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Budget Used Ratio")
    plt.title("Token Budget Usage Over Time")
    plt.tight_layout()
    plt.show()
