"""Entry point for the budget-aware LangGraph code-review agent.

Usage
-----
    # Analyse the default sample file
    python main.py

    # Analyse a custom file
    python main.py path/to/your_file.py

Prerequisites
-------------
Set the GOOGLE_API_KEY environment variable before running:

    export GOOGLE_API_KEY=<your-key>      # Linux / macOS
    $env:GOOGLE_API_KEY = '<your-key>'    # PowerShell

Alternatively, place it in a .env file in the project root (not tracked by git)
and load it with ``python-dotenv`` before invoking this script.
"""

from __future__ import annotations

import logging
import os
import sys

from agent import run_agent
from visualize import plot_demo_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

_DEFAULT_SAMPLE = "samples/sample_buggy_file.py"


def _require_api_key() -> None:
    """Exit with a clear message if the Gemini API key is not configured."""
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "Error: GOOGLE_API_KEY is not set.\n\n"
            "Export it before running:\n"
            "  export GOOGLE_API_KEY=<your-key>      # Linux / macOS\n"
            "  $env:GOOGLE_API_KEY = '<your-key>'    # PowerShell"
        )
        sys.exit(1)


def main() -> None:
    _require_api_key()

    file_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_SAMPLE
    print(f"\nAnalysing: {file_path}\n{'─' * 50}")

    result = run_agent(file_path)

    print("\n" + "═" * 50)
    print("FINAL REPORT")
    print("═" * 50)
    print(result["final_message"])

    print("\nGenerating telemetry plots …")
    plot_demo_metrics(result)


if __name__ == "__main__":
    main()
