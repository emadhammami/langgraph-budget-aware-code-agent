# langgraph-budget-aware-code-agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vje68dj4UjgrkMaFiWfvKV6NhzLAULX2?usp=sharing)

A small research prototype demonstrating how to build a **budget-aware, observable agentic
workflow** using [LangGraph](https://github.com/langchain-ai/langgraph) and a
Planner–Executor–Critic architecture.

> **Scope note:** This is an academic demo, not a production system.  It is
> designed to illustrate software-engineering principles for LLM-enabled agentic
> systems — guardrails, shared state, routing logic, and telemetry — rather than
> to achieve state-of-the-art code-repair performance.

---

## Why this project matters

Modern agentic systems face several unsolved engineering challenges.  This
prototype deliberately addresses four of them in a self-contained, inspectable
codebase:

| Challenge | How it is addressed here |
|---|---|
| **Planner–Executor–Critic architecture** | Three distinct LLM roles with explicit interfaces, preventing any single agent from both proposing and approving its own output. |
| **Guardrails and budget-aware orchestration** | A centralized `guardrail_router` enforces hard limits on loop count and cumulative token spend before every node transition, not only at the end of a cycle. |
| **Observability / telemetry** | Every node calls `finalize_update`, which appends a structured snapshot to `metrics_history` and emits a JSON log line — providing a full audit trail without additional infrastructure. |
| **Dependable agentic systems** | Explicit shared state (Pydantic models), deterministic routing, and early-exit ensure the graph always terminates cleanly, even when the LLM misbehaves or budgets are exhausted. |

---

## Architecture

```
START
  │
load_file ──(guardrail_router)──► planner ──(guardrail_router)──► executor
                                      ▲                                │
                                      │             (guardrail_router) │
                                      │                                ▼
                                      └──────────────────────────── critic
                                                 (if regions remain)

At any edge:  guardrail_router may redirect to → early_exit → finish → END
```

### Nodes

| Node | Role |
|---|---|
| `load_file` | Reads the target Python file, splits it into lines, and uses the standard `ast` module to discover function boundaries as *candidate regions*. |
| `planner` | Selects the single most suspicious unanalysed region and forms a bug hypothesis.  Increments `loop_count` each call. |
| `executor` | Proposes a minimal, targeted fix for the selected region. |
| `critic` | Validates the fix with two signals: a local AST syntax check (zero LLM cost) and an LLM review.  Appends accepted fixes to `accepted_fixes`. |
| `early_exit` | Records a budget-exceeded message when either hard limit is breached. |
| `finish` | Terminal node; seals the final message for output. |

### Shared state (`AgentState`)

All inter-node communication passes through a single Pydantic model.  There are
no side channels.  Key fields:

- `candidate_regions` / `analyzed_regions` / `accepted_fixes` — analysis bookkeeping.
- `planner_output`, `executor_output`, `critic_output` — typed outputs from each role.
- `loop_count`, `total_tokens`, `max_loops`, `max_tokens_budget` — guardrail counters.
- `metrics_history` — append-only telemetry log (one entry per node execution).
- `next_step` — routing intent set by each node; may be overridden by the router.

### Guardrail router

```python
def guardrail_router(state: AgentState) -> str:
    if state.loop_count > state.max_loops or state.total_tokens > state.max_tokens_budget:
        return "early_exit"
    return state.next_step
```

This one function is attached to **every** conditional edge, so budget checks
happen between each pair of nodes — not only after the critic.

### Telemetry

Each call to `finalize_update` appends a snapshot to `metrics_history`:

```json
{
  "step": 3,
  "node": "critic",
  "loop_count": 2,
  "total_tokens": 1847,
  "accepted_fixes": 1,
  "analyzed_regions": 2,
  "budget_used_ratio": 0.23
}
```

`visualize.py` plots accepted fixes and budget consumption over steps.

---

## Repository structure

```
langgraph-budget-aware-code-agent/
├── agent/
│   ├── __init__.py      # Public API: exposes run_agent
│   ├── state.py         # Pydantic state models (AgentState and sub-models)
│   ├── helpers.py       # Shared utilities: token estimation, metrics, LLM client
│   ├── nodes.py         # LangGraph node functions
│   ├── router.py        # Guardrail router
│   └── graph.py         # Graph assembly and run_agent entry point
├── samples/
│   └── sample_buggy_file.py   # Minimal demo input with three intentional bugs
├── main.py              # CLI entry point
├── visualize.py         # Telemetry charts (matplotlib)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to run

### Option A — Google Colab (no setup required)

Open the notebook directly in your browser and run all cells:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vje68dj4UjgrkMaFiWfvKV6NhzLAULX2?usp=sharing)

The notebook installs its own dependencies and prompts you for your Gemini API
key via `getpass`, so no local configuration is needed.

### Option B — Local Python environment

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Set your Gemini API key

```bash
# Linux / macOS
export GOOGLE_API_KEY=<your-key>

# PowerShell
$env:GOOGLE_API_KEY = '<your-key>'
```

#### 3. Run on the sample file

```bash
python main.py
```

#### 4. Run on a custom file

```bash
python main.py path/to/your_file.py
```

The agent will print a structured log to stdout as it runs, then output a final
report listing accepted fixes.  Two telemetry plots are displayed at the end.

### Expected output (sample)

```
Analysing: samples/sample_buggy_file.py
──────────────────────────────────────────────────
... (structured JSON log per node) ...

══════════════════════════════════════════════════
FINAL REPORT
══════════════════════════════════════════════════
Analysis complete.

Region: divide:lines 1-2
Summary: Add a zero-division guard
Fix:
def divide(a, b):
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b

---

Region: greet:lines 4-6
Summary: Replace mutable default argument
Fix:
def greet(name=None):
    if name is None:
        name = []
    name.append("x")
    return "Hello " + str(name)
...
```

---

## Sample input

`samples/sample_buggy_file.py` contains three intentional Python bugs that
demonstrate concrete, recognisable failure modes:

| Function | Bug | Failure |
|---|---|---|
| `divide(a, b)` | No zero-division guard | `ZeroDivisionError` at runtime |
| `greet(name=[])` | Mutable default argument | State persists silently across calls |
| `get_item(items, idx)` | No bounds check | `IndexError` for out-of-range index |

---

## Limitations

This is a **prototype built for demonstration purposes**.  Specific limitations:

- **No test suite.** The agent's outputs are not automatically validated against known-good fixes.
- **Single-pass per region.** The executor proposes one fix; there is no retry or refinement loop within a region.
- **No file patching.** Accepted fixes are printed to stdout; the original file is not modified.
- **LLM dependence.** Fix quality depends entirely on the underlying model and prompt; no static analysis beyond AST is used.
- **Token estimation fallback.** When usage metadata is unavailable, token counts are approximated with a 4-chars-per-token heuristic.
- **Flat region granularity.** Regions are individual top-level functions; nested functions and class methods are not separately tracked.
- **Gemini-specific.** The LLM client is `langchain-google-genai`; swapping to a different provider requires changing `build_llm` in `helpers.py`.

---

## Optional improvements (not implemented)

These are directions that would strengthen the prototype for research or
production use, but are **outside the scope of this demo**:

- [ ] **Retry loop in executor** — re-prompt if the critic rejects a fix (with a back-off counter to avoid infinite loops).
- [ ] **In-place file patching** — apply accepted fixes and write the corrected file.
- [ ] **Richer static analysis** — integrate `pylint` or `pyflakes` output as an additional Critic signal.
- [ ] **Configurable LLM provider** — make `build_llm` accept a provider argument so the graph can run against any LangChain-compatible model.
- [ ] **CLI budget flags** — expose `--max-loops` and `--max-tokens` as `argparse` arguments.
- [ ] **Structured output parsing** — use LangChain's `with_structured_output` to enforce JSON schemas instead of manual parsing.
- [ ] **Test suite** — property-based tests for helpers and golden-file integration tests for the full graph.

---

## Environment

Tested with:

- Python 3.11
- `langgraph` 0.2.x
- `langchain-google-genai` 2.x (Gemini 2.5 Flash)
- `pydantic` 2.x

---

## License

MIT
