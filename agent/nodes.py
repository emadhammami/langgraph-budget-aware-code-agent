"""LangGraph node functions for the budget-aware code-review agent.

Each node accepts the current ``AgentState``, performs its work, and returns
a partial state dict via ``finalize_update``.  LangGraph merges that dict back
into the shared state before routing to the next node.

Node responsibilities
---------------------
load_file   – Read the target file and discover candidate regions via AST.
planner     – Choose the next unanalysed region and form a bug hypothesis.
executor    – Propose a concrete minimal fix for the selected region.
critic      – Validate the fix (local AST check + LLM review).
early_exit  – Record a budget-exceeded message and hand off to finish.
finish      – Seal the final message for output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .helpers import (
    build_llm,
    extract_candidate_regions,
    finalize_update,
    llm_json_invoke,
    parse_region_to_line_range,
    safe_slice,
    try_ast_check,
)
from .state import AgentState, AnalysisPlan, CritiqueResult, FixProposal, ValidationResult
from .tools.sandbox_executor import run_code


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------

def load_file_node(state: AgentState) -> Dict[str, Any]:
    """
    Read the target Python file from disk, split it into lines, and extract
    candidate regions (function definitions) via AST walking.

    Routes to ``finish`` immediately if the file does not exist, so the rest
    of the graph never receives an empty state.
    """
    path = Path(state.file_path)
    if not path.exists():
        update = {
            "next_step": "finish",
            "final_message": f"File not found: {state.file_path}",
            "last_node": "load_file",
        }
        return finalize_update(state, update, "load_file")

    file_content = path.read_text(encoding="utf-8")
    file_lines = file_content.splitlines()
    candidate_regions = extract_candidate_regions(file_content)

    update = {
        "file_content": file_content,
        "file_lines": file_lines,
        "candidate_regions": candidate_regions,
        "next_step": "planner",
        "last_node": "load_file",
    }
    return finalize_update(state, update, "load_file")


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Identify the single most suspicious unanalysed region to inspect next.

    If all regions have already been analysed the planner short-circuits to
    ``finish`` without making an LLM call.  Otherwise it asks the LLM to
    select a region and form a bug hypothesis, then increments ``loop_count``
    to advance the guardrail counter.
    """
    remaining: List[str] = [r for r in state.candidate_regions if r not in state.analyzed_regions]

    if not remaining:
        summary = (
            "\n\n---\n\n".join(state.accepted_fixes) if state.accepted_fixes else "No accepted fixes."
        )
        update = {
            "next_step": "finish",
            "final_message": f"Analysis complete.\n\n{summary}",
            "last_node": "planner",
        }
        return finalize_update(state, update, "planner")

    model = build_llm()

    system_prompt = (
        "You are the Planner.\n"
        "Choose the single most suspicious function to inspect next.\n\n"
        "Return JSON only:\n"
        '{\n  "target_region": "...",\n  "rationale": "...",\n  "bug_hypothesis": "..."\n}'
    )
    user_prompt = (
        f"Candidate regions:\n{json.dumps(remaining, indent=2)}\n\n"
        f"File preview:\n{safe_slice(state.file_lines, 1, min(len(state.file_lines), 50))}"
    )

    data, used_tokens = llm_json_invoke(model, system_prompt, user_prompt)

    chosen = str(data.get("target_region", remaining[0]))
    if chosen not in remaining:
        chosen = remaining[0]

    plan = AnalysisPlan(
        target_region=chosen,
        rationale=str(data.get("rationale", "Selected most suspicious region.")),
        bug_hypothesis=str(data.get("bug_hypothesis", "Potential bug in selected region.")),
    )

    update = {
        "planner_output": plan,
        "total_tokens": state.total_tokens + used_tokens,
        "loop_count": state.loop_count + 1,
        "next_step": "executor",
        "last_node": "planner",
    }
    return finalize_update(state, update, "planner")


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------

def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Propose a minimal fix for the region selected by the Planner.

    The relevant code snippet is extracted from ``file_lines`` using the
    region's line-range label and included verbatim in the LLM prompt
    alongside the Planner's bug hypothesis.
    """
    model = build_llm()
    start, end = parse_region_to_line_range(
        state.planner_output.target_region, len(state.file_lines)
    )
    snippet = safe_slice(state.file_lines, start, end)

    system_prompt = (
        "You are the Executor.\n"
        "Propose one small Python fix.\n\n"
        "Return JSON only:\n"
        '{\n  "summary": "...",\n  "fixed_snippet": "...",\n  "explanation": "..."\n}'
    )
    user_prompt = (
        f"Region:\n{state.planner_output.target_region}\n\n"
        f"Bug hypothesis:\n{state.planner_output.bug_hypothesis}\n\n"
        f"Code:\n{snippet}"
    )

    data, used_tokens = llm_json_invoke(model, system_prompt, user_prompt)

    proposal = FixProposal(
        summary=str(data.get("summary", "Proposed a fix.")),
        fixed_snippet=str(data.get("fixed_snippet", snippet)),
        explanation=str(data.get("explanation", "Fix explanation.")),
    )

    update = {
        "executor_output": proposal,
        "total_tokens": state.total_tokens + used_tokens,
        "next_step": "validation",
        "last_node": "executor",
    }
    return finalize_update(state, update, "executor")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validation_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute the Executor's proposed fix in a sandboxed subprocess and attach
    objective execution evidence to the workflow state.

    Responsibilities
    ----------------
    - Pass ``executor_output.fixed_snippet`` to :func:`run_code`.
    - Capture stdout, stderr, return status, and wall-clock time.
    - Wrap results in a :class:`~agent.state.ValidationResult` and store it
      on state so the Critic can reason over concrete execution evidence.
    - Increment ``validation_failures`` when execution is unsuccessful,
      giving the telemetry subsystem a cumulative failure counter.

    The node always routes to ``"critic"`` — a failed execution is *evidence*
    for the Critic to act on, not a reason to skip it.
    """
    sandbox_result = run_code(state.executor_output.fixed_snippet)

    validation = ValidationResult(
        success=sandbox_result["success"],
        stdout=sandbox_result["stdout"],
        stderr=sandbox_result["stderr"],
        return_code=sandbox_result["return_code"],
        runtime_seconds=sandbox_result["runtime_seconds"],
        timed_out=sandbox_result["timed_out"],
        error_category=sandbox_result["error_category"],
    )

    new_failures = state.validation_failures + (0 if sandbox_result["success"] else 1)

    update = {
        "validation_result": validation,
        "validation_failures": new_failures,
        "validation_success": validation.success,
        "execution_stdout": validation.stdout,
        "execution_stderr": validation.stderr,
        "execution_return_code": validation.return_code,
        "runtime_seconds": validation.runtime_seconds,
        "validation_error_type": validation.error_category,
        "next_step": "critic",
        "last_node": "validation",
    }
    return finalize_update(state, update, "validation")


# ---------------------------------------------------------------------------
# critic
# ---------------------------------------------------------------------------

def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    Validate the Executor's proposed fix using three complementary signals:

    1. **Execution evidence** (primary) – the ``ValidationResult`` produced by
       :func:`validation_node` contains objective, deterministic facts:
       whether the code ran successfully, any stderr output, and the
       wall-clock runtime.  The Critic treats this as ground truth.
    2. **Local AST syntax check** – a fast, zero-cost static check that
       confirms basic syntactic correctness before the LLM call.
    3. **LLM review** (secondary) – interprets the execution evidence and
       the original bug hypothesis to make a final accept/reject decision.

    By anchoring the LLM prompt in concrete execution results rather than
    only code text, the Critic's judgement is grounded in observable facts.
    """
    model = build_llm()
    syntax_ok, syntax_msg = try_ast_check(state.executor_output.fixed_snippet)
    vr = state.validation_result

    # Format execution evidence for the prompt
    exec_summary_lines = [
        f"Executed successfully: {vr.success}",
        f"Return code:           {vr.return_code}",
        f"Exit category:        {vr.error_category}",
        f"Runtime (seconds):    {vr.runtime_seconds}",
        f"Timed out:            {vr.timed_out}",
    ]
    if vr.stdout.strip():
        exec_summary_lines.append(f"stdout:\n{vr.stdout.strip()}")
    if vr.stderr.strip():
        exec_summary_lines.append(f"stderr:\n{vr.stderr.strip()}")
    exec_summary = "\n".join(exec_summary_lines)

    system_prompt = (
        "You are the Critic in an evidence-based code-review workflow.\n"
        "Execution evidence is PRIMARY.  LLM reasoning is SECONDARY.\n\n"
        "Decision rules:\n"
        "  - If execution FAILED (success=false), reject unless the failure is\n"
        "    clearly unrelated to the fix (e.g. missing test data).\n"
        "  - If execution SUCCEEDED but there are semantic concerns, reason\n"
        "    carefully before accepting.\n"
        "  - Always justify your decision by referencing the execution evidence.\n\n"
        "Return JSON only:\n"
        '{\n  "valid": true,\n  "reason": "..."\n}'
    )
    user_prompt = (
        f"Region:\n{state.planner_output.target_region}\n\n"
        f"Bug hypothesis:\n{state.planner_output.bug_hypothesis}\n\n"
        f"Proposed fix:\n{state.executor_output.fixed_snippet}\n\n"
        f"--- Execution evidence (primary) ---\n{exec_summary}\n\n"
        f"--- Static syntax check ---\n{syntax_msg}"
    )

    data, used_tokens = llm_json_invoke(model, system_prompt, user_prompt)

    # If execution succeeded and the LLM gives no opinion, default to accept.
    # If execution failed, default to reject regardless of LLM silence.
    default_valid = vr.success and syntax_ok
    critique = CritiqueResult(
        valid=bool(data.get("valid", default_valid)),
        reason=str(data.get("reason", "Critic evaluation complete.")),
    )

    new_analyzed = state.analyzed_regions + [state.planner_output.target_region]
    new_accepted = list(state.accepted_fixes)

    if critique.valid:
        new_accepted.append(
            f"Region: {state.planner_output.target_region}\n"
            f"Summary: {state.executor_output.summary}\n"
            f"Fix:\n{state.executor_output.fixed_snippet}"
        )

    remaining = [r for r in state.candidate_regions if r not in new_analyzed]
    next_step = "planner" if remaining else "finish"

    final_message = ""
    if next_step == "finish":
        final_message = "Analysis complete.\n\n" + (
            "\n\n---\n\n".join(new_accepted) if new_accepted else "No accepted fixes."
        )

    update = {
        "critic_output": critique,
        "analyzed_regions": new_analyzed,
        "accepted_fixes": new_accepted,
        "total_tokens": state.total_tokens + used_tokens,
        "next_step": next_step,
        "final_message": final_message,
        "last_node": "critic",
    }
    return finalize_update(state, update, "critic")


# ---------------------------------------------------------------------------
# early_exit
# ---------------------------------------------------------------------------

def early_exit_node(state: AgentState) -> Dict[str, Any]:
    """
    Triggered by the guardrail router when a budget limit is breached.

    Records the reason for early termination before handing off to ``finish``
    so that callers can distinguish a budget stop from a normal completion.
    """
    update = {
        "final_message": (
            f"Early exit: budget limit reached "
            f"(loop_count={state.loop_count}, total_tokens={state.total_tokens})"
        ),
        "next_step": "finish",
        "last_node": "early_exit",
    }
    return finalize_update(state, update, "early_exit")


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------

def finish_node(state: AgentState) -> Dict[str, Any]:
    """Terminal node — seals the final message and records the last node."""
    update = {
        "final_message": state.final_message or "Finished.",
        "last_node": "finish",
    }
    return finalize_update(state, update, "finish")
