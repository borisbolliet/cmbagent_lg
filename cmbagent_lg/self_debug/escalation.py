"""Escalation node — the self_debug escape hatch (Path A: Claude Agent SDK).

When the strict loop hits a failure it structurally cannot fix — a missing
package, or a renamed/removed API — `route_after_execution_evaluator` routes
here instead of looping the engineer. This node runs a single, bounded Claude
Agent SDK `query()`: a free-form agent that can web-search the fix, `pip
install` a missing package, and/or make a minimal `Edit` to
`codebase/step_{N}.py`. Control then returns to the executor to re-run.

This is the **one Anthropic dependency** in cmbagent_lg — `claude-agent-sdk`
runs Claude models (needs `ANTHROPIC_API_KEY`). The rest of the graph is
model-agnostic. Escalation is opt-in (`PlanContext.enable_escalation`),
one-shot per step, and bounded by `escalation_max_budget_usd` /
`escalation_max_turns`. It does NOT consume an engineer attempt.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import time

from langgraph.runtime import Runtime

from cmbagent_lg.context import PlanContext
from cmbagent_lg.self_debug.nodes import _codebase_dir, _logs_dir
from cmbagent_lg.self_debug.prompts import escalation_instructions
from cmbagent_lg.self_debug.schemas import EngineerResponse
from cmbagent_lg.self_debug.state import DebugState
from cmbagent_lg.timing import timed_node


def _run_async(coro):
    """Run a coroutine to completion whether or not an event loop is already
    running — langgraph's sync `invoke`/`stream` has no loop; a notebook does."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running (e.g. Jupyter) — run in a separate thread.
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _summarize_tool_input(name: str, input_dict: dict) -> str:
    """One-line summary of a tool call's input, for the live ticker."""
    if not isinstance(input_dict, dict):
        return ""
    if name == "WebFetch":
        return str(input_dict.get("url", ""))[:120]
    if name == "WebSearch":
        return str(input_dict.get("query", ""))[:80]
    if name in ("Read", "Edit", "Write"):
        return str(input_dict.get("file_path", ""))
    if name == "Bash":
        cmd = str(input_dict.get("command", ""))
        return (cmd[:80] + "…") if len(cmd) > 80 else cmd
    # Fallback: first key=value, truncated.
    if input_dict:
        k, v = next(iter(input_dict.items()))
        sv = str(v)
        return f"{k}={sv[:60]}{'…' if len(sv) > 60 else ''}"
    return ""


async def _run_query(prompt: str, options):
    """Drive the Claude Agent SDK query to completion, streaming a live ticker
    of the agent's reasoning + tool calls to stderr as they arrive.

    Returns `(ResultMessage | None, [tool names invoked])`.
    """
    from claude_agent_sdk import query, AssistantMessage, ResultMessage

    result = None
    tool_calls: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                # Text block — the model's reasoning. Print the first line.
                text = getattr(block, "text", None)
                if text and text.strip():
                    snippet = text.strip().split("\n", 1)[0][:150]
                    print(f"[escalation]   {snippet}", file=sys.stderr, flush=True)
                # Tool-use block — show what's about to run.
                name = getattr(block, "name", None)
                if name:
                    tool_calls.append(name)
                    summary = _summarize_tool_input(
                        name, getattr(block, "input", {}) or {}
                    )
                    print(
                        f"[escalation] → {name}({summary})",
                        file=sys.stderr,
                        flush=True,
                    )
        elif isinstance(message, ResultMessage):
            result = message
    return result, tool_calls


def _emit_langfuse(record: dict) -> None:
    """Best-effort: attach the escalation summary to the current langfuse span
    (the escalation node's own span), tagged `escalation`, so the Claude work
    is visible in the trace next to the Gemini nodes.

    The JSON log is the reliable record; this is a convenience and is skipped
    silently on any error. Provisional until verified against a live run.
    """
    try:
        from langfuse import get_client

        get_client().update_current_span(
            metadata={
                "escalation_reason": record["reason"],
                "escalation_cost_usd": record["cost_usd"],
                "escalation_turns": record["turns"],
                "escalation_tool_calls": record["tool_calls"],
                "tags": ["escalation"],
            }
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never block the loop
        print(f"[escalation] langfuse note skipped: {e}", file=sys.stderr, flush=True)


@timed_node("escalation")
def escalation(state: DebugState, runtime: Runtime[PlanContext]) -> DebugState:
    """Run one bounded Claude Agent SDK pass to resolve an escalatable failure."""
    ctx = runtime.context
    step = state["step"]
    n = state.get("step_number", 1)
    code = state["current_code"].python_code
    stderr = state.get("execution_stderr", "")

    # Reason comes from execution_evaluator's classification (it reads
    # stdout+stderr and isn't fooled by a caught-and-printed exception).
    verdict = state.get("current_execution_verdict")
    kind = getattr(verdict, "failure_kind", None) or "unknown"
    summary = getattr(verdict, "error_summary", None) or ""
    reason = f"{kind}: {summary}".strip().rstrip(":")

    codebase = _codebase_dir(state)
    if codebase is None:
        # Escalation needs a real codebase dir for the SDK agent to edit.
        # No work_dir → degrade gracefully: mark escalated so the loop falls
        # through to the strict engineer retry.
        print("[escalation] no work_dir — skipping", file=sys.stderr, flush=True)
        return {"escalated": True, "escalation_reason": reason}

    # `execution_evaluator` already renamed the failed step_{N}.py to a
    # failure-variant. Re-materialize the canonical name from state so the
    # SDK agent has a file to read and edit.
    code_path = (codebase / f"step_{n}.py").resolve()
    code_path.write_text(code)
    work_dir = codebase.parent

    prompt = escalation_instructions(ctx, step, str(code_path), stderr, reason)

    from claude_agent_sdk import ClaudeAgentOptions

    options_kwargs = dict(
        allowed_tools=["WebSearch", "WebFetch", "Read", "Edit", "Bash"],
        permission_mode="acceptEdits",
        cwd=str(work_dir),
        max_turns=ctx.escalation_max_turns,
        max_budget_usd=ctx.escalation_max_budget_usd,
    )
    if ctx.escalation_model:
        options_kwargs["model"] = ctx.escalation_model
    options = ClaudeAgentOptions(**options_kwargs)

    print(
        f"[escalation] reason={reason!r}  model={ctx.escalation_model or '(SDK default)'}  "
        f"budget=${ctx.escalation_max_budget_usd}  max_turns={ctx.escalation_max_turns}",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        result, tool_calls = _run_async(_run_query(prompt, options))
    except Exception as e:  # noqa: BLE001 — a failed escalation must not crash the graph
        print(f"[escalation] query failed: {e}", file=sys.stderr, flush=True)
        result, tool_calls = None, []
    elapsed = time.perf_counter() - t0

    cost = float(getattr(result, "total_cost_usd", 0.0) or 0.0)
    turns = getattr(result, "num_turns", None)

    # Read whatever the agent left on disk — edited, or unchanged (a pure
    # `pip install` fix leaves the code byte-identical).
    fixed_code = code_path.read_text()
    code_changed = fixed_code != code
    print(
        f"[escalation] done in {elapsed:.1f}s  cost=${cost:.4f}  turns={turns}  "
        f"code_changed={code_changed}  tool_calls={tool_calls}",
        file=sys.stderr,
        flush=True,
    )

    record = {
        "step_number": n,
        "reason": reason,
        "elapsed_s": round(elapsed, 2),
        "cost_usd": cost,
        "turns": turns,
        "tool_calls": tool_calls,
        "code_changed": code_changed,
    }
    logs = _logs_dir(state)
    if logs is not None:
        (logs / f"step_{n}_escalation.json").write_text(json.dumps(record, indent=2))
    _emit_langfuse(record)

    new_code = EngineerResponse(
        python_code=fixed_code,
        code_explanation=state["current_code"].code_explanation,
        modification_summary=f"escalation resolved: {reason}",
    )
    return {
        "current_code": new_code,
        "escalated": True,
        "escalation_reason": reason,
    }


__all__ = ["escalation"]
