"""Deep-research orchestrator nodes: run one Step, then decide what's next.

Mirrors cmbagent's control phase (`cmbagent/workflows/deep_research.py:688-958`)
but as a tiny langgraph wrapper around the existing `self_debug_graph`:

  - `run_step` builds the cross-step context (prior summaries + a freshly-
    scanned workspace file manifest), then invokes `self_debug_graph` for the
    current step. The summary plumbing matches cmbagent's
    `{previous_steps_execution_summary}` placeholder (built in
    `cmbagent/functions/status.py:651-654`).
  - `after_step` advances or halts based on the step's outcome.

Per-step state reset is automatic: a fresh `self_debug_graph.invoke` per step
starts with clean `attempts=0`, `error_history=[]`, `escalated=False`, etc.
The deep_research state only carries the truly cross-step things.

v1 limitation: only `sub_task_agent == "engineer"` steps are supported.
Non-engineer steps halt the plan with a clear outcome.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from langgraph.graph import END
from langgraph.runtime import Runtime

from cmbagent_lg.context import PlanContext
from cmbagent_lg.deep_research.state import DeepResearchState
from cmbagent_lg.planning.schemas import Step
from cmbagent_lg.researcher.graph import graph as researcher_graph
from cmbagent_lg.self_debug.graph import graph as self_debug_graph
from cmbagent_lg.timing import timed_node


# ── summary + manifest builders ─────────────────────────────────────────


def _build_step_summary(step: Step, sub_state: dict) -> str:
    """One per-step block, cmbagent-style. Appended to `step_summaries`.

    Mirrors the snapshot built in `cmbagent/functions/status.py:625-645`:
    a header (`### Step N (agent): sub_task`) followed by the step's
    artifact (executed code + stdout for engineer; written report for
    researcher). The next step's agent sees this in its prompt.
    """
    n = sub_state.get("step_number", 1)
    lines = [f"### Step {n} ({step.sub_task_agent}): {step.sub_task}", ""]

    if step.sub_task_agent == "researcher":
        report = (sub_state.get("current_report") or "").strip()
        report_path = sub_state.get("report_path") or ""
        if report:
            # Full report, no truncation — same rationale as the engineer's
            # full-code rule: downstream steps may need exact wording (named
            # methods, numbers, recommendations) to act on the report.
            label = f"Written report ({report_path})" if report_path else "Written report"
            lines += [f"**{label}:**", "", report, ""]
        return "\n".join(lines)

    # engineer (and any future code-running agent)
    code_obj = sub_state.get("current_code")
    code = getattr(code_obj, "python_code", "") if code_obj else ""
    stdout = (sub_state.get("execution_stdout") or "").strip()
    if code:
        # Full code, no truncation — the next step's engineer needs the exact
        # I/O lines (e.g. `np.savez(..., train=...)` keys) to consume what
        # this step produced. (Truncating the middle once cost us a `KeyError`
        # when later steps had to guess at file schemas they couldn't see.)
        lines += ["**Executed code:**", "```python", code, "```", ""]
    if stdout:
        lines += ["**Execution output:**", stdout, ""]
    return "\n".join(lines)


def _build_workspace_manifest(work_dir: Optional[str]) -> str:
    """Flat snapshot of files in `codebase/`, `data/`, `logs/`. Paths only.

    Distinct from self_debug's per-step `data_manifest` (which is provenance
    for *one* step's outputs). This is what cmbagent calls the WORKSPACE FILE
    MANIFEST (`cmbagent/functions/status.py:181-245`): a flat listing of
    everything generated so far, so the engineer knows the exact relative
    paths to load earlier steps' work from.
    """
    if not work_dir:
        return ""
    wd = Path(work_dir).expanduser()
    sections = [
        ("Python scripts (codebase/)",  wd / "codebase"),
        ("Research reports (reports/)", wd / "reports"),
        ("Data files (data/)",          wd / "data"),
        ("Run logs (logs/)",            wd / "logs"),
    ]
    rendered = []
    for label, sub in sections:
        if not sub.is_dir():
            continue
        files = sorted(p.relative_to(wd) for p in sub.iterdir() if p.is_file())
        if not files:
            continue
        rendered.append(f"**{label}:**")
        rendered.extend(f"- {p}" for p in files)
        rendered.append("")
    if not rendered:
        return ""
    return (
        "**WORKSPACE FILE MANIFEST** (use these exact paths to load earlier "
        "steps' artifacts):\n\n" + "\n".join(rendered)
    )


def _render_previous_steps(summaries: list, manifest: str) -> str:
    """Combined block injected into the engineer prompt as
    `{previous_steps_execution_summary}`. Empty when both are empty so a
    standalone self_debug invocation isn't polluted with empty headers."""
    if not summaries and not manifest:
        return ""
    parts = []
    if summaries:
        parts.append("----- PREVIOUS STEPS (executed code + output) -----")
        parts.append("")
        parts.extend(summaries)
    if manifest:
        if parts:
            parts.append("")
        parts.append(manifest)
    return "\n".join(parts).rstrip() + "\n"


# ── run_step node ───────────────────────────────────────────────────────


@timed_node("run_step")
def run_step(
    state: DeepResearchState, runtime: Runtime[PlanContext]
) -> DeepResearchState:
    """Invoke self_debug for one Step; collect outcome + summary; advance index."""
    plan = state["plan"]
    n = state.get("step_index", 1)
    step = plan.sub_tasks[n - 1]

    # Step banner — the interleaved [time]/[escalation] lines from inside
    # the nested self_debug run otherwise blur step boundaries together.
    bar = "━" * 70
    print(
        f"\n{bar}\n▶ STEP {n}/{len(plan.sub_tasks)}: {step.sub_task}\n{bar}",
        file=sys.stderr,
        flush=True,
    )

    summary_block = _render_previous_steps(
        state.get("step_summaries", []),
        _build_workspace_manifest(state.get("work_dir")),
    )

    # Dispatch on agent type. Both sub-graphs return a StepVerdict-shaped
    # `current_step_verdict`, so the downstream outcome-building logic stays
    # uniform. Anything we haven't implemented (idea_maker, etc.) halts cleanly.
    invoke_input = {
        "step": step,
        "work_dir": state.get("work_dir"),
        "step_number": n,
        "previous_steps_execution_summary": summary_block,
    }
    if step.sub_task_agent == "engineer":
        sub_state = self_debug_graph.invoke(invoke_input, context=runtime.context)
    elif step.sub_task_agent == "researcher":
        sub_state = researcher_graph.invoke(invoke_input, context=runtime.context)
    else:
        outcome = {
            "step_number": n,
            "fulfilled": False,
            "reason": (
                f"agent '{step.sub_task_agent}' not supported in v1 "
                f"(only `engineer` and `researcher` steps are implemented)"
            ),
        }
        return {
            "step_outcomes": state.get("step_outcomes", []) + [outcome],
            "step_index": n + 1,
        }

    step_verdict = sub_state.get("current_step_verdict")
    fulfilled = bool(getattr(step_verdict, "fulfilled", False))

    outcome = {
        "step_number": n,
        "fulfilled": fulfilled,
        "attempts": sub_state.get("attempts", 0),
        "escalated": sub_state.get("escalated", False),
    }
    if not fulfilled:
        # Surface the most useful diagnostic.
        ev = sub_state.get("current_execution_verdict")
        sv = step_verdict
        outcome["reason"] = (
            getattr(sv, "feedback", None)
            or (getattr(sv, "unmet_requirements", None) or [None])[0]
            or getattr(ev, "error_summary", None)
            or "step not fulfilled"
        )

    return {
        "step_summaries": state.get("step_summaries", []) + [
            _build_step_summary(step, sub_state)
        ],
        "step_outcomes": state.get("step_outcomes", []) + [outcome],
        "step_index": n + 1,
    }


# ── router ──────────────────────────────────────────────────────────────


def after_step(
    state: DeepResearchState, runtime: Runtime[PlanContext]
) -> str:
    """Halt on step failure; END when the plan is exhausted; else next step."""
    outcomes = state.get("step_outcomes", [])
    if outcomes and not outcomes[-1].get("fulfilled", False):
        return END
    if state.get("step_index", 1) > len(state["plan"].sub_tasks):
        return END
    return "run_step"


__all__ = ["run_step", "after_step"]
