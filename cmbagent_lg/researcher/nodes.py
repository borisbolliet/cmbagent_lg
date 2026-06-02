"""Researcher + researcher_step_evaluator nodes for the researcher sub-graph.

Mirrors `self_debug/nodes.py` but for prose:
  - `researcher` writes a markdown report (raw LLM `.invoke()` — no structured
    output; the report content IS the deliverable). Persists to
    `reports/step_{N}.md`.
  - `step_evaluator` reads that report and emits a `StepVerdict` via the same
    `_critic().with_structured_output(StepVerdict)` pattern self_debug uses.

A goal-miss demotes the canonical `reports/step_{N}.md` to
`reports/step_{N}_failure_{attempt}.md` so the audit trail covers every
attempt (parallel to engineer's `codebase/step_{N}_failure_{I}.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime

from cmbagent_lg.context import PlanContext
from cmbagent_lg.llms import chat_model
from cmbagent_lg.prompt_utils import flatten_content, schema_field_brief
from cmbagent_lg.vlm import collect_images, with_images
from cmbagent_lg.researcher.prompts import (
    render_retry_context,
    researcher_instructions,
    step_evaluator_instructions,
)
from cmbagent_lg.researcher.state import ResearcherState
from cmbagent_lg.self_debug.schemas import StepVerdict
from cmbagent_lg.timing import timed_node


# Per-role chat models from the run context (None → llms._DEFAULT_MODEL).
# Here the "generator" is the researcher and the "critic" is the evaluator.
# chat_model caches by (model, role), so this stays lazy.
def _proposer(ctx: PlanContext):
    return chat_model(ctx.researcher_model, "generator")


def _critic(ctx: PlanContext):
    return chat_model(ctx.evaluator_model, "critic")


def _reports_dir(state: ResearcherState) -> Optional[Path]:
    """`{work_dir}/reports/` — markdown reports live here. None if no work_dir."""
    raw = state.get("work_dir")
    if not raw:
        return None
    out = Path(raw).expanduser() / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── researcher ─────────────────────────────────────────────────────────


@timed_node("researcher")
def researcher(state: ResearcherState, runtime: Runtime[PlanContext]) -> ResearcherState:
    """Write the markdown report for the current sub-task. Bumps `attempts`,
    persists to `reports/step_{N}.md`, returns `current_report` + `report_path`.
    """
    ctx = runtime.context
    step = state["step"]
    attempts = state.get("attempts", 0) + 1

    last_report = None
    last_step_unmet = None
    last_step_feedback = None
    if attempts > 1:
        last_report = state.get("current_report")
        prior_verdict = state.get("current_step_verdict")
        if prior_verdict is not None:
            last_step_unmet = prior_verdict.unmet_requirements
            last_step_feedback = prior_verdict.feedback

    retry_block = render_retry_context(
        attempts=attempts,
        max_attempts=ctx.max_n_attempts,
        last_report=last_report,
        last_step_unmet=last_step_unmet,
        last_step_feedback=last_step_feedback,
    )

    system = researcher_instructions(
        ctx,
        step,
        retry_block,
        previous_steps_execution_summary=state.get("previous_steps_execution_summary") or "",
    )
    user = (
        "Write the markdown report addressing the current sub-task and every "
        "bullet-point requirement. Output only the report body — no preamble."
    )

    # Multimodal grounding: attach the generated plots so the report is written
    # against the actual figures, not inferred from code/stdout alone.
    user_content = user
    if getattr(ctx, "vlm_enabled", False):
        images = collect_images(state.get("work_dir"), getattr(ctx, "vlm_max_images", 8))
        if images:
            user_content = with_images(
                user + " The figures produced by the analysis are attached below; "
                "read quantitative trends directly from them and make sure your "
                "report is consistent with what the plots show.",
                images,
            )

    msg = _proposer(runtime.context).invoke(
        [SystemMessage(system), HumanMessage(content=user_content)],
        config={"tags": ["researcher"]},
    )
    report = flatten_content(msg.content).strip()

    n = state.get("step_number", 1)
    report_path = ""
    reports = _reports_dir(state)
    if reports is not None:
        out = reports / f"step_{n}.md"
        out.write_text(report)
        report_path = str(out)

    return {
        "attempts": attempts,
        "current_report": report,
        "report_path": report_path,
    }


# ── step_evaluator ─────────────────────────────────────────────────────


@timed_node("researcher_step_evaluator")
def step_evaluator(
    state: ResearcherState, runtime: Runtime[PlanContext]
) -> ResearcherState:
    """Did the report ACHIEVE the sub-task's goal? One structured-output call
    → StepVerdict, reading the report itself (no code / stdout)."""
    ctx = runtime.context
    step = state["step"]
    report = state.get("current_report", "")
    prior_feedback = state.get("step_feedback_history", [])

    system = step_evaluator_instructions(
        ctx, step, current_report=report, step_feedback_history=prior_feedback,
    )
    user = (
        "Judge whether the researcher's report fulfills the step goal and emit "
        "a verdict. Cover these fields:\n\n" + schema_field_brief(StepVerdict)
    )
    structured = _critic(runtime.context).with_structured_output(StepVerdict)
    verdict: StepVerdict = structured.invoke(
        [SystemMessage(system), HumanMessage(user)],
        config={"tags": ["researcher_step_evaluator"]},
    )

    n = state.get("step_number", 1)
    attempt = state.get("attempts", 0)

    # On a goal-miss, record this attempt's feedback for the next pass.
    feedback_history = list(prior_feedback)
    if not verdict.fulfilled:
        bits = []
        if verdict.unmet_requirements:
            bits.append("unmet — " + "; ".join(verdict.unmet_requirements))
        if verdict.feedback:
            bits.append("feedback — " + verdict.feedback)
        feedback_history.append(
            f"attempt {attempt}: " + (" | ".join(bits) if bits else "goal not met")
        )

    # Persist the verdict next to engineer verdicts under logs/, with a
    # researcher-prefixed name to avoid colliding with engineer step verdicts
    # if a mixed plan ever needed both at the same step number.
    raw = state.get("work_dir")
    if raw:
        logs = Path(raw).expanduser() / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"step_{n}_researcher_verdict.json").write_text(
            verdict.model_dump_json(indent=2)
        )

    # Goal-miss: demote the canonical report to a failure-variant for the audit
    # trail (parallel to engineer's `step_{N}_failure_{I}.py`).
    if not verdict.fulfilled:
        reports = _reports_dir(state)
        if reports is not None:
            canonical = reports / f"step_{n}.md"
            if canonical.exists():
                canonical.rename(reports / f"step_{n}_failure_{attempt}.md")

    return {
        "current_step_verdict": verdict,
        "step_feedback_history": feedback_history,
    }


# ── router ──────────────────────────────────────────────────────────────


def route_after_step_evaluator(
    state: ResearcherState, runtime: Runtime[PlanContext]
) -> str:
    """Goal gate: fulfilled → END; not fulfilled & attempts left → researcher;
    not fulfilled & exhausted → END."""
    ctx = runtime.context
    if state["current_step_verdict"].fulfilled:
        return END
    if state.get("attempts", 0) >= ctx.max_n_attempts:
        return END
    return "researcher"


__all__ = ["researcher", "step_evaluator", "route_after_step_evaluator"]
