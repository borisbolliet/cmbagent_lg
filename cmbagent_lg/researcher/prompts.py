"""Prompt loaders for the researcher loop. Same shape as self_debug/prompts.py."""

from typing import List, Optional

from cmbagent_lg.context import PlanContext
from cmbagent_lg.planning.schemas import Step
from cmbagent_lg.prompt_utils import SafeDict, load_yaml


_TEMPLATES = "cmbagent_lg.researcher.templates"
RESEARCHER_YAML = load_yaml(_TEMPLATES, "researcher.yaml")
STEP_EVALUATOR_YAML = load_yaml(_TEMPLATES, "step_evaluator.yaml")


def render_retry_context(
    attempts: int,
    max_attempts: int,
    last_report: Optional[str],
    last_step_unmet: Optional[List[str]],
    last_step_feedback: Optional[str],
) -> str:
    """Block injected into the researcher prompt on retries. Empty on first attempt.

    Researcher has only a goal-miss retry mode (no execution errors to surface),
    so this is simpler than self_debug's `render_retry_context`.
    """
    if attempts <= 1:
        return ""

    remaining = max_attempts - attempts + 1
    lines = [
        f"**Attempt {attempts} of {max_attempts} ({remaining} remaining).**",
        "",
    ]
    if attempts >= max_attempts:
        lines += ["**LAST CHANCE — if this attempt fails, the loop terminates.**", ""]

    if last_report:
        # Full prior report, no truncation — the researcher needs to see what
        # exactly the previous attempt wrote in order to fix the gaps without
        # discarding the good parts.
        lines += [
            "**Your previous attempt's report was:**",
            "```markdown",
            last_report,
            "```",
            "",
        ]

    lines += ["**Outcome:** the report did NOT fulfill the step.", ""]
    if last_step_unmet:
        lines.append("**Unmet requirements:**")
        for r in last_step_unmet:
            lines.append(f"  - {r}")
        lines.append("")
    if last_step_feedback:
        lines += ["**Evaluator feedback:**", last_step_feedback, ""]
    return "\n".join(lines)


def researcher_instructions(
    ctx: PlanContext,
    step: Step,
    retry_context: str,
    previous_steps_execution_summary: str = "",
) -> str:
    return RESEARCHER_YAML["instructions"].format_map(
        SafeDict(
            main_task=ctx.main_task,
            researcher_append_instructions=ctx.researcher_append_instructions,
            current_sub_task=step.sub_task,
            current_instructions="\n".join(f"- {b}" for b in step.bullet_points),
            retry_context=retry_context,
            previous_steps_execution_summary=previous_steps_execution_summary,
        )
    )


def _render_step_history(step_feedback_history: List[str]) -> str:
    if not step_feedback_history:
        return "(this is the first attempt — no prior feedback on this step)"
    return "\n".join(step_feedback_history)


def step_evaluator_instructions(
    ctx: PlanContext,
    step: Step,
    current_report: str,
    step_feedback_history: List[str],
) -> str:
    return STEP_EVALUATOR_YAML["instructions"].format_map(
        SafeDict(
            evaluator_append_instructions=ctx.evaluator_append_instructions,
            current_sub_task=step.sub_task,
            current_instructions="\n".join(f"- {b}" for b in step.bullet_points),
            current_report=current_report,
            step_feedback_history=_render_step_history(step_feedback_history),
        )
    )
