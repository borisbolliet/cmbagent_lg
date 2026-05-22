"""Prompt loaders for the self_debug loop. Same shape as planning/prompts.py."""

from importlib import resources
from typing import List
import yaml

from cmbagent_lg.context import PlanContext
from cmbagent_lg.planning.schemas import Step


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def _load_yaml(name: str) -> dict:
    text = resources.files("cmbagent_lg.self_debug.templates").joinpath(name).read_text()
    return yaml.safe_load(text)


ENGINEER_YAML = _load_yaml("engineer.yaml")
EVALUATOR_YAML = _load_yaml("evaluator.yaml")
STEP_EVALUATOR_YAML = _load_yaml("step_evaluator.yaml")


_TRUNCATE_HEAD = 1500
_TRUNCATE_TAIL = 500


def _head_tail(s: str, head: int = _TRUNCATE_HEAD, tail: int = _TRUNCATE_TAIL) -> str:
    """Truncate a long string to head + '\\n…\\n' + tail (cmbagent's HeadTailTokenTruncate idea)."""
    if not s or len(s) <= head + tail + 4:
        return s
    return s[:head] + "\n…[truncated]…\n" + s[-tail:]


def render_retry_context(
    attempts: int,
    max_attempts: int,
    last_code: str | None,
    last_stderr: str | None,
    last_stdout: str | None,
    last_returncode: int | None,
    last_timed_out: bool | None,
    last_fix_suggestion: str | None,
    error_history: List[str],
    last_step_unmet: List[str] | None = None,
    last_step_feedback: str | None = None,
) -> str:
    """Block injected into the engineer prompt on retries. Empty on first attempt.

    Two retry modes: a **code-error** retry (the script crashed) shows the
    error trail; a **goal-miss** retry (the script ran clean but the
    step_evaluator found the sub-task unfulfilled) shows the unmet
    requirements + reviewer feedback. The caller selects the mode by passing
    `last_step_*` (goal-miss) or leaving them None (code-error).
    """
    if attempts <= 1:
        return ""

    remaining = max_attempts - attempts + 1
    lines = [
        f"**Attempt {attempts} of {max_attempts} ({remaining} remaining).**",
        "",
    ]

    if attempts >= max_attempts:
        lines.append("**LAST CHANCE — if this attempt fails, the loop terminates.**")
        lines.append("")

    if last_code:
        lines += [
            "**The previous attempt's code was:**",
            "```python",
            _head_tail(last_code),
            "```",
            "",
        ]

    goal_miss = bool(last_step_unmet or last_step_feedback)
    if goal_miss:
        lines += [
            "**Outcome:** the code ran cleanly, but the step goal was NOT met.",
            "",
        ]
        if last_stdout:
            lines += ["**stdout:**", "```", _head_tail(last_stdout), "```", ""]
        if last_step_unmet:
            lines.append("**Unmet requirements:**")
            for r in last_step_unmet:
                lines.append(f"  - {r}")
            lines.append("")
        if last_step_feedback:
            lines += ["**Reviewer feedback:**", last_step_feedback, ""]
    else:
        if last_timed_out:
            lines.append("**Outcome:** timed out — the script exceeded the execution-time limit.")
        elif last_returncode is not None:
            lines.append(f"**Outcome:** returncode {last_returncode}")
        lines.append("")
        if last_stderr:
            lines += ["**stderr:**", "```", _head_tail(last_stderr), "```", ""]
        if last_stdout:
            lines += ["**stdout:**", "```", _head_tail(last_stdout), "```", ""]
        if last_fix_suggestion:
            lines += ["**Evaluator's fix suggestion:**", last_fix_suggestion, ""]
        if len(error_history) > 1:
            lines.append("**Cumulative error history (do not repeat fixes that already failed):**")
            for i, err in enumerate(error_history[:-1], start=1):
                lines.append(f"  - attempt {i}: {err}")
            lines.append("")

    return "\n".join(lines)


def engineer_instructions(
    ctx: PlanContext,
    step: Step,
    retry_context: str,
) -> str:
    return ENGINEER_YAML["instructions"].format_map(
        _SafeDict(
            improved_main_task=ctx.improved_main_task,
            engineer_append_instructions=ctx.engineer_append_instructions,
            hardware_constraints=ctx.hardware_constraints,
            code_execution_timeout=step.code_execution_timeout or ctx.code_execution_timeout,
            current_sub_task=step.sub_task,
            current_instructions="\n".join(f"- {b}" for b in step.bullet_points),
            retry_context=retry_context,
        )
    )


def evaluator_instructions(
    ctx: PlanContext,
    step: Step,
    executed_code: str,
    stdout: str,
    stderr: str,
    returncode: int,
    timed_out: bool,
) -> str:
    return EVALUATOR_YAML["instructions"].format_map(
        _SafeDict(
            evaluator_append_instructions=ctx.evaluator_append_instructions,
            code_execution_timeout=step.code_execution_timeout or ctx.code_execution_timeout,
            current_sub_task=step.sub_task,
            executed_code=_head_tail(executed_code),
            returncode=returncode,
            timed_out=timed_out,
            stdout=_head_tail(stdout),
            stderr=_head_tail(stderr),
        )
    )


def _render_manifest(data_manifest: List[dict]) -> str:
    if not data_manifest:
        return "(no output files were produced)"
    return "\n".join(
        f"- {f['path']} ({f['bytes']} bytes)" for f in data_manifest
    )


def _render_step_history(step_feedback_history: List[str]) -> str:
    if not step_feedback_history:
        return "(this is the first attempt — no prior feedback on this step)"
    return "\n".join(step_feedback_history)


def step_evaluator_instructions(
    ctx: PlanContext,
    step: Step,
    stdout: str,
    data_manifest: List[dict],
    step_feedback_history: List[str],
) -> str:
    return STEP_EVALUATOR_YAML["instructions"].format_map(
        _SafeDict(
            evaluator_append_instructions=ctx.evaluator_append_instructions,
            current_sub_task=step.sub_task,
            current_instructions="\n".join(f"- {b}" for b in step.bullet_points),
            stdout=_head_tail(stdout),
            data_manifest=_render_manifest(data_manifest),
            step_feedback_history=_render_step_history(step_feedback_history),
        )
    )
