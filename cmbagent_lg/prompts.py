"""Prompt loading + schema-driven field briefs.

The YAML templates use cmbagent's `{improved_main_task}`, `{recommendations}`,
`{proposed_plan}`, etc. placeholders. We render them with `str.format_map(...)`
against a dict built from `PlanContext` + current `PlanState`. Missing keys
become empty strings (so the templates don't break mid-loop).
"""

from importlib import resources
import yaml

from cmbagent_lg.context import PlanContext


class _SafeDict(dict):
    """`format_map` helper: missing keys render as empty strings instead of KeyError."""

    def __missing__(self, key):
        return ""


def _load_yaml(name: str) -> dict:
    text = resources.files("cmbagent_lg.templates").joinpath(name).read_text()
    return yaml.safe_load(text)


PLANNER_YAML = _load_yaml("planner.yaml")
PLAN_REVIEWER_YAML = _load_yaml("plan_reviewer.yaml")


def planner_instructions(ctx: PlanContext, recommendations: str) -> str:
    return PLANNER_YAML["instructions"].format_map(
        _SafeDict(
            improved_main_task=ctx.improved_main_task,
            planner_append_instructions=ctx.planner_append_instructions,
            hardware_constraints=ctx.hardware_constraints,
            code_execution_timeout=ctx.code_execution_timeout,
            maximum_number_of_steps_in_plan=ctx.maximum_number_of_steps_in_plan,
            recommendations=recommendations or "(none yet — this is the first plan)",
        )
    )


def plan_reviewer_instructions(ctx: PlanContext, proposed_plan: str) -> str:
    return PLAN_REVIEWER_YAML["instructions"].format_map(
        _SafeDict(
            improved_main_task=ctx.improved_main_task,
            plan_reviewer_append_instructions=ctx.plan_reviewer_append_instructions,
            hardware_constraints=ctx.hardware_constraints,
            code_execution_timeout=ctx.code_execution_timeout,
            maximum_number_of_steps_in_plan=ctx.maximum_number_of_steps_in_plan,
            proposed_plan=proposed_plan,
        )
    )


def schema_field_brief(schema) -> str:
    """One bullet per Pydantic field, suitable for dropping into a generator prompt."""
    lines = []
    for name, field in schema.model_fields.items():
        t = str(field.annotation).replace("typing.", "")
        if t.startswith("<class '") and t.endswith("'>"):
            t = t[len("<class '") : -len("'>")]
        lines.append(f"- {name} ({t}): {field.description or ''}")
    return "\n".join(lines)
