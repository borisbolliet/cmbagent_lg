"""Prompt builders for the adaptive reviewer and the tail replanner."""

from __future__ import annotations

from typing import List

from cmbagent_lg.context import PlanContext


def adaptive_reviewer_instructions(ctx: PlanContext) -> str:
    return (
        "You are the adaptive plan reviewer in a step-by-step research system.\n"
        f"The overall task is:\n{ctx.main_task}\n\n"
        "A plan is being executed one step at a time. You are called right after "
        "a step finishes. Given a summary of what that step produced and the list "
        "of steps that have not yet run, decide whether the remaining steps should "
        "be rewritten so the plan still makes sense in light of the latest result.\n\n"
        "Adapt only when the new information genuinely changes what should happen "
        "next (e.g. a result invalidates a later step, or suggests a better one). "
        "If the remaining steps are still appropriate, do not adapt."
    )


def adaptive_reviewer_user(last_summary: str, last_outcome: dict, remaining_fmt: str) -> str:
    return (
        "----- STEP JUST COMPLETED (summary) -----\n"
        f"{last_summary}\n\n"
        f"Outcome: {last_outcome}\n\n"
        "----- REMAINING STEPS (not yet executed) -----\n"
        f"{remaining_fmt}\n\n"
        "Decide whether the remaining steps need to be rewritten. Set "
        "`needs_adaptation` accordingly, and if True give concrete "
        "`recommendations` for the remaining steps only."
    )


def adaptive_replanner_instructions(ctx: PlanContext) -> str:
    return (
        "You are the planner in an adaptive research system, rewriting the tail "
        "of a plan mid-execution.\n"
        f"The overall task is:\n{ctx.main_task}\n\n"
        "The already-completed steps are FIXED and must not be reproposed. Produce "
        "a new ordered list of steps to REPLACE the remaining (not-yet-executed) "
        "steps, incorporating the reviewer's recommendations. Keep the new tail "
        "focused and minimal — only what is still needed to finish the task."
    )


def adaptive_replanner_user(
    completed_fmt: str, remaining_fmt: str, recommendations: List[str]
) -> str:
    recs = "\n".join(f"- {r}" for r in recommendations) or "(none)"
    return (
        "----- COMPLETED STEPS (fixed — do not reproduce) -----\n"
        f"{completed_fmt}\n\n"
        "----- CURRENT REMAINING STEPS (to be replaced) -----\n"
        f"{remaining_fmt}\n\n"
        "----- REVIEWER RECOMMENDATIONS -----\n"
        f"{recs}\n\n"
        "Now output the replacement tail as a Plan (ordered steps)."
    )
