"""Adaptive-planning nodes: execute a step, review, maybe rewrite the tail.

The control loop is the deep_research one, with an adaptive-review branch added
after each step:

    execute_step ─► adaptive_review ─(needs_adaptation?)─► replan_tail ─► merge_plan ─┐
                          │ no                                                         │
                          └─────────────────────────────► execute_step / END ◄────────┘

`execute_step` is deep_research's `run_step` reused verbatim — it invokes
`self_debug` / `researcher` for the current step, appends the step summary and
outcome, and advances `step_index`. The three new nodes implement adaptation:

  - `adaptive_review`  — an LLM reviewer reads the latest step summary and the
    remaining steps and emits a typed `AdaptiveReview(needs_adaptation, ...)`.
  - `replan_tail`      — when adaptation is requested, an LLM planner rewrites
    the remaining (not-yet-executed) steps given the recommendations.
  - `merge_plan`       — splices `completed_prefix + new_tail` back into `plan`.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime

from cmbagent_lg.context import PlanContext
from cmbagent_lg.llms import chat_model
from cmbagent_lg.planning.nodes import _build_dynamic_plan_model
from cmbagent_lg.planning.schemas import Plan
from cmbagent_lg.timing import timed_node

from cmbagent_lg.adaptive_planning.prompts import (
    adaptive_replanner_instructions,
    adaptive_replanner_user,
    adaptive_reviewer_instructions,
    adaptive_reviewer_user,
)
from cmbagent_lg.adaptive_planning.schemas import AdaptiveReview
from cmbagent_lg.adaptive_planning.state import AdaptivePlanState

# Execution node — reuse the deep_research step runner unchanged. It reads the
# same state keys (plan, step_index, step_summaries, step_outcomes) and returns
# the same updates, so it drops straight into the adaptive graph.
from cmbagent_lg.deep_research.nodes import run_step as execute_step


# ── adaptive review ───────────────────────────────────────────────────────


@timed_node("adaptive_review")
def adaptive_review(
    state: AdaptivePlanState, runtime: Runtime[PlanContext]
) -> AdaptivePlanState:
    """Decide whether the remaining plan should be rewritten after this step."""
    ctx = runtime.context
    plan = state["plan"]
    n_done = state.get("step_index", 1) - 1                 # steps executed so far
    remaining = plan.sub_tasks[n_done:]                     # the tail not yet run

    last_summary = (state.get("step_summaries") or ["(no summary)"])[-1]
    last_outcome = (state.get("step_outcomes") or [{}])[-1]
    remaining_fmt = (
        Plan(sub_tasks=remaining).format() if remaining else "(no steps remain)"
    )

    model = chat_model(ctx.plan_reviewer_model, "critic").with_structured_output(
        AdaptiveReview
    )
    review: AdaptiveReview = model.invoke(
        [
            SystemMessage(adaptive_reviewer_instructions(ctx)),
            HumanMessage(adaptive_reviewer_user(last_summary, last_outcome, remaining_fmt)),
        ],
        config={"tags": ["adaptive_review"]},
    )
    return {"adaptive_review": review}


# ── tail replan ───────────────────────────────────────────────────────────


@timed_node("replan_tail")
def replan_tail(
    state: AdaptivePlanState, runtime: Runtime[PlanContext]
) -> AdaptivePlanState:
    """Rewrite the remaining steps in light of the reviewer's recommendations."""
    ctx = runtime.context
    plan = state["plan"]
    n_done = state.get("step_index", 1) - 1
    remaining = plan.sub_tasks[n_done:]
    review = state["adaptive_review"]

    completed_fmt = "\n".join(state.get("step_summaries", [])) or "(none)"
    remaining_fmt = (
        Plan(sub_tasks=remaining).format() if remaining else "(no steps remain)"
    )

    DynPlan = _build_dynamic_plan_model(ctx.available_agents)
    model = chat_model(ctx.planner_model, "generator").with_structured_output(DynPlan)
    dyn = model.invoke(
        [
            SystemMessage(adaptive_replanner_instructions(ctx)),
            HumanMessage(
                adaptive_replanner_user(completed_fmt, remaining_fmt, review.recommendations)
            ),
        ],
        config={"tags": ["replan_tail"]},
    )
    return {"new_tail": Plan(**dyn.model_dump())}


# ── merge ─────────────────────────────────────────────────────────────────


@timed_node("merge_plan")
def merge_plan(
    state: AdaptivePlanState, runtime: Runtime[PlanContext]
) -> AdaptivePlanState:
    """Splice the fixed completed prefix and the freshly replanned tail."""
    plan = state["plan"]
    n_done = state.get("step_index", 1) - 1
    completed_prefix = plan.sub_tasks[:n_done]              # held fixed
    new_tail = state["new_tail"].sub_tasks
    merged = Plan(sub_tasks=completed_prefix + new_tail)

    history = list(state.get("adaptive_history", []))
    history.append(
        {
            "after_step": n_done,
            "recommendations": list(state["adaptive_review"].recommendations),
        }
    )
    return {"plan": merged, "adaptive_history": history}


# ── routers ───────────────────────────────────────────────────────────────


def route_after_review(state: AdaptivePlanState, runtime: Runtime[PlanContext]) -> str:
    """Halt on failure / when the plan is exhausted; else adapt-or-advance."""
    outcomes = state.get("step_outcomes", [])
    if outcomes and not outcomes[-1].get("fulfilled", False):
        return END
    if state.get("step_index", 1) > len(state["plan"].sub_tasks):
        return END
    review = state.get("adaptive_review")
    if review is not None and getattr(review, "needs_adaptation", False):
        return "replan_tail"
    return "execute_step"


def route_after_merge(state: AdaptivePlanState, runtime: Runtime[PlanContext]) -> str:
    """After merging the new tail, continue executing unless nothing remains."""
    if state.get("step_index", 1) > len(state["plan"].sub_tasks):
        return END
    return "execute_step"


__all__ = [
    "execute_step",
    "adaptive_review",
    "replan_tail",
    "merge_plan",
    "route_after_review",
    "route_after_merge",
]
