"""Compiled adaptive-planning graph — execute a plan step-by-step, rewriting
the remaining steps after any step whose result warrants it.

    START → execute_step → adaptive_review ─► replan_tail → merge_plan ─┐
                                 │  (needs_adaptation)                   │
                                 │                                       │
                                 ├─► execute_step  (next step) ◄─────────┘
                                 └─► END           (plan done or step failed)

The planning loop in `cmbagent_lg.planning` produces the initial `Plan`; this
graph consumes it. `execute_step` is deep_research's `run_step`; the adaptive
branch (`adaptive_review → replan_tail → merge_plan`) revises the tail of the
plan in place. Only the not-yet-executed steps may change; the completed prefix
and its outputs are held fixed.
"""

from langgraph.graph import StateGraph, START, END

from cmbagent_lg.context import PlanContext
from cmbagent_lg.adaptive_planning.state import AdaptivePlanState
from cmbagent_lg.adaptive_planning.nodes import (
    execute_step,
    adaptive_review,
    replan_tail,
    merge_plan,
    route_after_review,
    route_after_merge,
)


graph = (
    StateGraph(AdaptivePlanState, context_schema=PlanContext)
    .add_node("execute_step", execute_step)
    .add_node("adaptive_review", adaptive_review)
    .add_node("replan_tail", replan_tail)
    .add_node("merge_plan", merge_plan)
    .add_edge(START, "execute_step")
    .add_edge("execute_step", "adaptive_review")
    .add_conditional_edges(
        "adaptive_review",
        route_after_review,
        {"replan_tail": "replan_tail", "execute_step": "execute_step", END: END},
    )
    .add_edge("replan_tail", "merge_plan")
    .add_conditional_edges(
        "merge_plan",
        route_after_merge,
        {"execute_step": "execute_step", END: END},
    )
    .compile()
)
