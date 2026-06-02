"""Compiled self-debug graph — two gates, one shared retry budget.

    engineer ─► format_engineer ─► executor ─► execution_evaluator
        ▲                                              │
        │              code SUCCESS ───────────────────┤
        │                                              ▼
        │                                   image_reviewer (if vlm_enabled)
        │   figure needs revision ◄─────────────────── │  (else → step_evaluator)
        │   (+ budget) → engineer                       ▼
        │                                          step_evaluator
        │   generic failure (retry / exhaust→END) ◄──┐       │
        │                                            │ goal MET → END
        │   escalatable failure (missing package /   │ goal MISS → engineer
        │   renamed API), once per step  ──► escalation ──► executor
        └──────────────────────────────────────────────┘

`execution_evaluator` decides whether the code ran cleanly; `image_reviewer`
(opt-in via `PlanContext.vlm_enabled`) visually reviews the figures and runs a
bounded revise-the-plot loop back to the engineer; `step_evaluator` decides
whether the sub-task's goal was achieved. `escalation` (opt-in via
`PlanContext.enable_escalation`) is the escape hatch: a missing-package or
renamed-API failure is handed once to a free-form Claude Agent SDK agent that
can web-search the fix, then control returns to the executor.

Total engineer passes = at most `max_n_attempts`; escalation does not consume
an attempt. Plot-revision cycles are additionally bounded by
`max_vlm_review_attempts`.
"""

from langgraph.graph import StateGraph, START, END

from cmbagent_lg.context import PlanContext
from cmbagent_lg.self_debug.state import DebugState
from cmbagent_lg.self_debug.nodes import (
    engineer,
    format_engineer,
    executor,
    execution_evaluator,
    step_evaluator,
    route_after_execution_evaluator,
    route_after_step_evaluator,
)
from cmbagent_lg.self_debug.escalation import escalation
from cmbagent_lg.vlm.reviewer import image_reviewer, route_after_image_reviewer


graph = (
    StateGraph(DebugState, context_schema=PlanContext)
    .add_node("engineer", engineer)
    .add_node("format_engineer", format_engineer)
    .add_node("executor", executor)
    .add_node("execution_evaluator", execution_evaluator)
    .add_node("image_reviewer", image_reviewer)
    .add_node("step_evaluator", step_evaluator)
    .add_node("escalation", escalation)
    .add_edge(START, "engineer")
    .add_edge("engineer", "format_engineer")
    .add_edge("format_engineer", "executor")
    .add_edge("executor", "execution_evaluator")
    .add_conditional_edges(
        "execution_evaluator",
        route_after_execution_evaluator,
        {
            "engineer": "engineer",
            "image_reviewer": "image_reviewer",
            "step_evaluator": "step_evaluator",
            "escalation": "escalation",
            END: END,
        },
    )
    .add_edge("escalation", "executor")
    .add_conditional_edges(
        "image_reviewer",
        route_after_image_reviewer,
        {"engineer": "engineer", "step_evaluator": "step_evaluator"},
    )
    .add_conditional_edges(
        "step_evaluator",
        route_after_step_evaluator,
        {"engineer": "engineer", END: END},
    )
    .compile()
)
