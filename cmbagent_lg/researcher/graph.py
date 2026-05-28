"""Compiled researcher graph — single gate, shared retry budget.

    researcher ─► step_evaluator
        ▲                 │
        │  goal MET ──────┤ → END
        │  goal MISS ─────┘ → researcher (until attempts == max_n_attempts)

No execution_evaluator (no code), no escalation (no missing-package failure
modes). Total researcher passes = at most `ctx.max_n_attempts`.
"""

from langgraph.graph import StateGraph, START, END

from cmbagent_lg.context import PlanContext
from cmbagent_lg.researcher.nodes import (
    researcher,
    step_evaluator,
    route_after_step_evaluator,
)
from cmbagent_lg.researcher.state import ResearcherState


graph = (
    StateGraph(ResearcherState, context_schema=PlanContext)
    .add_node("researcher", researcher)
    .add_node("step_evaluator", step_evaluator)
    .add_edge(START, "researcher")
    .add_edge("researcher", "step_evaluator")
    .add_conditional_edges(
        "step_evaluator",
        route_after_step_evaluator,
        {"researcher": "researcher", END: END},
    )
    .compile()
)
