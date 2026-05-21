"""Compiled self-debug graph — two gates, one shared retry budget.

    engineer ─► format_engineer ─► executor ─► execution_evaluator
                                                       │
                            code FAILURE ◄─────────────┤
                            (retry / exhaust→END)      │ code SUCCESS
                                                       ▼
                                                 step_evaluator
                                                       │
                            goal NOT met ◄─────────────┤
                            (retry / exhaust→END)      │ goal MET
                                                       ▼
                                                      END

`execution_evaluator` decides whether the code ran cleanly; `step_evaluator`
decides whether the sub-task's goal was achieved. Total engineer passes =
at most `max_n_attempts` (from `PlanContext`), shared across both failure
modes.
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


graph = (
    StateGraph(DebugState, context_schema=PlanContext)
    .add_node("engineer", engineer)
    .add_node("format_engineer", format_engineer)
    .add_node("executor", executor)
    .add_node("execution_evaluator", execution_evaluator)
    .add_node("step_evaluator", step_evaluator)
    .add_edge(START, "engineer")
    .add_edge("engineer", "format_engineer")
    .add_edge("format_engineer", "executor")
    .add_edge("executor", "execution_evaluator")
    .add_conditional_edges(
        "execution_evaluator",
        route_after_execution_evaluator,
        {"engineer": "engineer", "step_evaluator": "step_evaluator", END: END},
    )
    .add_conditional_edges(
        "step_evaluator",
        route_after_step_evaluator,
        {"engineer": "engineer", END: END},
    )
    .compile()
)
