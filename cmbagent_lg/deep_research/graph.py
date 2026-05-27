"""Compiled deep_research graph — one Plan in, executed step-by-step out.

    START → run_step → after_step ─► run_step  (next step)
                                  └─► END        (plan done or step failed)

`run_step` is a thin wrapper that invokes `self_debug_graph` once per `Step`
of the input `Plan`, threading an accumulated `previous_steps_execution_summary`
through so each engineer can see prior steps' code/output and the workspace
file manifest. Mirrors cmbagent's `cmbagent/workflows/deep_research.py:688-958`
but as a langgraph subgraph that composes the existing modules.
"""

from langgraph.graph import StateGraph, START, END

from cmbagent_lg.context import PlanContext
from cmbagent_lg.deep_research.nodes import run_step, after_step
from cmbagent_lg.deep_research.state import DeepResearchState


graph = (
    StateGraph(DeepResearchState, context_schema=PlanContext)
    .add_node("run_step", run_step)
    .add_edge(START, "run_step")
    .add_conditional_edges(
        "run_step",
        after_step,
        {"run_step": "run_step", END: END},
    )
    .compile()
)
