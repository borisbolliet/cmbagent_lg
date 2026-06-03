"""Graph state for the adaptive-planning orchestrator.

A superset of `DeepResearchState`: it carries the same step-by-step execution
bookkeeping (so the reused `run_step` node works unchanged) plus the fields the
adaptive-review branch threads through.

    plan                  the (mutable) Plan — the tail may be rewritten mid-run
    step_index            1-based pointer; advances after each executed step
    step_summaries        one text block per completed step (cross-step context)
    step_outcomes         structured per-step result {step_number, fulfilled, ...}
    adaptive_review       the latest reviewer verdict (routes the conditional edge)
    new_tail              the replanned remaining steps, before they are merged
    adaptive_history      one record per applied revision {after_step, recommendations}
"""

import operator
from typing import TypedDict, List, Optional, Annotated

from cmbagent_lg.planning.schemas import Plan
from cmbagent_lg.adaptive_planning.schemas import AdaptiveReview
from cmbagent_lg.timing import NodeTiming


class AdaptivePlanState(TypedDict, total=False):
    # Input / mutable plan.
    plan: Plan
    work_dir: Optional[str]

    # Execution bookkeeping (shared shape with DeepResearchState).
    step_index: int
    step_summaries: List[str]
    step_outcomes: List[dict]

    # Adaptive-review branch.
    adaptive_review: Optional[AdaptiveReview]
    new_tail: Optional[Plan]
    adaptive_history: List[dict]

    # Per-node wall-clock, concatenated by operator.add (matches the other states).
    node_elapsed_s: Annotated[List[NodeTiming], operator.add]
