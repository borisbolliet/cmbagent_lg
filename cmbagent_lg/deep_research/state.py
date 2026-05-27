"""Graph state for the deep_research orchestrator.

Flow:

    START → run_step → after_step ─► run_step (next)
                                  └─► END (plan done or step failed)

`step_summaries` accumulates one per-step block (header + executed code +
stdout) per *completed* step — mirrors cmbagent's `_step_summaries_accumulator`
(`cmbagent/functions/status.py:651-654`). Combined with a freshly-scanned
workspace file manifest, this is what the engineer at step N+1 sees in its
prompt so it can `from step_1 import …` or load `data/foo.csv`.

`step_outcomes` is the structured per-step result we report at the end and
that the router consults to decide advance-vs-halt.
"""

import operator
from typing import TypedDict, List, Optional, Annotated

from cmbagent_lg.planning.schemas import Plan
from cmbagent_lg.timing import NodeTiming


class DeepResearchState(TypedDict, total=False):
    # Input
    plan: Plan
    work_dir: Optional[str]

    # 1-based plan-step pointer. Mirrors cmbagent's `current_plan_step_number`.
    step_index: int

    # Accumulated across the loop.
    step_summaries: List[str]   # one block per completed step (cmbagent style)
    step_outcomes: List[dict]   # {step_number, fulfilled, attempts, escalated, reason?}

    # Bookkeeping. Per-node wall-clock concatenated by operator.add — matches
    # the pattern in PlanState / DebugState.
    node_elapsed_s: Annotated[List[NodeTiming], operator.add]
