"""Graph state for the researcher loop.

Flow — single gate, bounded retry:

    researcher ─► step_evaluator
        ▲                 │
        │   goal MET ─────┤ → END
        │   goal MISS ────┘ → researcher (until attempts == max_n_attempts)

`current_report` is the raw markdown the LLM wrote (no structured output —
the report content IS the deliverable; see ../researcher/__init__.py).
`step_feedback_history` accumulates one entry per goal-miss so the
step_evaluator sees its own prior reasoning across retries (same pattern as
self_debug's DebugState).
"""

import operator
from typing import TypedDict, List, Optional, Annotated

from cmbagent_lg.planning.schemas import Step
from cmbagent_lg.self_debug.schemas import StepVerdict
from cmbagent_lg.timing import NodeTiming


class ResearcherState(TypedDict, total=False):
    # Input
    step: Step
    step_number: int  # drives the on-disk filename reports/step_{N}.md
    work_dir: Optional[str]
    # Set by deep_research when this step runs as part of a multi-step plan
    # (prior steps' code+output / reports + a workspace file manifest). Empty
    # for standalone researcher_graph invocations.
    previous_steps_execution_summary: Optional[str]

    # Retry budget
    attempts: int  # bumped by the researcher node at the start of each attempt

    # Researcher output
    current_report: str          # raw markdown string the LLM emitted
    report_path: str             # absolute path written to disk

    # step_evaluator — did the report ACHIEVE the sub-task's goal?
    current_step_verdict: StepVerdict
    # One entry per goal-miss; surfaced to the next step_evaluator pass.
    step_feedback_history: List[str]

    # Bookkeeping
    node_elapsed_s: Annotated[List[NodeTiming], operator.add]
