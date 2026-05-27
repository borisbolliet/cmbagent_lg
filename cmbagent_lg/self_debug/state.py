"""Graph state for the self_debug loop.

Flow — two gates plus an opt-in escape hatch (see `self_debug/graph.py`):

    engineer ─► format_engineer ─► executor ─► execution_evaluator
        ▲                                              │
        │       code SUCCESS ──────────────────────────┼──► step_evaluator
        │                                              │           │
        │       code FAILURE → engineer (retry/END)    │           ├── goal MET → END
        │                                              │           └── goal MISS → engineer
        │       escalatable failure (opt-in)           │
        └────── escalation ──► executor ◄──────────────┘

`raw_engineer` is the proposer's free-form prose; `current_code` is the
formatter's structured output. `error_history` accumulates one summary per
code-execution failure; `step_feedback_history` accumulates one per goal-miss
so each evaluator sees its own prior reasoning (cmbagent's `error_history`
pattern — supports strategic pivots on later attempts).
"""

import operator
from typing import TypedDict, List, Dict, Optional, Annotated

from cmbagent_lg.planning.schemas import Step
from cmbagent_lg.self_debug.schemas import (
    EngineerResponse,
    ExecutionVerdict,
    StepVerdict,
)
from cmbagent_lg.timing import NodeTiming


class DebugState(TypedDict, total=False):
    # Input
    step: Step
    # Plan-step index — drives the on-disk filename `codebase/step_{N}.py`
    # (mirrors cmbagent's `current_plan_step_number`). Defaults to 1 for
    # standalone runs; deep_research passes the real index when it wires
    # planner → self_debug.
    step_number: int

    # Retry budget
    attempts: int  # bumped by the engineer node at the start of each attempt
    # Escalation is one-shot per step: set True once the escalation node has
    # run, so the router won't escalate the same step twice.
    escalated: bool
    escalation_reason: Optional[str]

    # Engineer
    raw_engineer: str
    current_code: EngineerResponse

    # Executor
    execution_stdout: str
    execution_stderr: str
    execution_returncode: int
    execution_timed_out: bool
    execution_elapsed_s: float  # subprocess wall clock (separate from node total)
    # Snapshot of data/ (filename → mtime) taken before this step's first
    # attempt — lets the executor attribute new/modified files to this step
    # in the data manifest, without touching the files or their names.
    data_baseline: Dict[str, float]
    # Files this step produced (path/bytes/modified) — what step_evaluator sees.
    data_manifest: List[dict]

    # execution_evaluator — did the code RUN cleanly?
    current_execution_verdict: ExecutionVerdict

    # step_evaluator — did the run ACHIEVE the sub-task's goal?
    current_step_verdict: StepVerdict

    # Accumulated across the loop — one entry per failed attempt.
    # `error_history`         — code-execution failures (execution_evaluator)
    # `step_feedback_history` — goal-misses (step_evaluator), so step_evaluator
    #                           sees its own prior feedback on later attempts
    error_history: List[str]
    step_feedback_history: List[str]

    # Bookkeeping
    work_dir: Optional[str]
    # appended per node pass via @timed_node; concatenated by operator.add
    node_elapsed_s: Annotated[List[NodeTiming], operator.add]
    # Set by `deep_research` when running this step as part of a multi-step
    # plan: per-step blocks of prior steps' code+output + a workspace file
    # manifest. The engineer node injects it into its system prompt so step N
    # can reference / import step N-1's outputs. Empty (or absent) for
    # standalone self_debug runs.
    previous_steps_execution_summary: Optional[str]
