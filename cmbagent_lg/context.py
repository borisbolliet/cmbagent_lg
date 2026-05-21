"""Run-scoped variables (LangGraph `context_schema`).

These are the values the cmbagent YAML templates reference as `{improved_main_task}`,
`{hardware_constraints}`, etc. We pass them at `graph.invoke` time, not as graph state,
so a single compiled graph generalizes across tasks.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# Agent inventory, exposed both to the planner/reviewer prompts AND used to
# dynamically constrain `Step.sub_task_agent` at format time (via a runtime
# `Literal[*names]` built inside the formatter).
#
# Default mirrors cmbagent's hard-coded set (planner_response_evaluator.Subtasks)
# so plans produced here are drop-in compatible with cmbagent's control phase.
DEFAULT_AVAILABLE_AGENTS: List[Tuple[str, str]] = [
    ("engineer", "Writes and runs Python code: numeric work, data I/O, computations, plotting."),
    ("researcher", "Reads, reasons, writes prose; no code execution."),
]


@dataclass
class PlanContext:
    improved_main_task: str
    hardware_constraints: str = "Standard laptop. Single CPU. No GPU."
    code_execution_timeout: int | None = 120
    maximum_number_of_steps_in_plan: int = 5
    planner_append_instructions: str = ""
    plan_reviewer_append_instructions: str = ""
    # Number of review cycles. Total planner passes = num_rounds + 1
    # (the extra pass is the final, un-reviewed plan).
    num_rounds: int = 2
    # `(name, one-line description)` pairs. Override per run to expose a
    # different team to the planner (e.g. add `("plotter", "Renders plots.")`).
    available_agents: List[Tuple[str, str]] = field(
        default_factory=lambda: list(DEFAULT_AVAILABLE_AGENTS)
    )

    # ── self_debug module ────────────────────────────────────────────────
    # Max engineer attempts per sub-task before giving up. Mirrors
    # cmbagent's `max_n_attempts` (default 3 there too).
    max_n_attempts: int = 3
    engineer_append_instructions: str = ""
    evaluator_append_instructions: str = ""
