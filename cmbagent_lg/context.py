"""Run-scoped variables (LangGraph `context_schema`).

These are the values the cmbagent YAML templates reference as `{main_task}`,
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
    main_task: str
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

    # ── escalation (self_debug escape hatch) ─────────────────────────────
    # When the strict loop hits a failure it structurally can't fix (missing
    # package, renamed/removed API), escalate once to a free-form Claude
    # Agent SDK agent that can web-search the fix. Opt-in: it runs Claude
    # models (the one Anthropic dependency — needs ANTHROPIC_API_KEY) and is
    # bounded by the budget/turn caps below.
    enable_escalation: bool = False
    escalation_max_budget_usd: float = 0.50
    escalation_max_turns: int = 12
    escalation_append_instructions: str = ""
    # Which Claude model the escalation agent uses. None = SDK default
    # (currently Sonnet — capable but expensive). Set to a Haiku for ~5-10x
    # cheaper runs; reach for Sonnet/Opus only for subtler migrations.
    escalation_model: str | None = None
