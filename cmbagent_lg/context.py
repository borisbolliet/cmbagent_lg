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

    # ── per-role model overrides ─────────────────────────────────────────
    # Each None falls back to llms._DEFAULT_MODEL. The provider is inferred
    # from the model name (gemini-* / gpt-* / o[1-4]* / claude-*), so these can
    # be passed straight through from a caller's config (e.g. Denario's
    # params.yaml). The node modules read the field matching their role:
    #   planner node      -> planner_model        (generator)
    #   plan_reviewer node -> plan_reviewer_model  (critic)
    #   engineer node     -> engineer_model        (generator)
    #   researcher node   -> researcher_model      (generator)
    #   execution/step evaluators -> evaluator_model (critic)
    #   all structured-output formatters -> formatter_model
    planner_model: str | None = None
    plan_reviewer_model: str | None = None
    engineer_model: str | None = None
    researcher_model: str | None = None
    evaluator_model: str | None = None
    formatter_model: str | None = None

    # ── multimodal grounding (cmbagent_lg.vlm) ───────────────────────────
    # When True, the researcher (and the engineer step evaluator) are given the
    # generated plots under `data/` as image content blocks, so they read trends
    # off the figures instead of inferring them from code + stdout alone. The
    # role's model must be vision-capable (gemini-*, gpt-4o/gpt-5*, claude-*).
    vlm_enabled: bool = False
    vlm_max_images: int = 8
    # Model for the image_reviewer node (vision-capable). None → default model.
    vlm_model: str | None = None
    # Bounded revise-the-plot loop: how many image-review/revision cycles a
    # single step may go through before continuing regardless. Mirrors old
    # cmbagent's `max_vlm_review_attempts`.
    max_vlm_review_attempts: int = 2

    # ── self_debug module ────────────────────────────────────────────────
    # Max engineer attempts per sub-task before giving up. Mirrors
    # cmbagent's `max_n_attempts` (default 3 there too).
    max_n_attempts: int = 3
    engineer_append_instructions: str = ""
    evaluator_append_instructions: str = ""

    # ── researcher module (prose steps inside deep_research) ─────────────
    researcher_append_instructions: str = ""

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
