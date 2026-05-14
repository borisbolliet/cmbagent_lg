"""Graph state — the mutable bag that flows between nodes.

Flow:

    planner ─► format_plan ─► [route: END if round==num_rounds else continue]
                              │
                              ▼
                          plan_reviewer ─► format_review ─► planner ...

`raw_*` are the generators' free-form prose; `current_*` are the typed
formatter outputs. `history` accumulates one `(plan, review)` pair per
completed review cycle, so each planner pass sees every prior round.
"""

from typing import TypedDict, List, Tuple, Optional
from cmbagent_lg.schemas import Plan, Review


class PlanState(TypedDict, total=False):
    raw_plan: str
    current_plan: Plan
    raw_review: str
    current_review: Optional[Review]
    history: List[Tuple[Plan, Review]]
    round: int
