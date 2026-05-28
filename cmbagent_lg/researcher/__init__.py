"""Researcher sub-graph: prose steps inside deep_research plans.

A tiny parallel to `self_debug/` for `sub_task_agent == "researcher"` steps.
The researcher writes a markdown report; a single `step_evaluator` gate
decides whether the report addressed the sub-task. Bounded retries via the
same `ctx.max_n_attempts` the engineer uses.

No code execution, no execution_evaluator (no code to fail), no escalation.
"""

from cmbagent_lg.researcher.graph import graph
from cmbagent_lg.researcher.state import ResearcherState

__all__ = ["graph", "ResearcherState"]
