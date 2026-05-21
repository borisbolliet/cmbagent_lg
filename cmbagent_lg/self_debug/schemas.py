"""Pydantic schemas for the self_debug loop.

Two distinct verdicts, mirroring cmbagent's split:
- `ExecutionVerdict` — did the code RUN cleanly? (cmbagent's executor_response_evaluator)
- `StepVerdict` — did the run ACHIEVE the sub-task's goal? (cmbagent's controller)
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EngineerResponse(BaseModel):
    """Structured output of the engineer + format_engineer pair."""

    python_code: str = Field(
        description="Single self-contained Python script, ready to be executed by "
        "`python <file.py>` in the host venv. No CLI arguments, wrapped in "
        "`if __name__ == '__main__':`."
    )
    code_explanation: str = Field(
        description="One concise paragraph explaining what the script does."
    )
    modification_summary: Optional[str] = Field(
        default=None,
        description="If this is a retry, summarize what changed vs the previous "
        "attempt and why. Omit on the first attempt.",
    )

    def format(self) -> str:
        out = "**Code Explanation:**\n" + self.code_explanation + "\n"
        if self.modification_summary:
            out += "\n**Modifications:**\n" + self.modification_summary + "\n"
        out += "\n**Python Code:**\n```python\n" + self.python_code + "\n```\n"
        return out


class ExecutionVerdict(BaseModel):
    """Structured output of the execution_evaluator node — judges only whether
    the last subprocess RAN cleanly, not whether the step goal was met."""

    status: Literal["success", "failure"] = Field(
        description="`success` if the script ran cleanly (returncode 0, no traceback, "
        "no timeout); `failure` otherwise."
    )
    error_summary: Optional[str] = Field(
        default=None,
        description="On failure: one-paragraph description of what went wrong, with the "
        "key traceback line if useful. Null on success.",
    )
    fix_suggestion: Optional[str] = Field(
        default=None,
        description="On failure: concrete, actionable guidance for the engineer's "
        "next attempt. Null on success.",
    )

    def format(self) -> str:
        out = f"**Status:** {self.status}\n"
        if self.error_summary:
            out += f"\n**Error summary:**\n{self.error_summary}\n"
        if self.fix_suggestion:
            out += f"\n**Fix suggestion:**\n{self.fix_suggestion}\n"
        return out


class StepVerdict(BaseModel):
    """Structured output of the step_evaluator node — judges whether the run
    ACHIEVED the sub-task's goal (the code is already known to have run
    cleanly). Mirrors cmbagent's controller verdict."""

    fulfilled: bool = Field(
        description="True only if every part of the sub-task AND every "
        "bullet-point requirement is satisfied by this run."
    )
    unmet_requirements: List[str] = Field(
        default_factory=list,
        description="When not fulfilled: the specific sub-task parts / bullet "
        "points that were not satisfied. Empty when fulfilled.",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="When not fulfilled: concrete, actionable guidance for the "
        "engineer's next attempt. Null when fulfilled.",
    )

    def format(self) -> str:
        out = f"**Step fulfilled:** {self.fulfilled}\n"
        if self.unmet_requirements:
            out += "\n**Unmet requirements:**\n"
            for r in self.unmet_requirements:
                out += f"- {r}\n"
        if self.feedback:
            out += f"\n**Feedback:**\n{self.feedback}\n"
        return out
