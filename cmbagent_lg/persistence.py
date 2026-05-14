"""Local persistence for the planning module.

Only the **final plan** is written to disk — telemetry (cost, usage, latency,
full prompts) lives in langfuse. The plan is special because it's the input
to a downstream control phase (cmbagent today, the langgraph port later),
so it needs to live somewhere stable and JSON-serializable.

Shape matches `cmbagent/agents/planning/planner_response_evaluator.save_final_plan()`:

    {
      "sub_tasks": [
        {
          "sub_task": "...",
          "sub_task_agent": "...",
          "bullet_points": [...],
          "code_execution_timeout": null
        },
        ...
      ]
    }
"""

from datetime import datetime
from pathlib import Path
from cmbagent_lg.schemas import Plan


def save_final_plan(plan: Plan, work_dir: str | Path) -> Path:
    """Write the final plan to `{work_dir}/planning/final_plan.json`. Returns the path."""
    out = Path(work_dir).expanduser() / "planning" / "final_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plan.model_dump_json(indent=2))
    return out


def save_trace_id(trace_id: str, work_dir: str | Path) -> Path:
    """Write the langfuse trace ID to `{work_dir}/langfuse_trace_id.txt`.

    Lets you later run `examples/trace_cost_summary.py <id>` to pull cost +
    usage for this specific run, or open the trace in the UI at
    `{LANGFUSE_HOST}/project/<pid>/traces/<id>`.
    """
    out = Path(work_dir).expanduser() / "langfuse_trace_id.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(trace_id + "\n")
    return out


def default_work_dir() -> Path:
    """`./work_dir/{YYYY-mm-dd_HH-MM-SS}/` — cmbagent-style timestamped run dir."""
    return Path("work_dir") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
