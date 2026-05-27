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

import shutil
from datetime import datetime
from pathlib import Path
from cmbagent_lg.planning.schemas import Plan


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


def save_deep_research_summary(
    work_dir: str | Path,
    plan: Plan,
    step_outcomes: list,
    step_summaries: list | None = None,
) -> Path:
    """Write a deep_research run summary to `{work_dir}/logs/deep_research_run.json`.

    Captures the plan that was executed plus each step's outcome. The
    one-file answer to "did the plan complete? which step failed? how many
    escalations? what was the plan even?" without grepping across the
    per-step verdict files.
    """
    import json as _json
    out_dir = Path(work_dir).expanduser() / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "deep_research_run.json"
    out.write_text(
        _json.dumps(
            {
                "plan": plan.model_dump(),
                "n_steps": len(plan.sub_tasks),
                "outcomes": step_outcomes,
                "all_fulfilled": bool(
                    step_outcomes
                    and all(o.get("fulfilled") for o in step_outcomes)
                ),
                "step_summaries": step_summaries or [],
            },
            indent=2,
        )
    )
    return out


def prepare_work_dir(work_dir: str | Path, clear: bool = True) -> Path:
    """Resolve, optionally wipe, and (re)create a work_dir. Returns the Path.

    Rerunning with the same explicit work_dir otherwise leaves stale artifacts
    behind — e.g. `codebase/step_2.py` from a previous, longer run, or an old
    `step_1_failure_*.py` — which makes the directory misleading. With
    `clear=True` (default) the directory is emptied first so each run starts
    from a clean slate.
    """
    wd = Path(work_dir).expanduser()
    if clear and wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    return wd
