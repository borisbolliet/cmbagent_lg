"""Tiny end-to-end run of the planner ↔ plan_reviewer loop.

    python examples/run_planner_review.py                       # timestamped workdir
    python examples/run_planner_review.py runs/cmb_lensing_v1   # explicit workdir
    WORK_DIR=runs/cmb_lensing_v1 python examples/run_planner_review.py

Requires GOOGLE_API_KEY in .env. Langfuse tracing is attached if
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set; otherwise the run proceeds
without tracing.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# override=True so values in .env beat any stale shell exports (LANGFUSE_*
# from .zshrc, etc.).
load_dotenv(override=True)

from cmbagent_lg import (
    graph,
    PlanContext,
    save_final_plan,
    save_trace_id,
    default_work_dir,
)

ctx = PlanContext(
    improved_main_task=(
        "Reconstruct the CMB lensing potential from a single Planck-like temperature "
        "map. We have only the map; no external catalogs."
    ),
    hardware_constraints="Standard laptop. Single CPU. No GPU. 16 GB RAM.",
    code_execution_timeout=120,
    maximum_number_of_steps_in_plan=5,
    num_rounds=2,  # 2 review cycles → plan v1 → review → v2 → review → v3 (final)
    # Agents the planner is allowed to assign work to. The dynamic Literal
    # built from this list constrains `Step.sub_task_agent` at format time,
    # AND the inventory is rendered into the planner/reviewer system prompts.
    # Add agents here as we port them from cmbagent (plotter, camb_context, …).
    available_agents=[
        ("engineer", "Writes and runs Python code: numeric work, data I/O, computations, plotting."),
        ("researcher", "Reads, reasons, writes prose; no code execution."),
    ],
)

handler = None
callbacks = []
try:
    from cmbagent_lg.tracing import langfuse_handler

    handler = langfuse_handler()
    callbacks.append(handler)
    print("[trace] langfuse handler attached")
except Exception as e:
    print(f"[trace] skipping langfuse: {e}")

result = graph.invoke({}, context=ctx, config={"callbacks": callbacks})

print("\n=== FINAL PLAN ===")
print(result["current_plan"].format())
print(f"\n=== HISTORY: {len(result.get('history', []))} reviewed round(s) before the final plan ===")

# Workdir resolution: CLI arg > $WORK_DIR > timestamped default.
if len(sys.argv) > 1:
    work_dir = Path(sys.argv[1])
elif os.environ.get("WORK_DIR"):
    work_dir = Path(os.environ["WORK_DIR"])
else:
    work_dir = default_work_dir()

plan_path = save_final_plan(result["current_plan"], work_dir)
print(f"\n[work_dir] final plan saved to    {plan_path}")

if handler is not None and handler.last_trace_id:
    trace_id = handler.last_trace_id
    tid_path = save_trace_id(trace_id, work_dir)
    print(f"[work_dir] langfuse trace id in    {tid_path}")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    print(f"[trace]    open in UI:             {host}/trace/{trace_id}")
    print(f"[trace]    cost summary:           cmbagent-lg-cost {work_dir}")
else:
    print("[trace]    (no trace id — langfuse handler not attached)")
