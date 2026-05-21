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
    prepare_work_dir,
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

# Stream node-by-node so we can follow the flow. For each node's state
# delta, render only the substantive payload: Pydantic objects via .format()
# if available (or pretty JSON), and raw text as-is. Skip bookkeeping fields.
import json as _json
from pydantic import BaseModel as _BaseModel

_SKIP_KEYS = {"round", "history", "node_elapsed_s"}

def _render_value(v):
    if isinstance(v, _BaseModel):
        fmt = getattr(v, "format", None)
        if callable(fmt):
            return fmt()
        return v.model_dump_json(indent=2)
    if isinstance(v, str):
        s = v.strip()
        # Try to pretty-print JSON if the string is JSON.
        if s and s[0] in "{[":
            try:
                return _json.dumps(_json.loads(s), indent=2)
            except Exception:
                pass
        return v
    try:
        return _json.dumps(v, indent=2, default=str)
    except Exception:
        return str(v)

# Fields the graph state reduces with `operator.add` — mirror that here so
# the local `result` dict stays consistent with the in-graph state.
_LIST_REDUCED_KEYS = {"node_elapsed_s"}

# Trace identity for langfuse — see run_self_debug.py for rationale. Workdir
# resolution happens later in this script, so derive a basename from whichever
# source is set; falls back to "default" if none.
import os as _os
_work_dir_basename = (
    Path(sys.argv[1]).name if len(sys.argv) > 1
    else (Path(_os.environ["WORK_DIR"]).name if _os.environ.get("WORK_DIR") else "default")
)
_task_snippet = ctx.improved_main_task.strip().split("\n")[0][:60]
_run_name = f"planning · {_task_snippet} · {_work_dir_basename}"
_tags = ["planning", _work_dir_basename]

result = {}
for chunk in graph.stream(
    {},
    context=ctx,
    config={
        "callbacks": callbacks,
        "run_name": _run_name,
        "tags": _tags,
        "metadata": {
            "langfuse_session_id": _work_dir_basename,
            "langfuse_tags": _tags,
        },
    },
    stream_mode="updates",
):
    # chunk is {node_name: state_delta}
    for node_name, delta in chunk.items():
        print(f"\n\n══════════ {node_name} ══════════")
        for k, v in (delta or {}).items():
            if k in _SKIP_KEYS:
                continue
            print(f"\n── {k} ──")
            print(_render_value(v))
        for k, v in (delta or {}).items():
            if k in _LIST_REDUCED_KEYS:
                result[k] = list(result.get(k, [])) + list(v or [])
            else:
                result[k] = v

print("\n\n=== FINAL PLAN ===")
print(result["current_plan"].format())
print(f"\n=== HISTORY: {len(result.get('history', []))} reviewed round(s) before the final plan ===")

timings = result.get("node_elapsed_s", [])
if timings:
    print("\n=== TIMINGS (wall-clock) ===")
    width = max(len(t["node"]) for t in timings)
    for t in timings:
        print(f"  {t['node']:<{width}}  {t['elapsed_s']:7.2f}s")
    print(f"  {'TOTAL':<{width}}  {sum(t['elapsed_s'] for t in timings):7.2f}s")

# Workdir resolution: CLI arg > $WORK_DIR > timestamped default.
if len(sys.argv) > 1:
    work_dir = Path(sys.argv[1])
elif os.environ.get("WORK_DIR"):
    work_dir = Path(os.environ["WORK_DIR"])
else:
    work_dir = default_work_dir()
# Clear any prior run at this path (KEEP_WORK_DIR=1 to opt out).
_clear = not os.environ.get("KEEP_WORK_DIR")
if _clear and work_dir.exists():
    print(f"[work_dir] clearing existing      {work_dir}")
work_dir = prepare_work_dir(work_dir, clear=_clear)

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
