"""Self-debug example that produces both a CSV and a plot.

The engineer generates a damped-oscillation dataset, saves it as CSV, and
plots it — two output files under `{work_dir}/data/`. The executor records
which files the step produced in `{work_dir}/logs/step_1_data_manifest.json`
(provenance tracked externally — file names and bytes are untouched, and the
engineer is never asked to embed a step id).

    python examples/run_self_debug_csv_plot.py                   # timestamped workdir
    python examples/run_self_debug_csv_plot.py runs/csv_plot     # explicit workdir
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

from cmbagent_lg import (
    PlanContext,
    Step,
    self_debug_graph,
    save_trace_id,
    default_work_dir,
    prepare_work_dir,
)

# ── inputs ──────────────────────────────────────────────────────────────

ctx = PlanContext(
    improved_main_task="Generate a damped-oscillation dataset, save it, and plot it.",
    hardware_constraints="Standard laptop. Single CPU. No GPU. 16 GB RAM.",
    code_execution_timeout=60,
    max_n_attempts=3,
)

step = Step(
    sub_task="Generate a damped oscillation, save it as CSV, and plot it.",
    sub_task_agent="engineer",
    bullet_points=[
        "Compute a damped sine wave y = exp(-0.15*t) * sin(2*pi*t) on 400 "
        "evenly spaced points t in [0, 20].",
        "Save the dataset as CSV at data/damped_oscillation.csv with a header "
        "row 't,y'.",
        "Plot y versus t with matplotlib and save the figure to "
        "data/damped_oscillation.png at dpi>=300.",
        "Print the path of each file you save.",
    ],
    code_execution_timeout=60,
)

_STEP_NUMBER = 1

# ── workdir + tracing ───────────────────────────────────────────────────

if len(sys.argv) > 1:
    work_dir = Path(sys.argv[1])
elif os.environ.get("WORK_DIR"):
    work_dir = Path(os.environ["WORK_DIR"])
else:
    work_dir = default_work_dir()
_clear = not os.environ.get("KEEP_WORK_DIR")
if _clear and work_dir.exists():
    print(f"[work_dir] clearing existing       {work_dir}")
work_dir = prepare_work_dir(work_dir, clear=_clear)

handler = None
callbacks = []
try:
    from cmbagent_lg.tracing import langfuse_handler

    handler = langfuse_handler()
    callbacks.append(handler)
    print("[trace] langfuse handler attached")
except Exception as e:
    print(f"[trace] skipping langfuse: {e}")

# ── stream renderer ─────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel

_SKIP_KEYS = {
    "attempts", "error_history", "work_dir", "node_elapsed_s",
    "step_number", "data_baseline", "data_manifest", "step_feedback_history",
}

def _render_value(v):
    if isinstance(v, _BaseModel):
        fmt = getattr(v, "format", None)
        if callable(fmt):
            return fmt()
        return v.model_dump_json(indent=2)
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in "{[":
            try:
                return json.dumps(json.loads(s), indent=2)
            except Exception:
                pass
        return v
    try:
        return json.dumps(v, indent=2, default=str)
    except Exception:
        return str(v)

# ── run ─────────────────────────────────────────────────────────────────

initial_state = {"step": step, "work_dir": str(work_dir), "step_number": _STEP_NUMBER}

_task_snippet = ctx.improved_main_task.strip().split("\n")[0][:60]
_run_name = f"self_debug · {_task_snippet} · {work_dir.name}"
_tags = ["self_debug", "csv_plot", work_dir.name]

_LIST_REDUCED_KEYS = {"node_elapsed_s"}

result = dict(initial_state)
for chunk in self_debug_graph.stream(
    initial_state,
    context=ctx,
    config={
        "callbacks": callbacks,
        "run_name": _run_name,
        "tags": _tags,
        "metadata": {
            "langfuse_session_id": work_dir.name,
            "langfuse_tags": _tags,
        },
    },
    stream_mode="updates",
):
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

# ── summary ─────────────────────────────────────────────────────────────

exec_verdict = result.get("current_execution_verdict")
step_verdict = result.get("current_step_verdict")
print("\n\n=== EXECUTION VERDICT (did the code run cleanly?) ===")
print(exec_verdict.format() if exec_verdict else "(none — exited before execution_evaluator)")
print("\n=== STEP VERDICT (did it achieve the goal?) ===")
print(step_verdict.format() if step_verdict else "(none — the code never ran cleanly)")
print(f"\n=== attempts: {result.get('attempts', 0)} / {ctx.max_n_attempts} ===")

timings = result.get("node_elapsed_s", [])
if timings:
    print("\n=== TIMINGS (wall-clock) ===")
    width = max(len(t["node"]) for t in timings)
    for t in timings:
        print(f"  {t['node']:<{width}}  {t['elapsed_s']:7.2f}s")
    print(f"  {'TOTAL':<{width}}  {sum(t['elapsed_s'] for t in timings):7.2f}s")

_logs = work_dir / "logs"
_logs.mkdir(parents=True, exist_ok=True)
(_logs / f"step_{_STEP_NUMBER}_timings.json").write_text(
    json.dumps(
        {
            "node_elapsed_s": timings,
            "total_node_s": sum(t["elapsed_s"] for t in timings),
        },
        indent=2,
    )
)

# Show the data manifest the executor wrote — step → file provenance.
_manifest_path = _logs / f"step_{_STEP_NUMBER}_data_manifest.json"
if _manifest_path.is_file():
    manifest = json.loads(_manifest_path.read_text())
    print(f"\n=== DATA MANIFEST (step {manifest['step_number']}) ===")
    for f in manifest["files"]:
        print(f"  {f['path']:<32s} {f['bytes']:>9d} B   {f['modified']}")
    if not manifest["files"]:
        print("  (no files produced)")

print(f"\n[work_dir] code under               {work_dir}/codebase/")
print(f"[work_dir] data files under         {work_dir}/data/")
print(f"[work_dir] manifest + verdict under {work_dir}/logs/")

if handler is not None and handler.last_trace_id:
    trace_id = handler.last_trace_id
    tid_path = save_trace_id(trace_id, work_dir)
    print(f"[work_dir] langfuse trace id in    {tid_path}")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    print(f"[trace]    open in UI:             {host}/trace/{trace_id}")
    print(f"[trace]    cost summary:           cmbagent-lg-cost {work_dir}")
else:
    print("[trace]    (no trace id — langfuse handler not attached)")
