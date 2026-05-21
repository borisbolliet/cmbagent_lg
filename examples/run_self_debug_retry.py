"""Self-debug example that should require a retry.

The step explicitly instructs the engineer to use `scipy.signal.gaussian`,
which was removed in scipy 1.13 (moved to `scipy.signal.windows.gaussian`).
The host venv has scipy 1.16+, so attempt 1 should fail with ImportError;
attempt 2 should locate the new path and succeed.

    python examples/run_self_debug_retry.py                       # timestamped workdir
    python examples/run_self_debug_retry.py runs/debug_retry_v1   # explicit workdir
"""

import os
import sys
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
    improved_main_task=(
        "Smooth a noisy 1D signal with a Gaussian window from scipy and print "
        "the first few smoothed values."
    ),
    hardware_constraints="Standard laptop. Single CPU. No GPU. 16 GB RAM.",
    code_execution_timeout=30,
    max_n_attempts=3,
)

step = Step(
    sub_task="Smooth a noisy 1D signal using a Gaussian window from scipy.",
    sub_task_agent="engineer",
    bullet_points=[
        "Generate a 1000-sample noisy sine: y = sin(2*pi*5*t/1000) + Gaussian noise (std=0.3, seed=0).",
        "Build the smoothing kernel by importing `gaussian` from `scipy.signal` (i.e. "
        "`from scipy.signal import gaussian; window = gaussian(M=51, std=7)`).",  # ← removed in scipy 1.13
        "Do NOT use `scipy.signal.windows.gaussian` or `scipy.signal.windows.*` — the "
        "downstream verification script does an AST check that the import is exactly "
        "`from scipy.signal import gaussian`. Any other path fails verification.",
        "Normalize the kernel so it sums to 1.",
        "Convolve with the signal using numpy.convolve(mode='same').",
        "Print the first 5 values of the smoothed signal, one per line.",
    ],
    code_execution_timeout=30,
)

# ── workdir + tracing ───────────────────────────────────────────────────

if len(sys.argv) > 1:
    work_dir = Path(sys.argv[1])
elif os.environ.get("WORK_DIR"):
    work_dir = Path(os.environ["WORK_DIR"])
else:
    work_dir = default_work_dir()
# Rerunning with the same work_dir clears it first, so stale step_*/failure
# files from a prior run don't linger. Set KEEP_WORK_DIR=1 to opt out.
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

import json as _json
from pydantic import BaseModel as _BaseModel

_SKIP_KEYS = {
    "attempts", "error_history", "work_dir", "node_elapsed_s",
    "step_number", "data_baseline", "data_manifest",
}
_STEP_NUMBER = 1  # standalone run — deep_research will pass the real plan index

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
                return _json.dumps(_json.loads(s), indent=2)
            except Exception:
                pass
        return v
    try:
        return _json.dumps(v, indent=2, default=str)
    except Exception:
        return str(v)

# ── run ─────────────────────────────────────────────────────────────────

initial_state = {"step": step, "work_dir": str(work_dir), "step_number": _STEP_NUMBER}

_task_snippet = ctx.improved_main_task.strip().split("\n")[0][:60]
_run_name = f"self_debug · {_task_snippet} · {work_dir.name}"
_tags = ["self_debug", "retry_test", work_dir.name]

# Only fields that use the `operator.add` reducer inside the graph need
# accumulation here. `error_history` is managed manually by the evaluator
# node (returns the full list each time), so plain overwrite is correct.
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

err_hist = result.get("error_history", [])
if err_hist:
    print("\n=== ERROR HISTORY ===")
    for i, e in enumerate(err_hist, start=1):
        print(f"  attempt {i}: {e[:200]}{'…' if len(e) > 200 else ''}")

timings = result.get("node_elapsed_s", [])
if timings:
    print("\n=== TIMINGS (wall-clock) ===")
    width = max(len(t["node"]) for t in timings)
    for t in timings:
        print(f"  {t['node']:<{width}}  {t['elapsed_s']:7.2f}s")
    print(f"  {'TOTAL':<{width}}  {sum(t['elapsed_s'] for t in timings):7.2f}s")

import json as _json2
_logs = work_dir / "logs"
_logs.mkdir(parents=True, exist_ok=True)
(_logs / f"step_{_STEP_NUMBER}_timings.json").write_text(
    _json2.dumps(
        {
            "node_elapsed_s": timings,
            "total_node_s": sum(t["elapsed_s"] for t in timings),
        },
        indent=2,
    )
)

print(f"\n[work_dir] code under               {work_dir}/codebase/")
print(f"[work_dir] verdict + timings under  {work_dir}/logs/")

if handler is not None and handler.last_trace_id:
    trace_id = handler.last_trace_id
    tid_path = save_trace_id(trace_id, work_dir)
    print(f"[work_dir] langfuse trace id in    {tid_path}")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    print(f"[trace]    open in UI:             {host}/trace/{trace_id}")
    print(f"[trace]    cost summary:           cmbagent-lg-cost {work_dir}")
else:
    print("[trace]    (no trace id — langfuse handler not attached)")
