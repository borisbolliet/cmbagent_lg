"""Self-debug example: build a data dictionary for a real tabular dataset.

The engineer reads an existing CSV from cmbagent's test dataset and produces
a *data dictionary*: one row per column describing its dtype, missingness,
cardinality, and summary stats. This exercises the *input file* path (the
engineer reads a user-provided absolute path) alongside the usual `data/`
output convention.

The dataset is selectable with the `DATASET` env var (default:
`synthetic_drug_dev_portfolio.csv`). `Stocks.csv` is a good harder case — it
has a `# Data source:` comment line before the header, which a naive
`pd.read_csv` mishandles.

    python examples/run_self_debug_data_dictionary.py runs/datadict
    DATASET=Stocks.csv python examples/run_self_debug_data_dictionary.py runs/datadict_stocks
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

# ── input dataset ───────────────────────────────────────────────────────

DATASET_DIR = Path.home() / "GitHub/cmbagent/tests/test_dataset"
DATASET = DATASET_DIR / os.environ.get("DATASET", "synthetic_drug_dev_portfolio.csv")
assert DATASET.is_file(), (
    f"dataset not found at {DATASET} — set DATASET=<filename> or adjust the path."
)

# ── inputs ──────────────────────────────────────────────────────────────

ctx = PlanContext(
    improved_main_task="Produce a data dictionary for a tabular dataset.",
    hardware_constraints="Standard laptop. Single CPU. No GPU. 16 GB RAM.",
    code_execution_timeout=60,
    max_n_attempts=3,
)

step = Step(
    sub_task="Build a data dictionary for the input CSV dataset.",
    sub_task_agent="engineer",
    bullet_points=[
        f"Load the dataset from this exact absolute path: {DATASET}",
        "For every column, determine: the inferred dtype, the count of non-null "
        "values, the fraction of missing values, and the number of unique values.",
        "For numeric columns also record min, max, mean and std; for non-numeric "
        "columns record up to 5 example category values.",
        "Save the data dictionary as data/data_dictionary.csv — one row per "
        "column of the input dataset.",
        "Print the data dictionary as a readable table to stdout.",
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
_tags = ["self_debug", "data_dictionary", DATASET.stem, work_dir.name]

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

for f in result.get("data_manifest", []):
    print(f"\n[data] produced {f['path']} ({f['bytes']} bytes)")

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
