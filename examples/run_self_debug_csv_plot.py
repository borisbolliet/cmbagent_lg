"""Self-debug example that produces both a CSV and a plot.

The engineer generates a damped-oscillation dataset, saves it as CSV, and
plots it — two output files under `{work_dir}/data/`. The executor records
which files the step produced in `{work_dir}/logs/step_1_data_manifest.json`
(provenance tracked externally — file names and bytes are untouched, and the
engineer is never asked to embed a step id).

    python examples/run_self_debug_csv_plot.py runs/csv_plot
"""

import json
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

from cmbagent_lg import PlanContext, Step, self_debug_graph
from _common import (
    attach_langfuse,
    print_self_debug_verdicts,
    print_timings,
    print_trace_info,
    resolve_work_dir,
    save_node_timings,
    stream_and_render,
)

# ── inputs ──────────────────────────────────────────────────────────────

ctx = PlanContext(
    main_task="Generate a damped-oscillation dataset, save it, and plot it.",
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

STEP_NUMBER = 1

# ── run ─────────────────────────────────────────────────────────────────

work_dir = resolve_work_dir(sys.argv)
handler, callbacks = attach_langfuse()

task_snippet = ctx.main_task.strip().split("\n")[0][:60]
run_name = f"self_debug · {task_snippet} · {work_dir.name}"
tags = ["self_debug", "csv_plot", work_dir.name]

result = stream_and_render(
    self_debug_graph,
    {"step": step, "work_dir": str(work_dir), "step_number": STEP_NUMBER},
    context=ctx,
    config={
        "callbacks": callbacks,
        "run_name": run_name,
        "tags": tags,
        "metadata": {"langfuse_session_id": work_dir.name, "langfuse_tags": tags},
    },
)

# ── summary ─────────────────────────────────────────────────────────────

print_self_debug_verdicts(result, ctx)
timings = result.get("node_elapsed_s", [])
print_timings(timings)
save_node_timings(work_dir, STEP_NUMBER, timings)

# Show the data manifest the executor wrote — step → file provenance.
manifest_path = work_dir / "logs" / f"step_{STEP_NUMBER}_data_manifest.json"
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text())
    print(f"\n=== DATA MANIFEST (step {manifest['step_number']}) ===")
    for f in manifest["files"]:
        print(f"  {f['path']:<32s} {f['bytes']:>9d} B   {f['modified']}")
    if not manifest["files"]:
        print("  (no files produced)")

print(f"\n[work_dir] code under               {work_dir}/codebase/")
print(f"[work_dir] data files under         {work_dir}/data/")
print(f"[work_dir] manifest + verdict under {work_dir}/logs/")
print_trace_info(handler, work_dir)
