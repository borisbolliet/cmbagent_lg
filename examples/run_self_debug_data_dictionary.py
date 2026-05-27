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
from pathlib import Path
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

# ── input dataset ───────────────────────────────────────────────────────

DATASET_DIR = Path.home() / "GitHub/cmbagent/tests/test_dataset"
DATASET = DATASET_DIR / os.environ.get("DATASET", "synthetic_drug_dev_portfolio.csv")
assert DATASET.is_file(), (
    f"dataset not found at {DATASET} — set DATASET=<filename> or adjust the path."
)

# ── inputs ──────────────────────────────────────────────────────────────

ctx = PlanContext(
    main_task="Produce a data dictionary for a tabular dataset.",
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

STEP_NUMBER = 1

# ── run ─────────────────────────────────────────────────────────────────

work_dir = resolve_work_dir(sys.argv)
handler, callbacks = attach_langfuse()

task_snippet = ctx.main_task.strip().split("\n")[0][:60]
run_name = f"self_debug · {task_snippet} · {work_dir.name}"
tags = ["self_debug", "data_dictionary", DATASET.stem, work_dir.name]

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

for f in result.get("data_manifest", []):
    print(f"\n[data] produced {f['path']} ({f['bytes']} bytes)")

print(f"\n[work_dir] code under               {work_dir}/codebase/")
print(f"[work_dir] data files under         {work_dir}/data/")
print(f"[work_dir] manifest + verdict under {work_dir}/logs/")
print_trace_info(handler, work_dir)
