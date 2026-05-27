"""End-to-end run: planning → deep_research → multi-step self_debug.

Demonstrates the *cross-step* carryover that's the whole point of
`deep_research`:

  1. The planner produces a multi-step Plan (engineer steps only — we force
     this via `available_agents`).
  2. `deep_research_graph` iterates the steps: each step is one fresh
     `self_debug_graph` invocation, threading a `previous_steps_execution_summary`
     (prior steps' code + stdout + a workspace file manifest) into the
     engineer's prompt.
  3. The demo task ("learn a 5D nonlinear function with a small MLP") forces
     a real cross-step dependency: step 1 generates the synthetic dataset
     and trains the model; step 2 loads what step 1 produced and renders
     diagnostic plots. The 4-sentence main_task is intentionally vague — the
     planner decides the decomposition.

    python examples/run_deep_research.py runs/deep_research
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

from cmbagent_lg import (
    PlanContext,
    graph as planning_graph,
    deep_research_graph,
    save_final_plan,
    save_deep_research_summary,
)
from _common import (
    attach_langfuse,
    print_timings,
    print_trace_info,
    resolve_work_dir,
)

# ── inputs ──────────────────────────────────────────────────────────────

ctx = PlanContext(
    main_task=(
        "Investigate how well a small MLP can learn a smooth nonlinear "
        "function of five random inputs. Generate a synthetic 5D regression "
        "dataset, train an MLP on it, and produce diagnostic plots of the "
        "result."
    ),
    hardware_constraints="Standard laptop. Single CPU. No GPU. 16 GB RAM.",
    code_execution_timeout=120,
    max_n_attempts=3,
    # Engineer-only — deep_research v1 only handles engineer steps.
    available_agents=[
        ("engineer", "Writes and runs Python code: numeric work, data I/O, plotting, ML."),
    ],
    maximum_number_of_steps_in_plan=3,
    num_rounds=1,
    # Escalation on — sklearn isn't installed in this venv, so the MLP step
    # will hit ModuleNotFoundError; escalation pip-installs it and re-runs.
    # Demonstrates deep_research + self_debug + escalation in one demo.
    enable_escalation=True,
    escalation_model="claude-haiku-4-5-20251001",
    escalation_max_budget_usd=0.50,
    escalation_max_turns=12,
)

# ── workdir + tracing ───────────────────────────────────────────────────

work_dir = resolve_work_dir(sys.argv)
handler, callbacks = attach_langfuse()

task_snippet = ctx.main_task.strip().split("\n")[0][:60]
run_name = f"deep_research · {task_snippet} · {work_dir.name}"
tags = ["deep_research", work_dir.name]
common_config = {
    "callbacks": callbacks,
    "run_name": run_name,
    "tags": tags,
    "metadata": {"langfuse_session_id": work_dir.name, "langfuse_tags": tags},
}

# ── 1. plan ─────────────────────────────────────────────────────────────

print("\n══════════ PLANNING ══════════")
plan_result = planning_graph.invoke({}, context=ctx, config=common_config)
plan = plan_result["current_plan"]
save_final_plan(plan, work_dir)
print(plan.format())

# ── 2. execute ──────────────────────────────────────────────────────────

print("\n══════════ DEEP_RESEARCH ══════════")
dr_result = deep_research_graph.invoke(
    {"plan": plan, "work_dir": str(work_dir)},
    context=ctx,
    config=common_config,
)

# ── summary ─────────────────────────────────────────────────────────────

outcomes = dr_result.get("step_outcomes", [])
all_ok = bool(outcomes and all(o.get("fulfilled") for o in outcomes))

print("\n\n=== STEP OUTCOMES ===")
for o in outcomes:
    flag = "✓" if o.get("fulfilled") else "✗"
    extras = []
    if "attempts" in o:
        extras.append(f"attempts={o['attempts']}")
    if o.get("escalated"):
        extras.append("escalated")
    if not o.get("fulfilled") and o.get("reason"):
        extras.append(f"reason={o['reason']!r}")
    print(f"  {flag} step {o['step_number']}  " + " ".join(extras))

print(
    f"\n=== PLAN: {'COMPLETE' if all_ok else 'HALTED'} "
    f"({len(outcomes)}/{len(plan.sub_tasks)} steps executed) ==="
)

print_timings(dr_result.get("node_elapsed_s", []))

save_deep_research_summary(
    work_dir, plan, outcomes, dr_result.get("step_summaries", [])
)

# Show what landed on disk.
print("\n=== WORK_DIR TREE ===")
for root, _, files in sorted(os.walk(work_dir)):
    rel = Path(root).relative_to(work_dir)
    for f in sorted(files):
        print(f"  {rel / f if str(rel) != '.' else f}")

print_trace_info(handler, work_dir)
