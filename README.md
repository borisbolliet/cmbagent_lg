# cmbagent_lg

LangGraph re-implementation of [cmbagent](https://github.com/CMBAgents/cmbagent) deep research.

cmbagent's deep-research workflow, rebuilt in LangGraph as a set of small, composable graphs. A `deep_research` orchestrator generates a plan and runs each step through an engineer self-debug loop or a researcher, with optional multimodal plot review — and checkpoints after every step so a run is resumable.

| Module | What it does | Status |
| --- | --- | --- |
| `planning` | Research task → a structured **plan** (ordered `Step`s, each assigned an agent), via a `planner` ↔ `plan_reviewer` propose-critique loop. | ✅ |
| `self_debug` | One engineer `Step` → **working, executed code**, with bounded retries, an optional **image-reviewer** revise-the-plot loop, and an opt-in escalation hatch. | ✅ |
| `researcher` | One researcher `Step` → a **written markdown report** (`reports/step_N.md`), graded by a verdict. | ✅ |
| `deep_research` | Generate a plan, then run each step through `self_debug` / `researcher`, threading cross-step context and **checkpointing** after every step. | ✅ |

## Key capabilities

- **Per-role, multi-provider models.** Each role (planner, plan_reviewer, engineer, researcher, evaluator, formatter, vlm) takes its own model via `PlanContext`; the **provider is inferred from the name** — `gemini-*` → Google, `gpt-*`/`o[1-4]*` → OpenAI, `claude-*` → Anthropic. Mix providers in one run (matching API key must be set).
- **Multimodal grounding (opt-in, `vlm_enabled`).** The researcher is shown the generated plots, and an **`image_reviewer`** node runs a bounded revise-the-plot loop (`max_vlm_review_attempts`) — a vision model flags concrete figure defects and sends fixes back to the engineer.
- **Crash-recoverable restart.** `deep_research` writes `logs/deep_research_run.json` after every completed step. Re-invoke with `step_index=N` (+ the prior summaries/outcomes) to resume from step N without re-running earlier steps. `restart_at_step` reads this checkpoint.
- **Escalation hatch (opt-in, `enable_escalation`).** A failure the strict loop can't fix (missing package / renamed API) is handed once to a free-form Claude Agent SDK agent that can web-search the fix.
- **Langfuse tracing** of every LLM call, tagged by node, with a `cmbagent-lg-cost` CLI.

## Layout

```
cmbagent_lg/
├── cmbagent_lg/
│   ├── context.py          # PlanContext — run-scoped knobs (models, vlm, timeouts, …)
│   ├── llms.py             # chat_model(model, role): provider-by-name factory (+ proposer/critic/formatter)
│   ├── persistence.py      # work_dir lifecycle; save_final_plan / load_final_plan / save_deep_research_summary
│   ├── prompt_utils.py     # schema-field briefs, content flattening
│   ├── tracing.py          # langfuse callback handler factory
│   ├── timing.py           # @timed_node wall-clock decorator
│   ├── cli.py              # `cmbagent-lg-cost` — per-agent cost/latency from langfuse
│   ├── planning/           # planner ↔ plan_reviewer graph (Plan, Step, Review)
│   ├── self_debug/         # engineer → executor → execution_evaluator → image_reviewer → step_evaluator
│   │   ├── schemas.py      #   EngineerResponse, ExecutionVerdict, StepVerdict, ImageReview
│   │   ├── escalation.py   #   opt-in Claude Agent SDK escape hatch
│   │   └── templates/      #   engineer.yaml, evaluator.yaml, step_evaluator.yaml, escalation.yaml
│   ├── researcher/         # researcher → step_evaluator (writes reports/step_N.md)
│   ├── deep_research/      # orchestrator: run_step → after_step, checkpointing each step
│   └── vlm/                # multimodal grounding (self-contained)
│       ├── images.py       #   collect_images / with_images (provider-agnostic image blocks)
│       └── reviewer.py     #   image_reviewer node + bounded revise-the-plot router
├── examples/
│   ├── run_planner_review.py
│   ├── run_self_debug.py / _retry.py / _lorenz.py / _csv_plot.py
│   ├── run_deep_research.py          # full plan → multi-step execution
│   └── trace_cost_summary.py
├── pyproject.toml
└── .env.example
```

## Install

```bash
python3.12 -m venv ~/pyvenvs/py312-cmbagent-lg
source ~/pyvenvs/py312-cmbagent-lg/bin/activate
pip install -e .
cp .env.example .env  # GOOGLE_API_KEY (+ OPENAI_API_KEY / ANTHROPIC_API_KEY for those providers; LANGFUSE_* for tracing)
```

The engine runs generated code as a subprocess **in this venv**, so install whatever the analyses need (e.g. `pip install numpy scipy matplotlib pandas`). Requires `langchain-core` 1.x and a modern langchain stack.

## Public API

```python
from cmbagent_lg import (
    PlanContext, Plan, Step, Review,
    graph as planning_graph,           # planning
    deep_research_graph,               # orchestrator
    self_debug_graph, researcher_graph,
    save_final_plan, load_final_plan, save_deep_research_summary,
    EngineerResponse, ExecutionVerdict, StepVerdict, ImageReview,
)
```

## End-to-end

```python
ctx = PlanContext(
    main_task="Investigate how a small MLP learns a smooth 5-D function. "
              "Generate data, train, and produce diagnostic plots.",
    maximum_number_of_steps_in_plan=3,
    available_agents=[("engineer", "writes+runs code"), ("researcher", "writes prose")],
    # per-role, multi-provider (provider inferred from the name)
    engineer_model="gpt-5.4", researcher_model="claude-sonnet-4-6",
    planner_model="gemini-3.5-flash", evaluator_model="gemini-3.1-flash-lite",
    # multimodal grounding + plot-review loop
    vlm_enabled=True, vlm_model="gemini-3.1-flash-lite", max_vlm_review_attempts=2,
)

plan = planning_graph.invoke({}, context=ctx)["current_plan"]
save_final_plan(plan, "runs/mlp")

res = deep_research_graph.invoke({"plan": plan, "work_dir": "runs/mlp"}, context=ctx)
outcomes = res["step_outcomes"]      # [{step_number, fulfilled, attempts, escalated, reason?}, …]
```

### Resume after a crash

```python
import json
ck = json.load(open("runs/mlp/logs/deep_research_run.json"))   # written after each step
resume_at = max(o["step_number"] for o in ck["outcomes"] if o["fulfilled"]) + 1
deep_research_graph.invoke(
    {"plan": plan, "work_dir": "runs/mlp", "step_index": resume_at,
     "step_summaries": ck["step_summaries"][:resume_at-1],
     "step_outcomes":  ck["outcomes"][:resume_at-1]},
    context=ctx,
)
```

## The graphs

### `planning`

```
START → planner → format_plan → [END if last round else continue]
                                 │
                                 ▼
                             plan_reviewer → format_review → planner …
```

`num_rounds` counts **review cycles**; total planner passes = `num_rounds + 1`. Generators write free-form prose; tiny formatter nodes convert it to typed Pydantic via `with_structured_output`. The formatter dynamically constrains `Step.sub_task_agent` to `available_agents`.

### `self_debug`

```
engineer → format_engineer → executor → execution_evaluator
    ▲                                          │ success
    │                                          ▼
    │                          image_reviewer (if vlm_enabled)
    │   figure needs revision ◄──────────────  │  (else → step_evaluator)
    │   (+budget) → engineer                    ▼
    │                                      step_evaluator
    │   code FAILURE → engineer ◄─────────┐         │ goal MET → END
    │   escalatable → escalation → executor│        │ goal MISS → engineer
    └──────────────────────────────────────┘
```

Two LLM gates over one shared retry budget (`max_n_attempts`): **`execution_evaluator`** (did it *run*? → `ExecutionVerdict`) and **`step_evaluator`** (did it *meet the goal*? → `StepVerdict`). When `vlm_enabled`, **`image_reviewer`** runs between them and can bounce a fixable figure back to the engineer (bounded by `max_vlm_review_attempts`). The `executor` writes `codebase/step_N.py` and runs it with `cwd = work_dir`, so the script's `data/<file>` outputs land in `work_dir/data/`; the data baseline is re-snapshotted each attempt so a step's manifest reflects only its successful attempt.

```
work_dir/
├── codebase/   step_N.py, step_N.log, step_N_failure_I.py (audit trail)
├── data/       output files the script produced (flat — only these are tracked)
├── reports/    step_N.md  (researcher steps)
└── logs/       step_N_execution_verdict.json, step_N_verdict.json,
                step_N_data_manifest.json, step_N_image_review_*.json,
                deep_research_run.json  (the per-step CHECKPOINT)
```

### `researcher`

```
researcher → step_evaluator
    ▲              │ goal MET → END
    └──────────────┘ goal MISS → researcher   (until attempts == max_n_attempts)
```

A researcher step writes **prose, not code** (uses the `researcher_model`). The node makes one LLM call, saves the raw markdown to `reports/step_N.md`, and a `step_evaluator` grades it (`StepVerdict`); on a goal-miss the unmet requirements + feedback feed back into the next attempt (bounded by `max_n_attempts`, failures demoted to `reports/step_N_failure_*.md`). The highest-numbered non-failure `reports/step_N.md` is the final report — Denario takes it as `results.md`.

**What's in the researcher's context** (its system prompt is assembled from):

- `main_task` — the overall research task.
- `researcher_append_instructions` — task-specific guidance (e.g. Denario's "write the full Results section, ~2000 words, academic, interpret the plots/tables").
- `previous_steps_execution_summary` — the cross-step context: each prior step's **executed code + stdout**, plus a **workspace file manifest** (the relative paths under `codebase/`, `data/`, `reports/`). This is how the researcher learns what the analysis produced.
- `current_sub_task` + its bullet-point `current_instructions`.
- `retry_context` — on a retry, the previous report and the evaluator's unmet requirements / feedback.
- **the generated plots** — when `vlm_enabled`, the figures under `data/` are attached to the prompt as images, so the report is grounded in what the plots actually show rather than inferred.

**Grounding caveat (important).** The researcher sees prior steps' **stdout**, *not* the contents of saved data files — so the engineer must **print** its key results/tables (the engineer prompt mandates this) for them to reach the report. With `vlm_enabled` the researcher also sees the plots. If the numbers aren't printed and the plots aren't shown, the researcher writes from the code alone and can state conclusions the data doesn't support.

## Run the examples

```bash
python examples/run_planner_review.py        # planning
python examples/run_self_debug_csv_plot.py   # self_debug (CSV + plot)
python examples/run_deep_research.py runs/dr # full plan → multi-step execution
```

Examples accept an optional explicit work_dir (cleared on rerun; `KEEP_WORK_DIR=1` to keep).

## Langfuse (self-hosted, optional)

```bash
cd ~/GitHub/langfuse && docker compose up -d   # http://localhost:3000 → create project → keys into .env
```

`tracing.py` builds a LangChain `CallbackHandler` from the `LANGFUSE_*` env vars; pass it via `config={"callbacks": [handler]}` and every LLM call (including formatter sub-calls) is traced, tagged by node name. Then:

```bash
cmbagent-lg-cost <trace_id | work_dir>   # per-agent cost / tokens / latency
```

## Used by Denario

cmbagent_lg is the analysis/results engine behind [Denario](https://github.com/AstroPilot-AI/Denario): `denario_results` maps `params.yaml`'s `Analysis module` (per-role models, `max_n_steps`, the VLM block) onto a `PlanContext` and runs `planning_graph` + `deep_research_graph`.
