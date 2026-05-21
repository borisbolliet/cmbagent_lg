# cmbagent_lg

LangGraph re-implementation of [cmbagent](https://github.com/CMBAgents/cmbagent) deep research.

cmbagent's deep-research workflow is being ported to LangGraph one **capability module** at a time. Each module is a self-contained, separately runnable graph; eventually a `deep_research` module will orchestrate them.

| Module | What it does | Status |
| --- | --- | --- |
| `planning` | Research task → a structured **plan** (ordered `Step`s, each assigned an agent), via a `planner` ↔ `plan_reviewer` propose-critique loop. | ✅ |
| `self_debug` | One `Step` → **working, executed code** for it, with bounded retries on failure. | ✅ |
| `deep_research` | Make a plan, then run each step through `self_debug`. | planned |

The propose → format → critique → format pattern is the architectural blueprint from `agents_lab_2026/Lab3.ipynb` (`idea_maker` / `idea_hater`). Lab 4 covers `planning` + Langfuse tracing; Lab 5 covers `self_debug`.

## Layout

```
cmbagent_lg/
├── cmbagent_lg/
│   ├── context.py          # PlanContext — run-scoped knobs (shared by all modules)
│   ├── llms.py             # proposer / critic / formatter model factories (shared)
│   ├── persistence.py      # work_dir lifecycle + artifact helpers (shared)
│   ├── tracing.py          # langfuse callback handler factory (shared)
│   ├── timing.py           # @timed_node wall-clock decorator (shared)
│   ├── cli.py              # `cmbagent-lg-cost` — per-agent cost/latency from langfuse
│   ├── planning/
│   │   ├── graph.py        # planner ↔ plan_reviewer graph
│   │   ├── nodes.py        # planner / format_plan / plan_reviewer / format_review
│   │   ├── schemas.py      # Plan, Step, Review
│   │   ├── state.py  prompts.py
│   │   └── templates/      # planner.yaml, plan_reviewer.yaml (vendored from cmbagent)
│   └── self_debug/
│       ├── graph.py        # engineer → executor → execution_evaluator → step_evaluator
│       ├── nodes.py        # the five nodes + two routers
│       ├── schemas.py      # EngineerResponse, ExecutionVerdict, StepVerdict
│       ├── state.py  prompts.py
│       └── templates/      # engineer.yaml, evaluator.yaml, step_evaluator.yaml
├── examples/
│   ├── run_planner_review.py
│   ├── run_self_debug.py            # happy path (primes — pure stdlib)
│   ├── run_self_debug_retry.py      # exercises the retry loop
│   ├── run_self_debug_lorenz.py     # produces a plot
│   ├── run_self_debug_csv_plot.py   # produces a CSV + a plot
│   └── trace_cost_summary.py
├── pyproject.toml
└── .env.example
```

## Install

```bash
python3.12 -m venv ~/pyvenvs/py312-cmbagent-lg
source ~/pyvenvs/py312-cmbagent-lg/bin/activate
pip install -e .
cp .env.example .env  # fill in GOOGLE_API_KEY (and LANGFUSE_* for tracing)
```

The `self_debug` examples run real generated code as a subprocess **in this venv**, so install whatever the examples need:

```bash
pip install numpy scipy matplotlib
```

## The two modules

### `planning`

```
START → planner → format_plan → [END if last round else continue]
                                 │
                                 ▼
                             plan_reviewer → format_review → planner …
```

`num_rounds` (from `PlanContext`) counts **review cycles**; total planner passes = `num_rounds + 1`. The final pass has seen every prior review. Generators write free-form prose; tiny formatter nodes convert it to typed Pydantic objects via `with_structured_output`.

### `self_debug`

```
engineer → format_engineer → executor → execution_evaluator
                                               │
                    code FAILURE  ◄────────────┤
                    (retry, or exhaust → END)  │ code SUCCESS
                                               ▼
                                         step_evaluator
                                               │
                    goal NOT met  ◄────────────┤
                    (retry, or exhaust → END)  │ goal MET
                                               ▼
                                              END
```

Two gates, one shared retry budget (`max_n_attempts`):

- **`execution_evaluator`** — did the code *run* cleanly? → `ExecutionVerdict`.
- **`step_evaluator`** — did the run *achieve the Step's goal*? → `StepVerdict`.

The `executor` writes `codebase/step_N.py` and runs it as a subprocess in the host venv (`cwd = work_dir`, so the script's `data/<file>` outputs land in `work_dir/data/`). Each run writes a predictable tree:

```
work_dir/
├── codebase/   step_N.py, step_N.log, step_N_failure_I.py (audit trail)
├── data/       output files the script produced
└── logs/       step_N_execution_verdict.json, step_N_verdict.json,
                step_N_data_manifest.json, step_N_timings.json
```

## Run

```bash
python examples/run_planner_review.py            # planning
python examples/run_self_debug_csv_plot.py       # self_debug (CSV + plot)
```

Both accept an optional explicit work_dir: `python examples/run_self_debug.py runs/my_run` (cleared on rerun; set `KEEP_WORK_DIR=1` to keep).

## Langfuse (self-hosted, optional)

The langfuse repo is cloned at `~/GitHub/langfuse`. To start it:

```bash
cd ~/GitHub/langfuse && docker compose up -d
# open http://localhost:3000, create a project, copy the public/secret keys
# into the LANGFUSE_* vars in .env
```

`cmbagent_lg/tracing.py` builds a LangChain `CallbackHandler` from those env vars; pass it via `config={"callbacks": [handler]}` and every LLM call (including formatter sub-calls) is traced automatically — no graph changes. Each call is tagged with its node name, so:

```bash
cmbagent-lg-cost <trace_id | work_dir>   # per-agent cost / tokens / latency
```

prints a cmbagent-style cost table.
