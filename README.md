# cmbagent_lg

LangGraph re-implementation of [cmbagent](https://github.com/CMBAgents/cmbagent) deep research.

Starting point: the `planner` ↔ `plan_reviewer` propose-critique loop, ported from the AG2 agents in `cmbagent/cmbagent/agents/planning/`. The Lab3 (`agents_lab_2026/Lab3.ipynb`) `idea_maker` / `idea_hater` graph is the architectural blueprint — same four-node propose → format → critique → format pattern, applied to planning.

Layout follows the [LangGraph application structure](https://docs.langchain.com/oss/python/langgraph/application-structure):

```
cmbagent_lg/
├── cmbagent_lg/
│   ├── __init__.py
│   ├── context.py        # run-scoped variables (improved_main_task, …)
│   ├── state.py          # IdeaState equivalent
│   ├── schemas.py        # Pydantic Plan + Review
│   ├── prompts.py        # YAML loader + schema_field_brief helper
│   ├── nodes.py          # planner / format_plan / plan_reviewer / format_review
│   ├── graph.py          # compiled StateGraph
│   ├── tracing.py        # langfuse callback handler factory
│   ├── llms.py           # model factory
│   └── templates/
│       ├── planner.yaml         (vendored from cmbagent)
│       └── plan_reviewer.yaml   (vendored from cmbagent)
├── examples/
│   └── run_planner_review.py
├── pyproject.toml
└── .env.example
```

## Install

```bash
python3.12 -m venv ~/pyvenvs/py312-cmbagent-lg
source ~/pyvenvs/py312-cmbagent-lg/bin/activate
pip install -e .
cp .env.example .env  # fill in keys
```

## Run

```bash
python examples/run_planner_review.py
```

## Langfuse (self-hosted)

The langfuse repo is cloned at `~/GitHub/langfuse`. To start it:

```bash
cd ~/GitHub/langfuse
docker compose up -d
# open http://localhost:3000, create a project, copy the public/secret keys
# into the LANGFUSE_* vars in .env
```

`cmbagent_lg/tracing.py` constructs a LangChain `CallbackHandler` from those env
vars; pass it into `graph.invoke(..., config={"callbacks": [handler]})` and
every LLM call (including the formatter sub-calls) is traced automatically.
