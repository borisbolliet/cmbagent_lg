"""Langfuse tracing — replaces Lab3's JSONL log.

The langfuse repo is cloned at ~/GitHub/langfuse. Bring it up with:

    cd ~/GitHub/langfuse && docker compose up -d

Then visit http://localhost:3000, create a project, and put its keys into .env
as LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.

Usage:

    from cmbagent_lg.tracing import langfuse_handler
    graph.invoke({}, context=ctx, config={"callbacks": [langfuse_handler()]})

Because LangChain callbacks propagate into child runs, every LLM call inside
the graph — including the `with_structured_output` formatter calls — shows up
in the langfuse trace automatically. No per-node plumbing needed.
"""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def langfuse_handler():
    """Build a `langfuse.langchain.CallbackHandler` from LANGFUSE_* env vars.

    Requires langfuse>=4 and the umbrella `langchain` package — both are
    declared in pyproject.toml. Cached per process so all `graph.invoke`
    calls share one handler (and one trace context).
    """
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set. "
            "Bring up the self-hosted instance (`cd ~/GitHub/langfuse && docker compose up -d`), "
            "create a project at http://localhost:3000, and paste its keys into .env."
        )

    from langfuse.langchain import CallbackHandler

    # langfuse v4 reads creds + host from env (LANGFUSE_PUBLIC_KEY,
    # LANGFUSE_SECRET_KEY, LANGFUSE_HOST) — no kwargs needed.
    return CallbackHandler()
