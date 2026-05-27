"""Shared helpers for the example scripts under `examples/`.

Every `run_*.py` script needs the same scaffolding: resolve the work_dir,
maybe attach a Langfuse handler, stream a graph node-by-node into a local
result dict (respecting `operator.add` reducers), print a wall-clock table,
print the Langfuse trace link. This module is that scaffolding, in one
place, so adding a new state field to skip in the stream renderer (etc.)
is a one-line edit here, not in 8 places.

Underscore prefix on the filename keeps it out of "which example should I
run" lists.
"""

from __future__ import annotations

import json as _json
import os
import sys
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel as _BaseModel

from cmbagent_lg import default_work_dir, prepare_work_dir, save_trace_id


# State fields every example wants suppressed from the streaming display —
# bookkeeping / reducer-managed lists that would clutter the per-node blocks.
# Per-example sets union with this.
STANDARD_SKIP_KEYS = frozenset({
    "attempts",
    "error_history",
    "work_dir",
    "node_elapsed_s",
    "step_number",
    "data_baseline",
    "data_manifest",
    "step_feedback_history",
    "escalated",
})


# Reducer-managed lists that `graph.stream(stream_mode="updates")` yields as
# deltas — we must concatenate, not overwrite, to mirror the graph's actual
# state at the end. Anything not in this set: overwrite (default).
DEFAULT_LIST_REDUCED_KEYS = frozenset({"node_elapsed_s"})


def resolve_work_dir(argv: list[str]) -> Path:
    """`argv[1] > $WORK_DIR > default_work_dir()`, then `prepare_work_dir`.

    `prepare_work_dir` clears any existing contents (so reruns into the same
    path start clean). Set `KEEP_WORK_DIR=1` in the env to opt out.
    """
    if len(argv) > 1:
        wd = Path(argv[1])
    elif os.environ.get("WORK_DIR"):
        wd = Path(os.environ["WORK_DIR"])
    else:
        wd = default_work_dir()
    clear = not os.environ.get("KEEP_WORK_DIR")
    if clear and wd.exists():
        print(f"[work_dir] clearing existing       {wd}")
    return prepare_work_dir(wd, clear=clear)


def attach_langfuse() -> tuple[object | None, list]:
    """Return `(handler, callbacks)` — `(None, [])` if Langfuse isn't configured."""
    try:
        from cmbagent_lg.tracing import langfuse_handler

        handler = langfuse_handler()
        print("[trace] langfuse handler attached")
        return handler, [handler]
    except Exception as e:  # noqa: BLE001 — Langfuse is optional everywhere
        print(f"[trace] skipping langfuse: {e}")
        return None, []


def render_value(v) -> str:
    """Pretty-print a state-delta value for the per-node display.

    Pydantic models render via their `.format()` if present (else JSON);
    JSON-looking strings are re-pretty-printed; everything else stringifies.
    """
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


def stream_and_render(
    graph,
    initial_state: dict,
    *,
    context,
    config: dict,
    skip_keys: Iterable[str] = (),
    list_reduced_keys: Iterable[str] = DEFAULT_LIST_REDUCED_KEYS,
) -> dict:
    """Run `graph.stream(...)` in `stream_mode="updates"`, printing each node's
    state delta and accumulating a local `result` dict that mirrors the in-graph
    state — including manually replaying `operator.add` reducers on `list_reduced_keys`
    so accumulated lists (like `node_elapsed_s`) survive the loop.

    Returns the accumulated `result` dict.
    """
    skip = frozenset(STANDARD_SKIP_KEYS) | frozenset(skip_keys)
    reduced = frozenset(list_reduced_keys)

    result = dict(initial_state)
    for chunk in graph.stream(
        initial_state, context=context, config=config, stream_mode="updates"
    ):
        for node_name, delta in chunk.items():
            print(f"\n\n══════════ {node_name} ══════════")
            for k, v in (delta or {}).items():
                if k in skip:
                    continue
                print(f"\n── {k} ──")
                print(render_value(v))
            for k, v in (delta or {}).items():
                if k in reduced:
                    result[k] = list(result.get(k, [])) + list(v or [])
                else:
                    result[k] = v
    return result


def print_timings(timings: list[dict]) -> None:
    """The wall-clock-per-node table. Skipped if `timings` is empty."""
    if not timings:
        return
    print("\n=== TIMINGS (wall-clock) ===")
    width = max(len(t["node"]) for t in timings)
    for t in timings:
        print(f"  {t['node']:<{width}}  {t['elapsed_s']:7.2f}s")
    print(f"  {'TOTAL':<{width}}  {sum(t['elapsed_s'] for t in timings):7.2f}s")


def print_self_debug_verdicts(result: dict, ctx) -> None:
    """The "EXECUTION VERDICT / STEP VERDICT / attempts" footer that every
    standalone self_debug example prints. Pulls the two verdicts (if present)
    and the `attempts / max_n_attempts` line out of the final state."""
    exec_verdict = result.get("current_execution_verdict")
    step_verdict = result.get("current_step_verdict")
    print("\n\n=== EXECUTION VERDICT (did the code run cleanly?) ===")
    print(exec_verdict.format() if exec_verdict else "(none — exited before execution_evaluator)")
    print("\n=== STEP VERDICT (did it achieve the goal?) ===")
    print(step_verdict.format() if step_verdict else "(none — the code never ran cleanly)")
    extras = ""
    if result.get("escalated"):
        extras = " | escalated"
    print(
        f"\n=== attempts: {result.get('attempts', 0)} / {ctx.max_n_attempts}{extras} ==="
    )


def save_node_timings(work_dir: Path, step_number: int, timings: list[dict]) -> Path:
    """Persist a step's per-node wall-clock to `logs/step_{N}_timings.json`."""
    out_dir = Path(work_dir) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"step_{step_number}_timings.json"
    out.write_text(
        _json.dumps(
            {
                "node_elapsed_s": timings,
                "total_node_s": sum(t["elapsed_s"] for t in timings),
            },
            indent=2,
        )
    )
    return out


def print_trace_info(handler, work_dir: Path) -> None:
    """Save the Langfuse trace id (if any) and print the UI link + cost CLI hint."""
    if handler is not None and getattr(handler, "last_trace_id", None):
        trace_id = handler.last_trace_id
        tid_path = save_trace_id(trace_id, work_dir)
        host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
        print(f"\n[work_dir] langfuse trace id in    {tid_path}")
        print(f"[trace]    open in UI:             {host}/trace/{trace_id}")
        print(f"[trace]    cost summary:           cmbagent-lg-cost {work_dir}")
    else:
        print("\n[trace]    (no trace id — langfuse handler not attached)")
