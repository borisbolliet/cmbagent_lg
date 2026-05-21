"""Wall-clock timing decorator for graph nodes.

Wraps a node function so that:
  - elapsed wall-clock is measured (perf_counter — monotonic, sub-µs)
  - a single `[time] <name>: <x.xx>s` line is printed to stderr
  - the elapsed seconds are merged into the node's returned state delta as
    `node_elapsed_s` (a list, so multiple node passes in a loop accumulate)

We rely on `Annotated[list[...], operator.add]` reducers on the state to
concatenate `node_elapsed_s` across passes — see `state.py` of each module.

Langfuse already records per-span latency in the trace UI; this is the
local-console + on-disk counterpart, so a run is debuggable without
opening langfuse.
"""

from __future__ import annotations

import sys
import time
from functools import wraps
from typing import Callable, TypedDict


class NodeTiming(TypedDict):
    node: str
    elapsed_s: float


def timed_node(name: str) -> Callable:
    """Decorator: wrap a langgraph node fn to measure + record wall-clock time."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"[time] {name}: {elapsed:.2f}s", file=sys.stderr, flush=True)
            if isinstance(result, dict):
                result = {
                    **result,
                    "node_elapsed_s": [{"node": name, "elapsed_s": elapsed}],
                }
            return result

        return wrapper

    return decorator
