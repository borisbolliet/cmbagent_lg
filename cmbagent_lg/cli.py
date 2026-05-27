"""Command-line entry points.

Registered as console scripts in pyproject.toml. After `pip install -e .`:

    cmbagent-lg-cost                              # latest trace in project
    cmbagent-lg-cost <trace_id>                   # specific trace
    cmbagent-lg-cost <work_dir>                   # reads langfuse_trace_id.txt
    cmbagent-lg-cost <path/to/langfuse_trace_id.txt>

Also runnable as a module:

    python -m cmbagent_lg.cli cost [arg]
"""

import sys
import os
import base64
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv


def _auth_header() -> str:
    load_dotenv(override=True)
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    if not pk or not sk:
        sys.exit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set. "
            "Bring up langfuse and put keys in .env."
        )
    return base64.b64encode(f"{pk}:{sk}".encode()).decode()


def _get(path: str) -> dict:
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    req = urllib.request.Request(
        f"{host}{path}", headers={"Authorization": f"Basic {_auth_header()}"}
    )
    return json.loads(urllib.request.urlopen(req).read())


def _all_observations(trace_id: str) -> list[dict]:
    """Langfuse caps `limit` at 100; paginate."""
    out, page = [], 1
    while True:
        resp = _get(
            f"/api/public/observations?traceId={trace_id}&limit=100&page={page}"
        )
        batch = resp.get("data", [])
        out.extend(batch)
        if len(batch) < 100:
            return out
        page += 1


def _resolve_trace_id(arg: str) -> str:
    """Accept a trace ID, a workdir path, or a langfuse_trace_id.txt path."""
    p = Path(arg)
    if p.is_dir():
        p = p / "langfuse_trace_id.txt"
    if p.is_file():
        return p.read_text().strip()
    return arg  # not a path — treat as raw trace id


_KNOWN_AGENT_TAGS = (
    # planning module
    "planner",
    "plan_reviewer",
    "format_plan",
    "format_review",
    # self_debug module (executor makes no LLM call → never a GENERATION)
    "engineer",
    "format_engineer",
    "execution_evaluator",
    "step_evaluator",
    "escalation",
    # deep_research (outer orchestrator)
    "run_step",
)


def _latency_s(o: dict) -> float | None:
    """Return generation latency in seconds, or None if unknown.

    Langfuse stores `startTime` and `endTime` as ISO-8601 UTC strings
    (e.g. '2026-05-14T08:54:20.123Z'). The diff is the model's own
    end-to-end latency (request arrival → final token).
    """
    s, e = o.get("startTime"), o.get("endTime")
    if not s or not e:
        return None
    try:
        # `Z` suffix → +00:00 for fromisoformat
        ts0 = datetime.fromisoformat(s.replace("Z", "+00:00"))
        ts1 = datetime.fromisoformat(e.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (ts1 - ts0).total_seconds()


def trace_summary(trace_id: str) -> None:
    obs = _all_observations(trace_id)

    by_model: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "in_tok": 0, "out_tok": 0, "cost": 0.0}
    )
    by_tag: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "in_tok": 0,
            "out_tok": 0,
            "cost": 0.0,
            "models": set(),
            "latencies_s": [],
        }
    )

    for o in obs:
        if o.get("type") != "GENERATION":
            continue
        # langfuse 4.x: cost is in `calculatedTotalCost`, NOT `totalCost`
        # (`totalCost` on observations returns 0 — undocumented gotcha).
        cost = float(o.get("calculatedTotalCost") or 0)
        in_tok = o.get("promptTokens") or 0
        out_tok = o.get("completionTokens") or 0
        model = o.get("model") or "?"
        lat = _latency_s(o)

        by_model[model]["calls"] += 1
        by_model[model]["in_tok"] += in_tok
        by_model[model]["out_tok"] += out_tok
        by_model[model]["cost"] += cost

        for t in (
            (o.get("metadata") or {}).get("tags") or o.get("tags") or []
        ):
            if t in _KNOWN_AGENT_TAGS:
                row = by_tag[t]
                row["calls"] += 1
                row["in_tok"] += in_tok
                row["out_tok"] += out_tok
                row["cost"] += cost
                row["models"].add(model)
                if lat is not None:
                    row["latencies_s"].append(lat)

    print(f"trace {trace_id}\n")

    # --- by model (unchanged) ---
    print("--- by model ---")
    print(f"{'key':40s} {'calls':>6s} {'in_tok':>8s} {'out_tok':>8s} {'$cost':>10s}")
    total = 0.0
    for k, d in sorted(by_model.items(), key=lambda kv: -kv[1]["cost"]):
        print(
            f"{k:40s} {d['calls']:>6d} {d['in_tok']:>8d} "
            f"{d['out_tok']:>8d} {d['cost']:>10.6f}"
        )
        total += d["cost"]
    print(f"{'TOTAL':40s} {'':>6s} {'':>8s} {'':>8s} {total:>10.6f}\n")

    # --- by agent (enriched: model, calls, tokens, latency, cost) ---
    if not by_tag:
        return
    print("--- by agent (tag) ---")
    hdr = (
        f"{'agent':14s} {'model':30s} {'calls':>5s} {'in_tok':>7s} "
        f"{'out_tok':>7s} {'avg_s':>6s} {'min_s':>6s} {'max_s':>6s} {'$cost':>10s}"
    )
    print(hdr)
    total = 0.0
    for k, d in sorted(by_tag.items(), key=lambda kv: -kv[1]["cost"]):
        models = ", ".join(sorted(d["models"])) or "?"
        lats = d["latencies_s"]
        if lats:
            avg_s = f"{sum(lats) / len(lats):.2f}"
            min_s = f"{min(lats):.2f}"
            max_s = f"{max(lats):.2f}"
        else:
            avg_s = min_s = max_s = "-"
        print(
            f"{k:14s} {models:30s} {d['calls']:>5d} {d['in_tok']:>7d} "
            f"{d['out_tok']:>7d} {avg_s:>6s} {min_s:>6s} {max_s:>6s} "
            f"{d['cost']:>10.6f}"
        )
        total += d["cost"]
    print(
        f"{'TOTAL':14s} {'':30s} {'':>5s} {'':>7s} {'':>7s} "
        f"{'':>6s} {'':>6s} {'':>6s} {total:>10.6f}\n"
    )


def cost_main() -> None:
    """`cmbagent-lg-cost` entry point."""
    if len(sys.argv) > 1:
        trace_id = _resolve_trace_id(sys.argv[1])
    else:
        trace_id = _get("/api/public/traces?limit=1")["data"][0]["id"]
    trace_summary(trace_id)


def _module_main() -> None:
    """`python -m cmbagent_lg.cli <subcommand> [args]`."""
    sub = sys.argv[1] if len(sys.argv) > 1 else None
    if sub == "cost":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        cost_main()
    else:
        sys.exit(f"usage: python -m cmbagent_lg.cli cost [trace_id|work_dir]")


if __name__ == "__main__":
    _module_main()
