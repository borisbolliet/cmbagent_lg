"""Eval demo on GSM8K for teaching what an eval actually IS.

A from-scratch LangGraph eval harness — two nodes (`solve`, `grade`) and a
straight edge — so the eval is visible in the graph itself: a model node
that *answers*, and a deterministic node that *grades*. The grader is part
of the graph on purpose: the lesson is that **the harness is as much the
eval as the model is.**

Three things this demo is built to show live:

  1. The harness is the eval. We grade every answer two ways — `strict`
     (must end with `#### <number>`, the GSM8K convention) and `lenient`
     (just take the last number anywhere). The *same generations* score
     differently. The gap is "right number, wrong format" — points the
     model earned but the harness threw away.

  2. A benchmark number is meaningless without the protocol. pass@1 (one
     sample), pass@k (any of k correct — an upper bound), and maj@k
     (majority vote of k samples — self-consistency). Same model, same
     questions, three different numbers.

  3. Temperature does something visible. At T=0 the k samples are nearly
     identical, so maj@k ≈ pass@1. Raise T and maj@k pulls ahead of pass@1
     (self-consistency helps) while pass@k rises faster still (diversity
     helps if you have a verifier).

Usage:
  python examples/eval_demo.py --temperature 0.0 --repeat 1     # plain accuracy
  python examples/eval_demo.py --temperature 0.8 --repeat 5     # pass@k / maj@k
  python examples/eval_demo.py --grader lenient                 # watch the score move
  python examples/eval_demo.py --show-trace                     # one full solve→grade trace

Optional first positional arg is a work_dir (Langfuse session id + where the
trace_id file is written); defaults to a timestamped path.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from cmbagent_lg.prompt_utils import flatten_content as _flatten_content

from _common import attach_langfuse, print_trace_info, resolve_work_dir

# ── the eval set ─────────────────────────────────────────────────────────
# Five canonical GSM8K problems, embedded so the demo runs offline. Gold
# answers are the single integers GSM8K grades on. Pass --use-hf to pull a
# larger sample from the real dataset instead (needs `datasets` installed).

GSM8K_SAMPLE = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold "
     "half as many clips in May. How many clips did she sell altogether in "
     "April and May?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
     "minutes of babysitting. How much did she earn?", 10),
    ("Betty is saving money for a new wallet which costs $100. Betty has only "
     "half of the money she needs. Her parents decided to give her $15 for "
     "that purpose, and her grandparents twice as much as her parents. How "
     "much more money does Betty need to buy the wallet?", 5),
    ("James writes a 3-page letter to 2 different friends twice a week. How "
     "many pages does he write a year?", 624),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How "
     "many bolts in total does it take?", 3),
]


def load_problems(use_hf: bool, n: int) -> list[tuple[str, int]]:
    """Return `[(question, gold_int), ...]`. Embedded sample unless --use-hf."""
    if not use_hf:
        return GSM8K_SAMPLE[:n] if n else GSM8K_SAMPLE
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=f"test[:{n or 8}]")
    out = []
    for ex in ds:
        gold = int(ex["answer"].split("####")[-1].strip().replace(",", ""))
        out.append((ex["question"], gold))
    return out


# ── answer extraction = the grader. Two policies, deliberately. ──────────

_NUM = re.compile(r"-?\d[\d,]*")


def extract_strict(text: str) -> int | None:
    """GSM8K convention: the answer is the number after the final `####`."""
    if "####" not in text:
        return None
    tail = text.rsplit("####", 1)[-1]
    m = _NUM.search(tail)
    return int(m.group().replace(",", "")) if m else None


def extract_lenient(text: str) -> int | None:
    """Just take the last number anywhere in the response."""
    nums = _NUM.findall(text)
    return int(nums[-1].replace(",", "")) if nums else None


# ── graph state ──────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    gold: int
    final_text: str
    strict: int | None
    lenient: int | None


# ── graph builder ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a careful grade-school math tutor. Think step by step, then on "
    "the final line write the answer as `#### <number>` and nothing else."
)


def make_graph(temperature: float):
    model = ChatGoogleGenerativeAI(
        model=os.environ.get("EVAL_MODEL", "gemini-3.1-flash-lite"),
        temperature=temperature,
        thinking_level="low",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    def solve_node(state: State):
        return {"messages": [model.invoke(state["messages"])]}

    def grade_node(state: State):
        text = _flatten_content(getattr(state["messages"][-1], "content", ""))
        return {
            "final_text": text,
            "strict": extract_strict(text),
            "lenient": extract_lenient(text),
        }

    # NB: no `compile(cache=...)` and no per-node CachePolicy on purpose.
    # `solve` must re-run on every invocation — the k samples for one question
    # share an identical input state, so node caching would collapse them all
    # to one result and silently kill pass@k / maj@k diversity. Sample
    # diversity here comes only from temperature>0 server-side sampling.
    g = StateGraph(State)
    g.add_node("solve", solve_node)
    g.add_node("grade", grade_node)
    g.add_edge(START, "solve")
    g.add_edge("solve", "grade")
    g.add_edge("grade", END)
    return g.compile()


# ── runner ───────────────────────────────────────────────────────────────

def run_once(graph, question: str, gold: int, config: dict | None = None) -> dict:
    final = graph.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "gold": gold,
        },
        config=config,
    )
    return {
        "strict": final.get("strict"),
        "lenient": final.get("lenient"),
        "final_text": final.get("final_text", ""),
        "messages": final["messages"],
    }


def print_trace(messages):
    """Pretty-print the full solve→grade trace of one run — for live demos."""
    for i, m in enumerate(messages):
        role = m.__class__.__name__.replace("Message", "")
        body = _flatten_content(getattr(m, "content", "")).strip()
        print(f"\n[{i}] {role}")
        if body:
            print(f"    {body}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir", nargs="?", default=None,
                    help="Optional work_dir (used as Langfuse session id).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Samples per question (k). >1 enables pass@k / maj@k.")
    ap.add_argument("--grader", choices=["strict", "lenient"], default="strict",
                    help="Which extractor scores pass@k / maj@k.")
    ap.add_argument("--num", type=int, default=0,
                    help="How many problems to use (0 = all).")
    ap.add_argument("--use-hf", action="store_true",
                    help="Pull problems from the real gsm8k dataset.")
    ap.add_argument("--show-trace", action="store_true",
                    help="Print one full solve→grade trace then exit.")
    args = ap.parse_args()

    # ── workdir + langfuse ───────────────────────────────────────────────
    fake_argv = [sys.argv[0]] + ([args.work_dir] if args.work_dir else [])
    work_dir = resolve_work_dir(fake_argv)
    handler, callbacks = attach_langfuse()

    tags = ["eval_demo", f"T={args.temperature}", f"k={args.repeat}", work_dir.name]
    base_metadata = {"langfuse_session_id": work_dir.name, "langfuse_tags": tags}

    def make_config(label: str) -> dict:
        return {
            "callbacks": callbacks,
            "run_name": f"eval_demo · T={args.temperature} · {label}",
            "tags": tags,
            "metadata": base_metadata,
        }

    graph = make_graph(args.temperature)
    problems = load_problems(args.use_hf, args.num)
    pick = (lambda r: r["strict"]) if args.grader == "strict" else (lambda r: r["lenient"])

    if args.show_trace:
        q, gold = problems[0]
        print(f"\n=== Trace (T={args.temperature}) ===\nQ: {q}\nGold: {gold}")
        out = run_once(graph, q, gold, config=make_config("show-trace"))
        print_trace(out["messages"])
        print(f"\nextracted  strict={out['strict']}  lenient={out['lenient']}  gold={gold}")
        print_trace_info(handler, work_dir)
        return

    print(f"\n=== GSM8K eval — T={args.temperature}, k={args.repeat}, "
          f"grader={args.grader}, n={len(problems)} ===\n")

    # ── what pass@1, pass@k and maj@k mean ───────────────────────────────
    #
    # For each question we draw k=`--repeat` independent samples from the
    # model (diversity comes from temperature>0; at T=0 the k samples are
    # near-identical). From the k extracted answers we compute three metrics,
    # each averaged over the n questions:
    #
    #   pass@1  — is the FIRST sample correct? This is "plain accuracy": what
    #             you get from one shot per question. Independent of k.
    #
    #   pass@k  — is ANY of the k samples correct? An OPTIMISTIC ceiling: it
    #             assumes a perfect verifier that can pick the right answer
    #             out of the k. Monotonically rises with k and is only
    #             achievable in practice if you actually have such a verifier
    #             (a unit test, a proof checker, ...). For free-form QA you
    #             usually don't, so pass@k overstates real-world accuracy.
    #
    #   maj@k   — take the MAJORITY VOTE of the k extracted answers, then check
    #             that against gold. This is "self-consistency" (Wang et al.,
    #             2022): no verifier needed, you just trust the consensus. It
    #             denoises a model that is right on average but wobbly per
    #             sample, so typically  pass@1 ≤ maj@k ≤ pass@k.
    #
    # Worked example for one question, k=5, extracted answers [72, 72, 71, 72, 70],
    # gold=72:
    #   pass@1 = 1 (first is 72 ✓);  pass@k = 1 (a 72 appears ✓);
    #   maj@k  = 1 (72 wins the vote 3–1–1 ✓).
    # If instead they were [71, 72, 71, 70, 71]: pass@1 = 0 (first is 71 ✗),
    # pass@k = 1 (a 72 is in there ✓), maj@k = 0 (71 wins the vote, so consensus
    # is wrong ✗) — the case that separates the optimistic ceiling from what
    # voting actually recovers.
    #
    # ── Aggregate counters across all problems. ──────────────────────────
    pass1 = passk = majk = 0          # under the chosen grader
    strict_ok = lenient_ok = 0        # pass@1 under each grader (the harness gap)

    for qi, (q, gold) in enumerate(problems):
        samples = [
            run_once(graph, q, gold, config=make_config(f"q{qi+1} sample {s+1}/{args.repeat}"))
            for s in range(args.repeat)
        ]
        preds = [pick(r) for r in samples]            # extracted answer per sample
        first_correct = preds[0] == gold
        any_correct = any(p == gold for p in preds)
        # majority vote over the (non-None) extracted answers
        votes = Counter(p for p in preds if p is not None)
        vote = votes.most_common(1)[0][0] if votes else None
        maj_correct = vote == gold

        pass1 += int(first_correct)
        passk += int(any_correct)
        majk += int(maj_correct)
        strict_ok += int(samples[0]["strict"] == gold)
        lenient_ok += int(samples[0]["lenient"] == gold)

        flags = "".join("." if p == gold else "x" for p in preds)
        print(f"  q{qi+1:>2}  gold={gold:<5}  pass@1={'Y' if first_correct else 'n'}"
              f"  maj={vote}  [{flags}]")

    n = len(problems)
    print(f"\n--- metrics under '{args.grader}' grader (n={n}, k={args.repeat}) ---")
    print(f"  pass@1 (first sample) : {pass1}/{n} = {pass1/n:.0%}")
    if args.repeat > 1:
        print(f"  maj@{args.repeat}   (self-consistency) : {majk}/{n} = {majk/n:.0%}")
        print(f"  pass@{args.repeat}  (any correct, oracle): {passk}/{n} = {passk/n:.0%}")
    print(f"\n--- the harness gap (pass@1, first sample) ---")
    print(f"  strict  (#### required) : {strict_ok}/{n} = {strict_ok/n:.0%}")
    print(f"  lenient (last number)   : {lenient_ok}/{n} = {lenient_ok/n:.0%}")
    if lenient_ok != strict_ok:
        print(f"  → {lenient_ok - strict_ok} answer(s) were RIGHT but the strict "
              f"harness scored them wrong (formatting, not reasoning).")
    print_trace_info(handler, work_dir)


if __name__ == "__main__":
    main()
