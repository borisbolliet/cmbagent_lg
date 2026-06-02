"""Engineer → format_engineer → executor → execution_evaluator → step_evaluator.

Flow — two gates, one shared retry budget (`max_n_attempts`):

    engineer ─► format_engineer ─► executor ─► execution_evaluator
                                                       │
                            code FAILURE ◄─────────────┤
                            (retry / exhaust→END)      │ code SUCCESS
                                                       ▼
                                                 step_evaluator
                                                       │
                            goal NOT met ◄─────────────┤
                            (retry / exhaust→END)      │ goal MET
                                                       ▼
                                                      END

`execution_evaluator` judges only whether the code RAN cleanly; `step_evaluator`
judges whether the run ACHIEVED the sub-task's goal (cmbagent's controller
role). A run can pass the first gate and fail the second.

`error_history` accumulates one summary per failed attempt so the engineer
sees the full debug trail on retries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime

from cmbagent_lg.context import PlanContext
from cmbagent_lg.llms import chat_model
from cmbagent_lg.prompt_utils import schema_field_brief
from cmbagent_lg.self_debug.prompts import (
    engineer_instructions,
    evaluator_instructions,
    step_evaluator_instructions,
    render_retry_context,
)
from cmbagent_lg.self_debug.schemas import (
    EngineerResponse,
    ExecutionVerdict,
    StepVerdict,
)
from cmbagent_lg.self_debug.state import DebugState
from cmbagent_lg.timing import timed_node


# Per-role chat models from the run context (None → llms._DEFAULT_MODEL).
# Here the "generator" is the engineer and the "critic" is the evaluator.
# chat_model caches by (model, role), so this stays lazy.
def _proposer(ctx: PlanContext):
    return chat_model(ctx.engineer_model, "generator")


def _critic(ctx: PlanContext):
    return chat_model(ctx.evaluator_model, "critic")


def _formatter(ctx: PlanContext):
    return chat_model(ctx.formatter_model, "formatter")


# ── generators ──────────────────────────────────────────────────────────


@timed_node("engineer")
def engineer(state: DebugState, runtime: Runtime[PlanContext]) -> DebugState:
    """Propose code for the current sub-task. Bumps `attempts`."""
    ctx = runtime.context
    step = state["step"]
    attempts = state.get("attempts", 0) + 1
    error_history = state.get("error_history", [])

    last_code = None
    last_stderr = None
    last_stdout = None
    last_returncode = None
    last_timed_out = None
    last_fix_suggestion = None
    last_step_unmet = None
    last_step_feedback = None
    if attempts > 1:
        prev = state.get("current_code")
        last_code = prev.python_code if prev else None
        last_stdout = state.get("execution_stdout")
        exec_verdict = state.get("current_execution_verdict")
        # Why are we retrying? If the last run's code crashed → code-error
        # retry. Otherwise the code ran clean and the step_evaluator rejected
        # it → goal-miss retry. Selecting the mode here avoids showing a
        # stale step verdict after a code crash.
        if exec_verdict is not None and exec_verdict.status == "failure":
            last_stderr = state.get("execution_stderr")
            last_returncode = state.get("execution_returncode")
            last_timed_out = state.get("execution_timed_out")
            last_fix_suggestion = exec_verdict.fix_suggestion
        else:
            step_verdict = state.get("current_step_verdict")
            if step_verdict is not None:
                last_step_unmet = step_verdict.unmet_requirements
                last_step_feedback = step_verdict.feedback

    retry_block = render_retry_context(
        attempts=attempts,
        max_attempts=ctx.max_n_attempts,
        last_code=last_code,
        last_stderr=last_stderr,
        last_stdout=last_stdout,
        last_returncode=last_returncode,
        last_timed_out=last_timed_out,
        last_fix_suggestion=last_fix_suggestion,
        error_history=error_history,
        last_step_unmet=last_step_unmet,
        last_step_feedback=last_step_feedback,
    )

    system = engineer_instructions(
        ctx,
        step,
        retry_block,
        previous_steps_execution_summary=state.get("previous_steps_execution_summary") or "",
    )
    user = (
        "Produce the script now, in the format described above. Write in natural "
        "prose around a single Python code block — a downstream specialist will "
        "extract the structured fields. Make sure your response covers:\n\n"
        + schema_field_brief(EngineerResponse)
    )
    msg = _proposer(runtime.context).invoke(
        [SystemMessage(system), HumanMessage(user)],
        config={"tags": ["engineer"]},
    )
    return {"raw_engineer": msg.text, "attempts": attempts}


@timed_node("format_engineer")
def format_engineer(state: DebugState, runtime: Runtime[PlanContext]) -> DebugState:
    """Convert the engineer's prose into a typed `EngineerResponse`."""
    structured = _formatter(runtime.context).with_structured_output(EngineerResponse)
    sys_prompt = SystemMessage(
        "You are a formatter. Convert the user's text into an EngineerResponse "
        "object. Preserve the Python code VERBATIM — do not reformat, lint, or "
        "alter it. The code block is the substantive content; do not summarize it."
    )
    obj = structured.invoke(
        [sys_prompt, HumanMessage(state["raw_engineer"])],
        config={"tags": ["format_engineer"]},
    )
    return {"current_code": obj}


# ── on-disk layout ───────────────────────────────────────────────────────
#
# Two flat sibling dirs under work_dir, mirroring cmbagent's `codebase/`:
#
#   {work_dir}/codebase/step_{N}.py              ← canonical, overwritten each attempt
#   {work_dir}/codebase/step_{N}.log             ← stdout+stderr of the latest run
#   {work_dir}/codebase/step_{N}_failure_{I}.py  ← attempt I's code, kept on failure
#   {work_dir}/codebase/step_{N}_failure_{I}.log ← attempt I's run log, kept on failure
#   {work_dir}/data/                                  ← output files the script produces
#   {work_dir}/logs/step_{N}_execution_verdict.json   ← did the code RUN cleanly
#   {work_dir}/logs/step_{N}_timings.json             ← per-node wall-clock (written by caller)
#   {work_dir}/logs/step_{N}_data_manifest.json       ← which data/ files this step produced
#
# On failure the evaluator demotes `step_{N}.{py,log}` → `step_{N}_failure_{I}.*`,
# freeing the canonical name for the next attempt's overwrite. cmbagent gets the
# rename for free from AG2's executor; we do it explicitly so it's traceable.
#
# The data manifest records step→file provenance externally — the data files
# and their names are never touched, and the engineer is never asked to embed
# a step id in its code.


def _codebase_dir(state: DebugState) -> Optional[Path]:
    """`{work_dir}/codebase/` — generated scripts live here. None if no work_dir."""
    raw = state.get("work_dir")
    if not raw:
        return None
    out = Path(raw).expanduser() / "codebase"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _logs_dir(state: DebugState) -> Optional[Path]:
    """`{work_dir}/logs/` — verdicts + run metadata, flat, sibling of codebase/.

    Files are step-prefixed (`step_{N}_execution_verdict.json`) rather than
    nested in a per-step folder. None if no work_dir.
    """
    raw = state.get("work_dir")
    if not raw:
        return None
    out = Path(raw).expanduser() / "logs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_log(returncode: int, timed_out: bool, elapsed: float, stdout: str, stderr: str) -> str:
    return (
        f"# returncode={returncode} timed_out={timed_out} elapsed={elapsed:.4f}s\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
    )


def _snapshot_data_dir(data_dir: Path) -> dict:
    """Map filename → mtime for every file currently in `data/`."""
    if not data_dir.is_dir():
        return {}
    return {p.name: p.stat().st_mtime for p in data_dir.iterdir() if p.is_file()}


def _data_manifest(data_dir: Path, baseline: dict) -> list:
    """Files in `data/` that are new or modified vs `baseline` — i.e. produced
    by the current step. Paths are relative (`data/<name>`) so they read the
    same way the engineer referred to them."""
    out = []
    if not data_dir.is_dir():
        return out
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        st = p.stat()
        prior = baseline.get(p.name)
        if prior is None or st.st_mtime > prior:
            out.append(
                {
                    "path": f"data/{p.name}",
                    "bytes": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            )
    return out


# ── executor ────────────────────────────────────────────────────────────


@timed_node("executor")
def executor(state: DebugState, runtime: Runtime[PlanContext]) -> DebugState:
    """Write `codebase/step_{N}.py` and run it in a subprocess (host venv).

    The subprocess runs with `cwd = work_dir`, so the script's relative output
    paths (`data/<file>`) land in `{work_dir}/data/` — cmbagent's convention.
    The script's own directory (`codebase/`) stays on `sys.path[0]` regardless
    of cwd, so cross-step `from step_1 import …` still resolves.
    """
    ctx = runtime.context
    code = state["current_code"].python_code
    step = state["step"]
    n = state.get("step_number", 1)
    timeout = step.code_execution_timeout or ctx.code_execution_timeout
    codebase = _codebase_dir(state)

    # Canonical name when work_dir is set; ephemeral tempfile otherwise
    # (deleted in the finally below — the evaluator reads output from state,
    # not disk, so the file is only needed for the subprocess run itself).
    data_dir = None
    data_baseline = state.get("data_baseline")
    if codebase is not None:
        code_path = codebase / f"step_{n}.py"
        # {work_dir}/data/ — where the script saves plots & generated data.
        run_cwd = codebase.parent
        data_dir = run_cwd / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Re-snapshot data/ at the START of *every* attempt (not just the
        # first). The manifest below is then the files produced by the LATEST
        # attempt only: throwaway plots/files written by an earlier *failed*
        # attempt land in this snapshot's baseline and are excluded, so only the
        # successful (final) attempt's outputs are attributed to the step and
        # propagate downstream. Each attempt is a complete standalone script, so
        # the successful attempt regenerates everything it needs.
        data_baseline = _snapshot_data_dir(data_dir)
    else:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            code_path = Path(f.name)
        run_cwd = None
    if data_baseline is None:
        data_baseline = {}
    code_path.write_text(code)

    # Absolute script path — it must be absolute or it'd resolve relative to
    # the cwd we hand subprocess.run below.
    code_path = code_path.resolve()
    if run_cwd is None:
        run_cwd = code_path.parent

    timed_out = False
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(code_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(run_cwd),
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")) + (
            f"\n[killed: TimeoutExpired after {timeout}s]"
        )
        # 124 is the GNU `timeout` convention; the `timed_out` flag carries
        # the real signal — see DebugState.execution_timed_out.
        returncode = 124
    finally:
        if codebase is None:
            code_path.unlink(missing_ok=True)
    execution_elapsed_s = time.perf_counter() - t0
    print(
        f"[time] executor.subprocess: {execution_elapsed_s:.2f}s (timeout={timeout}s, timed_out={timed_out})",
        file=sys.stderr,
        flush=True,
    )

    manifest = _data_manifest(data_dir, data_baseline) if data_dir is not None else []

    if codebase is not None:
        (codebase / f"step_{n}.log").write_text(
            _run_log(returncode, timed_out, execution_elapsed_s, stdout, stderr)
        )
        # Provenance: record which data/ files this step produced — external
        # to the files themselves (names + bytes untouched).
        logs = _logs_dir(state)
        if logs is not None:
            (logs / f"step_{n}_data_manifest.json").write_text(
                json.dumps(
                    {
                        "step_number": n,
                        "sub_task": step.sub_task,
                        "attempt": state.get("attempts", 0),
                        "files": manifest,
                    },
                    indent=2,
                )
            )

    return {
        "execution_stdout": stdout,
        "execution_stderr": stderr,
        "execution_returncode": returncode,
        "execution_timed_out": timed_out,
        "execution_elapsed_s": execution_elapsed_s,
        "data_baseline": data_baseline,
        "data_manifest": manifest,
    }


# ── execution_evaluator ─────────────────────────────────────────────────


@timed_node("execution_evaluator")
def execution_evaluator(state: DebugState, runtime: Runtime[PlanContext]) -> DebugState:
    """Did the code RUN cleanly? One structured-output call → ExecutionVerdict.

    This is *not* a step-goal judgment — see `step_evaluator`.
    """
    ctx = runtime.context
    step = state["step"]
    system = evaluator_instructions(
        ctx,
        step,
        executed_code=state["current_code"].python_code,
        stdout=state.get("execution_stdout", ""),
        stderr=state.get("execution_stderr", ""),
        returncode=state.get("execution_returncode", -1),
        timed_out=state.get("execution_timed_out", False),
    )
    user = (
        "Judge the execution above and emit a verdict. Cover these fields:\n\n"
        + schema_field_brief(ExecutionVerdict)
    )
    structured = _critic(runtime.context).with_structured_output(ExecutionVerdict)
    verdict: ExecutionVerdict = structured.invoke(
        [SystemMessage(system), HumanMessage(user)],
        config={"tags": ["execution_evaluator"]},
    )

    history = list(state.get("error_history", []))
    if verdict.status == "failure":
        # error_summary is the evaluator's own words; fall back to the last
        # stderr line if it left the field empty. Both branches yield a str.
        summary = verdict.error_summary or (
            state.get("execution_stderr", "").strip().splitlines() or ["(no stderr)"]
        )[-1]
        history.append(summary)

    n = state.get("step_number", 1)
    attempt = state.get("attempts", 0)

    logs = _logs_dir(state)
    if logs is not None:
        (logs / f"step_{n}_execution_verdict.json").write_text(
            verdict.model_dump_json(indent=2)
        )

    # On failure, demote the canonical script/log to failure-variants so the
    # next attempt's executor pass can overwrite `step_{N}.py` cleanly. The
    # failed code + its run log are preserved as an audit trail.
    codebase = _codebase_dir(state)
    if codebase is not None and verdict.status == "failure":
        for ext in ("py", "log"):
            canonical = codebase / f"step_{n}.{ext}"
            if canonical.exists():
                canonical.rename(codebase / f"step_{n}_failure_{attempt}.{ext}")

    return {"current_execution_verdict": verdict, "error_history": history}


# ── step_evaluator ──────────────────────────────────────────────────────


@timed_node("step_evaluator")
def step_evaluator(state: DebugState, runtime: Runtime[PlanContext]) -> DebugState:
    """Did the run ACHIEVE the sub-task's goal? Runs only after the code is
    known to have executed cleanly. One structured-output call → StepVerdict.
    Judges from stdout + the data manifest (file paths/sizes, not contents).
    """
    ctx = runtime.context
    step = state["step"]
    prior_feedback = state.get("step_feedback_history", [])
    system = step_evaluator_instructions(
        ctx,
        step,
        stdout=state.get("execution_stdout", ""),
        data_manifest=state.get("data_manifest", []),
        step_feedback_history=prior_feedback,
    )
    user = (
        "Judge whether the step goal was achieved and emit a verdict. "
        "Cover these fields:\n\n" + schema_field_brief(StepVerdict)
    )
    structured = _critic(runtime.context).with_structured_output(StepVerdict)
    verdict: StepVerdict = structured.invoke(
        [SystemMessage(system), HumanMessage(user)],
        config={"tags": ["step_evaluator"]},
    )

    n = state.get("step_number", 1)
    attempt = state.get("attempts", 0)

    # On a goal-miss, record this attempt's feedback so the *next*
    # step_evaluator pass sees its own trail (mirrors `error_history`).
    feedback_history = list(prior_feedback)
    if not verdict.fulfilled:
        bits = []
        if verdict.unmet_requirements:
            bits.append("unmet — " + "; ".join(verdict.unmet_requirements))
        if verdict.feedback:
            bits.append("feedback — " + verdict.feedback)
        feedback_history.append(
            f"attempt {attempt}: " + (" | ".join(bits) if bits else "goal not met")
        )

    logs = _logs_dir(state)
    if logs is not None:
        (logs / f"step_{n}_verdict.json").write_text(verdict.model_dump_json(indent=2))

    # A goal-miss is a failure too: demote this attempt's script/log to
    # failure-variants so the audit trail covers BOTH gates (the code here
    # ran cleanly, so execution_evaluator left the canonical name in place).
    # An attempt fails exactly one gate, so `failure_{attempt}` never collides.
    codebase = _codebase_dir(state)
    if codebase is not None and not verdict.fulfilled:
        for ext in ("py", "log"):
            canonical = codebase / f"step_{n}.{ext}"
            if canonical.exists():
                canonical.rename(codebase / f"step_{n}_failure_{attempt}.{ext}")

    return {
        "current_step_verdict": verdict,
        "step_feedback_history": feedback_history,
    }


# ── routers ─────────────────────────────────────────────────────────────


# Failure kinds the strict loop structurally cannot fix — the engineer can't
# install packages, and a renamed/removed API may need information the model
# lacks. Classification comes from `execution_evaluator` (an LLM that reads
# stdout+stderr) — NOT a stderr regex, because the engineer often catches the
# exception and prints it to stdout, leaving no traceback to match.
_ESCALATABLE_KINDS = ("missing_module", "renamed_api")


def route_after_execution_evaluator(
    state: DebugState, runtime: Runtime[PlanContext]
) -> str:
    """Code gate. success → step_evaluator. On failure: an escalatable failure
    (missing package / renamed API) routes to `escalation` once per step, even
    if attempts are exhausted — it's the escape hatch precisely for when the
    strict loop is stuck. Otherwise: engineer (retry) / END (exhausted)."""
    ctx = runtime.context
    verdict = state["current_execution_verdict"]
    if verdict.status == "success":
        return "step_evaluator"
    if (
        ctx.enable_escalation
        and not state.get("escalated", False)
        and verdict.failure_kind in _ESCALATABLE_KINDS
    ):
        return "escalation"
    if state.get("attempts", 0) >= ctx.max_n_attempts:
        return END
    return "engineer"


def route_after_step_evaluator(
    state: DebugState, runtime: Runtime[PlanContext]
) -> str:
    """Goal gate: fulfilled → END; not fulfilled & attempts left → engineer;
    not fulfilled & exhausted → END."""
    ctx = runtime.context
    if state["current_step_verdict"].fulfilled:
        return END
    if state.get("attempts", 0) >= ctx.max_n_attempts:
        return END
    return "engineer"


__all__ = [
    "engineer",
    "format_engineer",
    "executor",
    "execution_evaluator",
    "step_evaluator",
    "route_after_execution_evaluator",
    "route_after_step_evaluator",
]
