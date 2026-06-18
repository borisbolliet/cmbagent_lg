"""Smoke-test the API keys in `.env` against each provider.

For every provider cmbagent_lg can use, this checks two independent things:

  1. **key auth** — a cheap, no-cost auth probe (a `GET .../models` style call)
     that proves the key in `.env` actually authenticates with the provider.
  2. **client wiring** — whether the LangChain package `llms.chat_model()`
     lazily imports for that provider is installed, i.e. whether cmbagent_lg
     can really drive the provider end-to-end.

A key can be valid while the client is missing (e.g. OPENAI_API_KEY works but
`langchain-openai` isn't installed yet) — the report shows both so you know
exactly what to fix.

    python examples/smoke_test_keys.py            # auth + wiring checks (no token cost)
    python examples/smoke_test_keys.py --live     # also do a 1-token round-trip
                                                  #   through llms.chat_model() for
                                                  #   every provider whose client is installed

Exit code is non-zero if any *present* key fails to authenticate, so this is
CI-friendly. Absent optional keys are reported as SKIP, not failure.

Note: Mistral is intentionally flagged informational — cmbagent_lg's
`llms._provider()` has no mistral branch, so a `mistral-*` model name would
route to Google. The key is still validated in case you wire it in later.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import urllib.error
import urllib.request
from base64 import b64encode
from pathlib import Path

from dotenv import load_dotenv

# Load the repo's .env explicitly (independent of cwd), matching the other examples.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

TIMEOUT = 20  # seconds per probe

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, BAD, SKIP = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}–{RESET}"


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _get(url: str, headers: dict[str, str]) -> tuple[bool, str]:
    """GET `url`; return (ok, detail). `ok` is True only on HTTP 2xx."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        # 401/403 → bad key; surface the status so the cause is obvious.
        return False, f"HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return False, f"unreachable: {e.reason}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# ── per-provider auth probes ───────────────────────────────────────────────
# Each returns (ok, detail). They authenticate the key with a read-only,
# zero-token endpoint so running the smoke test costs nothing.

def probe_google(key: str) -> tuple[bool, str]:
    return _get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        headers={},
    )


def probe_openai(key: str) -> tuple[bool, str]:
    return _get("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})


def probe_anthropic(key: str) -> tuple[bool, str]:
    return _get(
        "https://api.anthropic.com/v1/models",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )


def probe_mistral(key: str) -> tuple[bool, str]:
    return _get("https://api.mistral.ai/v1/models", {"Authorization": f"Bearer {key}"})


def probe_langfuse() -> tuple[bool, str]:
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (host and pk and sk):
        missing = [n for n, v in
                   [("LANGFUSE_HOST", host), ("LANGFUSE_PUBLIC_KEY", pk),
                    ("LANGFUSE_SECRET_KEY", sk)] if not v]
        return False, f"missing {', '.join(missing)}"
    token = b64encode(f"{pk}:{sk}".encode()).decode()
    ok, detail = _get(
        f"{host.rstrip('/')}/api/public/projects",
        {"Authorization": f"Basic {token}"},
    )
    return ok, f"{detail} @ {host}"


# name, env var, probe, the langchain module llms.chat_model() needs, notes
PROVIDERS = [
    ("GOOGLE",    "GOOGLE_API_KEY",    probe_google,    "langchain_google_genai", True,  ""),
    ("OPENAI",    "OPENAI_API_KEY",    probe_openai,    "langchain_openai",        False, ""),
    ("ANTHROPIC", "ANTHROPIC_API_KEY", probe_anthropic, "langchain_anthropic",     False, "used by self_debug escalation"),
    ("MISTRAL",   "MISTRAL_API_KEY",   probe_mistral,   None,                      False, "not wired into llms._provider()"),
]

# Tiny, cheap models for the optional --live round-trip, keyed by provider.
LIVE_MODELS = {
    "GOOGLE": "gemini-3.1-flash-lite",
    "OPENAI": "gpt-5.4",
    "ANTHROPIC": "claude-sonnet-4-6",
}


def _content_text(content) -> str:
    """LangChain message `.content` is a str on most providers but a list of
    content blocks on Gemini-3 / Anthropic. Flatten either to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def live_invoke(name: str) -> tuple[bool, str]:
    """One-token round-trip through the real `llms.chat_model()` path."""
    model = LIVE_MODELS.get(name)
    if not model:
        return False, "no live model mapped"
    try:
        from cmbagent_lg.llms import chat_model

        llm = chat_model(model, "formatter")
        resp = llm.invoke("Reply with the single word: pong")
        text = _content_text(getattr(resp, "content", "")).strip().replace("\n", " ")
        return True, f"{model} → {text[:40]!r}"
    except Exception as e:  # noqa: BLE001
        return False, f"{model}: {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--live", action="store_true",
        help="also do a 1-token round-trip through llms.chat_model() per installed client",
    )
    args = ap.parse_args()

    print(f"\ncmbagent_lg · API key smoke test  {DIM}(.env at {Path(__file__).resolve().parents[1] / '.env'}){RESET}")
    print("=" * 70)

    failures = 0   # present key that fails auth
    skipped = 0    # absent key

    for name, env_var, probe, module, required, note in PROVIDERS:
        key = os.environ.get(env_var)
        suffix = f"  {DIM}{note}{RESET}" if note else ""

        if not key:
            tag = "REQUIRED, MISSING" if required else "optional, not set"
            mark = BAD if required else SKIP
            print(f"{mark} {name:10} {DIM}{env_var}{RESET}  {tag}{suffix}")
            if required:
                failures += 1
            else:
                skipped += 1
            continue

        ok, detail = probe(key)
        if module is None:
            client = f"{DIM}n/a{RESET}"
        elif _installed(module):
            client = f"client {OK} {DIM}{module}{RESET}"
        else:
            client = f"client {BAD} {DIM}{module} (pip install {module.replace('_', '-')}){RESET}"

        mark = OK if ok else BAD
        print(f"{mark} {name:10} auth {OK if ok else BAD} {DIM}({detail}){RESET}  {client}{suffix}")
        if not ok:
            failures += 1

        if args.live and ok and (module is None or _installed(module)) and name in LIVE_MODELS:
            live_ok, live_detail = live_invoke(name)
            print(f"    {OK if live_ok else BAD} live {DIM}{live_detail}{RESET}")
            if not live_ok:
                failures += 1

    # Langfuse (tracing) — separate shape (basic auth, optional)
    ok, detail = probe_langfuse()
    lf_installed = _installed("langfuse")
    client = f"client {OK if lf_installed else BAD} {DIM}langfuse{RESET}"
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        print(f"{OK if ok else BAD} {'LANGFUSE':10} auth {OK if ok else BAD} {DIM}({detail}){RESET}  {client}")
        if not ok:
            failures += 1
    else:
        print(f"{SKIP} {'LANGFUSE':10} {DIM}not configured — tracing disabled{RESET}  {client}")
        skipped += 1

    print("=" * 70)
    status = f"{GREEN}all present keys OK{RESET}" if failures == 0 else f"{RED}{failures} failure(s){RESET}"
    print(f"{status}  ·  {skipped} skipped\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
