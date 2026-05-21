"""Model factory.

Three roles, mirroring Lab3:
- `proposer`  — generates plans (the `planner`).
- `critic`    — generates reviews (the `plan_reviewer`).
- `formatter` — deterministic, converts prose → Pydantic via `with_structured_output`.

Dev defaults: gemini-3.1-flash-lite (stable) everywhere — cheap, fully on the
3.1 line. The `-preview` variant of this same model was prone to token-
repetition loops on the formatter; the stable variant should be steadier.

Per-role overrides via env without touching code, e.g. upgrade the critic
for a sharper review pass:

    CRITIQUE_MODEL=gemini-3.1-pro-preview python examples/run_planner_review.py

If the formatter ever loops again, the safest non-preview flash-family
fallback is `gemini-3-flash-preview`:

    FORMAT_MODEL=gemini-3-flash-preview python examples/run_planner_review.py
"""

import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# The google-genai SDK logs "Both GOOGLE_API_KEY and GEMINI_API_KEY are set.
# Using GOOGLE_API_KEY." at WARNING once per client construction
# (`google.genai._api_client.get_env_api_key`). We deliberately standardize
# on GOOGLE_API_KEY and pass it explicitly, so the notice is pure noise —
# silence just that logger.
logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)


def _api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return key


_DEFAULT_MODEL = "gemini-3.1-flash-lite"


def proposer():
    return ChatGoogleGenerativeAI(
        model=os.environ.get("PROPOSE_MODEL", _DEFAULT_MODEL),
        temperature=1.0,
        thinking_level="low",
        google_api_key=_api_key(),
    )


def critic():
    # Same flash-lite as the proposer — cheapest. The multi-model split is
    # preserved via the `tags=["plan_reviewer"]` and the temperature
    # difference, not via a different model name.
    return ChatGoogleGenerativeAI(
        model=os.environ.get("CRITIQUE_MODEL", _DEFAULT_MODEL),
        temperature=0.4,
        thinking_level="low",
        google_api_key=_api_key(),
    )


def formatter():
    return ChatGoogleGenerativeAI(
        model=os.environ.get("FORMAT_MODEL", _DEFAULT_MODEL),
        temperature=0.0,
        google_api_key=_api_key(),
    )
