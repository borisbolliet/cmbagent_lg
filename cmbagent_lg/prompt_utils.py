"""Shared prompt-loading helpers used by `planning` and `self_debug`.

The two modules' `prompts.py` files both render YAML templates against a
`PlanContext`-derived dict using `str.format_map(SafeDict(...))`, where
missing placeholders render as empty strings (so a template doesn't break
mid-loop when a field is absent). Those helpers — `SafeDict`, the
`importlib.resources` YAML loader, and the schema-field-brief renderer —
are not module-specific, so they live here once.
"""

from importlib import resources
import yaml


class SafeDict(dict):
    """`format_map` helper: missing keys render as empty strings instead of KeyError."""

    def __missing__(self, key):
        return ""


def load_yaml(package: str, name: str) -> dict:
    """Load `<package>/<name>` as a parsed YAML dict via `importlib.resources`.

    Pass the dotted package path (e.g. `"cmbagent_lg.planning.templates"`)
    and the YAML filename (e.g. `"planner.yaml"`).
    """
    text = resources.files(package).joinpath(name).read_text()
    return yaml.safe_load(text)


def schema_field_brief(schema) -> str:
    """One bullet per Pydantic field — drop into a generator prompt so the
    LLM knows which fields to cover before the formatter extracts them."""
    lines = []
    for name, field in schema.model_fields.items():
        t = str(field.annotation).replace("typing.", "")
        if t.startswith("<class '") and t.endswith("'>"):
            t = t[len("<class '") : -len("'>")]
        lines.append(f"- {name} ({t}): {field.description or ''}")
    return "\n".join(lines)
