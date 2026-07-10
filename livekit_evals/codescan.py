"""
Static source scanning for config fields the runtime can't expose (opt-in).

A focused port of ``superbryn.codescan`` (the SuperBryn Python SDK): where
the SDK scans for the whole manifest, this module only targets the
``config`` sections a running LiveKit agent can never tell us —
identity, telephony, policy guardrails, additional details, concurrency —
plus the behavior prompt as a fallback when the agent has no instructions.

Activate it by passing ``scan_root=`` to ``sync_config()`` /
``build_manifest_from_agent()``. Explicit keyword overrides always win,
then runtime extraction, then the scan.

Python files are parsed with ``ast`` — keyword arguments on any call
(``agent_name=``, ``phone_number=``, ``policy_guardrails=``, ...) and
module-level constant assignments (``POLICY_GUARDRAILS = "..."``) are
collected; simple variable references resolve to their assigned string
constants. Scanning is best-effort and read-only: unparseable files are
skipped and secrets are never read (only the known config keys are
collected).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

PY_SUFFIXES = {".py"}
SKIP_DIRS = {
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    "site-packages",
}

# Key sets are conservative on purpose — generic names ("description",
# "name") would produce false positives all over a codebase.
PROMPT_KEYS = {
    "prompt",
    "system_prompt",
    "systemprompt",
    "instructions",
    "system_message",
    "systemmessage",
    "agent_prompt",
    "base_prompt",
    "general_prompt",
}
NAME_KEYS = {"agent_name", "bot_name", "assistant_name"}
DESCRIPTION_KEYS = {"agent_description", "bot_description", "assistant_description"}
PHONE_KEYS = {"phone_number", "phonenumber", "from_number", "fromnumber", "agent_phone_number"}
GUARDRAILS_KEYS = {"policy_guardrails", "policy_guardrail", "guardrails"}
DETAILS_KEYS = {"additional_details", "additionaldetails"}
CONCURRENCY_KEYS = {"concurrency_calls", "concurrent_calls", "max_concurrent_calls"}

_BUCKET_BY_KEY: dict[str, str] = {}
for _keys, _bucket in (
    (PROMPT_KEYS, "prompt"),
    (NAME_KEYS, "name"),
    (DESCRIPTION_KEYS, "description"),
    (PHONE_KEYS, "phone_number"),
    (GUARDRAILS_KEYS, "policy_guardrails"),
    (DETAILS_KEYS, "additional_details"),
    (CONCURRENCY_KEYS, "concurrency_calls"),
):
    for _key in _keys:
        _BUCKET_BY_KEY[_key] = _bucket

# Minimum length before a prompt-key string is considered a real system
# prompt (filters out placeholders like prompt="hi").
MIN_PROMPT_LENGTH = 40

_PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,20}$")


class _Visitor(ast.NodeVisitor):
    """Collects (bucket, value) hits from call kwargs and constant assignments."""

    def __init__(self, hits: dict[str, list[Any]]):
        self.hits = hits
        self.assignments: dict[str, Any] = {}

    def _record(self, key: str, value: Any) -> None:
        bucket = _BUCKET_BY_KEY.get(key.lower())
        if bucket is None or value is None or value == "":
            return
        self.hits.setdefault(bucket, []).append(value)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignments[target.id] = node.value.value
                    self._record(target.id, node.value.value)
        self.generic_visit(node)

    def _resolve(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.assignments.get(node.id)
        if isinstance(node, ast.JoinedStr):
            # f-string: keep only the constant parts (best effort).
            parts = [
                v.value
                for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            ]
            return "".join(parts) or None
        return None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        for keyword in node.keywords:
            if keyword.arg is not None:
                self._record(keyword.arg, self._resolve(keyword.value))
        self.generic_visit(node)


def _scan_files(root: str | Path) -> dict[str, list[Any]]:
    root_path = Path(root)
    paths = (
        [root_path]
        if root_path.is_file()
        else [
            p
            for p in sorted(root_path.rglob("*"))
            if p.is_file()
            and p.suffix in PY_SUFFIXES
            and not any(part in SKIP_DIRS for part in p.parts)
        ]
    )
    hits: dict[str, list[Any]] = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        _Visitor(hits).visit(tree)
    return hits


def _longest_str(values: list[Any], minimum: int = 1) -> str | None:
    texts = [v for v in values if isinstance(v, str) and v.strip()]
    if not texts:
        return None
    best = max(texts, key=len)
    return best if len(best) >= minimum else None


def _first_str(values: list[Any]) -> str | None:
    return next((v for v in values if isinstance(v, str) and v.strip()), None)


def scan_source_config(root: str | Path) -> dict[str, Any]:
    """Scan a file/directory and return the config sections it can find.

    Returns a partial dict with any of: ``identity`` (name/description),
    ``behavior`` (prompt), ``telephony`` (phone_number),
    ``policy_guardrails``, ``additional_details``, ``concurrency_calls``.
    Sections with no hits are simply absent.
    """
    hits = _scan_files(root)
    config: dict[str, Any] = {}

    identity: dict[str, Any] = {}
    name = _first_str(hits.get("name", []))
    if name:
        identity["name"] = name
    description = _first_str(hits.get("description", []))
    if description:
        identity["description"] = description
    if identity:
        config["identity"] = identity

    prompt = _longest_str(hits.get("prompt", []), minimum=MIN_PROMPT_LENGTH)
    if prompt:
        config["behavior"] = {"prompt": prompt}

    phone = next(
        (
            v
            for v in hits.get("phone_number", [])
            if isinstance(v, str) and _PHONE_RE.match(v.strip())
        ),
        None,
    )
    if phone:
        config["telephony"] = {"phone_number": phone.strip()}

    guardrails = _longest_str(hits.get("policy_guardrails", []))
    if guardrails:
        config["policy_guardrails"] = guardrails

    details = _longest_str(hits.get("additional_details", []))
    if details:
        config["additional_details"] = details

    concurrency = next(
        (
            v
            for v in hits.get("concurrency_calls", [])
            if isinstance(v, int) and not isinstance(v, bool)
        ),
        None,
    )
    if concurrency is not None:
        config["concurrency_calls"] = concurrency

    return config
