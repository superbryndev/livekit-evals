"""
Prompt sync: keep SuperBryn in step with the prompt your agent runs.

Every call carries ``metadata.prompt_ref = {"hash", "version"}``: which prompt it ran on. The template's text goes to
SuperBryn once per process per version (``POST {api_base}/public-api/v1/prompts``), so a prompt edit you deploy shows
up in the agent's settings as a change you can turn into a new version. Calls are never held back by it: the push is
best effort, and a call whose prompt text has not arrived yet is matched when it does.

The hash is sha256 over the template's UTF-8 bytes exactly as given, ``sha256:<hex>``; any stack can compute it.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger("livekit_evals.prompt_sync")

PUSH_PATH = "/public-api/v1/prompts"
_PUSH_TIMEOUT_S = 5

# Hashes pushed (or known to SuperBryn) by this process: each version goes over the wire once.
_pushed: set[str] = set()
_warned: set[str] = set()


def prompt_hash(template: str) -> str:
    return "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()


def prompt_ref(template: str, version: str | None = None) -> dict[str, Any]:
    """What a call carries to name the prompt it ran on."""
    ref: dict[str, Any] = {"hash": prompt_hash(template)}
    if version:
        ref["version"] = version
    return ref


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


async def push_prompt(
    *,
    api_key: str,
    api_base_url: str,
    template: str,
    version: str | None = None,
    first_message: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> bool:
    """Send one prompt version to SuperBryn, once per process. Never raises; True when SuperBryn has it."""
    digest = prompt_hash(template)
    if digest in _pushed:
        return True
    if not api_key or not api_base_url:
        _warn_once("config", "SUPERBRYN_PROMPT_SYNC_SKIPPED: no API key or API base URL configured")
        return False
    try:
        import aiohttp
    except ImportError:
        _warn_once(
            "aiohttp", "SUPERBRYN_PROMPT_SYNC_SKIPPED: install aiohttp to push prompt versions"
        )
        return False

    body: dict[str, Any] = {"template": template}
    if version:
        body["version"] = version
    if first_message:
        body["first_message"] = first_message
    if tools:
        body["tools"] = tools
    try:
        timeout = aiohttp.ClientTimeout(total=_PUSH_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                api_base_url.rstrip("/") + PUSH_PATH,
                json=body,
                headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            ) as resp:
                if resp.status in (200, 201):
                    _pushed.add(digest)
                    logger.info(
                        "SUPERBRYN_PROMPT_SYNC_PUSHED: version=%s hash=%s", version, digest[:19]
                    )
                    return True
                if resp.status == 403:
                    _warn_once(
                        "scope",
                        "SUPERBRYN_PROMPT_SYNC_FORBIDDEN: the API key lacks the obs:write scope. Generate a new SDK key "
                        "in SuperBryn (Monitor > Configure) to sync prompt versions; calls are still recorded.",
                    )
                else:
                    _warn_once(
                        f"status:{resp.status}",
                        "SUPERBRYN_PROMPT_SYNC_FAILED: HTTP %s",
                        resp.status,
                    )
    except Exception as exc:  # noqa: BLE001 — prompt sync never breaks call delivery
        _warn_once(f"error:{type(exc).__name__}", "SUPERBRYN_PROMPT_SYNC_FAILED: %s", exc)
    return False
