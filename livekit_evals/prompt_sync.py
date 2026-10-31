"""
Prompt sync: keep SuperBryn in step with the prompt your agent runs.

Every call carries ``context.prompt_ref = {"hash", "version"}``: which prompt it ran on. The template's text goes to
SuperBryn once per host per version (``POST {api_base}/public-api/v1/prompts``; a temp file every process on the
host reads, so a new process per call doesn't push it again; pushed again after a day), so a prompt edit you deploy shows
up in the agent's settings as a change you can turn into a new version. Calls are never held back by it: the push is
best effort, and a call whose prompt text has not arrived yet is matched when it does.

The hash is sha256 over the template's UTF-8 bytes exactly as given, ``sha256:<hex>``; any stack can compute it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import stat
import tempfile
import time
from typing import Any

logger = logging.getLogger("livekit_evals.prompt_sync")

PUSH_PATH = "/public-api/v1/prompts"
_PUSH_TIMEOUT_S = 5
_PUSHED_TTL_S = 24 * 3600  # in case SuperBryn lost a version, it is pushed again after a day
_REFUSED_TTL_S = 3600  # a push SuperBryn refused (key, scope, agent) is tried again after an hour, not on every call

# (host and key, hash) -> when this process pushed it or found it pushed; pushed again after a day, like the file record.
_pushed: dict[tuple[str, str], float] = {}
_refused: dict[tuple[str, str], float] = {}  # the same, for pushes SuperBryn refused
_warned: set[str] = set()


def prompt_hash(template: str) -> str:
    return "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()


def prompt_ref(template: str, version: str | None = None) -> dict[str, Any]:
    """What a call carries to name the prompt it ran on."""
    ref: dict[str, Any] = {"hash": prompt_hash(template)}
    if version:
        ref["version"] = version
    return ref


def _key_id(api_base_url: str, api_key: str) -> str:
    """Names a (base URL, API key) pair without the key itself."""
    return hashlib.sha256(f"{api_base_url}\n{api_key}".encode()).hexdigest()[:16]


def _pushed_file(api_base_url: str, api_key: str) -> str:
    """This host's record of the versions pushed with this key, in the temp dir."""
    return os.path.join(tempfile.gettempdir(), f"superbryn_prompts_pushed_{_key_id(api_base_url, api_key)}.json")


def _read_pushed(path: str) -> dict[str, float]:
    """Hash -> when it was pushed, for pushes under a day old; {} when missing or unreadable."""
    try:
        info = os.lstat(path)
        # only our own regular file: no link, pipe or device that could block, nor another user's record
        if not stat.S_ISREG(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        now = time.time()
        return {h: t for h, t in data.items() if isinstance(t, (int, float)) and 0 <= now - t < _PUSHED_TTL_S}
    except Exception:  # noqa: BLE001 - pushed instead
        return {}


def _remember_pushed(path: str, digest: str) -> None:
    """Add a pushed version to the host's record; never raises."""
    pushed = {**_read_pushed(path), digest: time.time()}
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=".superbryn_prompts_pushed_", dir=os.path.dirname(path))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(pushed, f)
        os.replace(tmp, path)  # atomic: a reader sees the old record or the new one
    except Exception as e:  # noqa: BLE001 - this process still remembers it
        logger.debug("SUPERBRYN_PROMPT_SYNC: pushed versions not recorded on disk: %s", e)
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


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
    agent_id: str | None = None,
) -> bool:
    """Send one prompt version to SuperBryn, once per host. agent_id names the agent for an org-scoped key. Never
    raises; True when SuperBryn has it."""
    digest = prompt_hash(template)
    if not api_key or not api_base_url:
        _warn_once("config", "SUPERBRYN_PROMPT_SYNC_SKIPPED: no API key or API base URL configured")
        return False
    seen = (_key_id(api_base_url, api_key), digest)
    if 0 <= time.time() - _pushed.get(seen, -_PUSHED_TTL_S) < _PUSHED_TTL_S:
        return True
    try:
        record = _pushed_file(api_base_url, api_key)
    except Exception:  # noqa: BLE001 - no temp dir: this process's memory only
        record = None
    on_host = _read_pushed(record) if record else {}
    if on_host.get(digest) is not None:
        _pushed[seen] = on_host[digest]
        return True
    refused_at = max(_refused.get(seen, 0.0), on_host.get("refused:" + digest, 0.0))
    if 0 <= time.time() - refused_at < _REFUSED_TTL_S:
        return False
    try:
        import aiohttp
    except ImportError:
        _warn_once(
            "aiohttp", "SUPERBRYN_PROMPT_SYNC_SKIPPED: install aiohttp to push prompt versions"
        )
        return False

    body: dict[str, Any] = {"template": template}
    if agent_id:
        body["agent_id"] = agent_id
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
                    _pushed[seen] = time.time()
                    if record:
                        _remember_pushed(record, digest)
                    logger.info(
                        "SUPERBRYN_PROMPT_SYNC_PUSHED: version=%s hash=%s", version, digest[:19]
                    )
                    return True
                if 400 <= resp.status < 500 and resp.status not in (408, 429):
                    _refused[seen] = time.time()
                    if record:
                        _remember_pushed(record, "refused:" + digest)
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
