"""
Agent config sync for livekit-evals (opt-in).

Builds a SuperBryn AgentSyncManifest from a LiveKit ``Agent`` /
``AgentSession`` and pushes it to
``POST {BASE_URL}/public-api/v1/agents/me/sync``. Nothing here runs unless
the customer explicitly calls ``sync_config`` / ``async_sync_config`` —
installing the webhook handler alone never syncs.

Requires an **agent-scoped** API key (created against a single agent in the
SuperBryn dashboard); org-scoped keys are rejected by the endpoint. The
pushed manifest lands as a pending draft that a human approves in the
review UI — syncing never changes the live agent directly.

Extraction reads a fixed allow-list of configuration attributes on the
pipeline components — including private fields such as ``_opts``, ``_model``
and ``_voice`` where LiveKit plugins keep their settings. Credential
attributes (API keys, tokens, secrets) are never part of that list and are
never read or transmitted.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Literal, TypedDict

from ._component_unwrap import (
    _INNER_ATTRS_BY_ROLE,
    _fallback_instances,
    _unwrap_to_base_component,
)
from .config import BASE_URL, WEBHOOK_CONFIG

logger = logging.getLogger("superbryn.livekit.sync")

SYNC_PATH = "/public-api/v1/agents/me/sync"


# ── config section shapes ────────────────────────────────────────────────
# These mirror the canonical AgentSyncManifest schema field-for-field. The
# sync endpoint validates strictly (unknown keys → HTTP 400), so the same
# key sets are enforced client-side in build_manifest_from_agent and a
# typo fails at the call site instead of at deploy time.


class IdentityConfig(TypedDict, total=False):
    name: str
    type: Literal["inbound", "outbound"]
    agent_modality: Literal["voice", "chat"]
    description: str
    pain_point: str | None
    gender: str | None
    age: int
    dob: str  # ISO date, e.g. "1990-01-31"


class BehaviorConfig(TypedDict, total=False):
    prompt: str | None
    flow: dict[str, Any] | None


class ToolServerConfig(TypedDict, total=False):
    type: Literal["http", "mcp", "native"]
    url: str


class ToolConfig(TypedDict, total=False):
    name: str
    description: str
    schema: Any
    server: ToolServerConfig


class AdditionalLanguage(TypedDict, total=False):
    code: str
    priority: int


class LanguageConfig(TypedDict, total=False):
    primary_language: str | None
    additional_languages: list[AdditionalLanguage]


class IvrConfig(TypedDict, total=False):
    enabled: bool
    number: str


class TelephonyConfig(TypedDict, total=False):
    phone_number: str | None
    ivr_config: IvrConfig | None


def _check_keys(section: str, value: Any, allowed: frozenset[str]) -> None:
    """Reject unknown keys locally — the server schema is strict and would 400."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError(f"{section} must be a dict, got {type(value).__name__}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"unknown {section} field(s) {unknown}; allowed fields: {sorted(allowed)}. "
            "The SuperBryn sync endpoint rejects unknown keys."
        )


_IDENTITY_KEYS = frozenset(IdentityConfig.__annotations__)
_BEHAVIOR_KEYS = frozenset(BehaviorConfig.__annotations__)
_TOOL_KEYS = frozenset(ToolConfig.__annotations__)
_TOOL_SERVER_KEYS = frozenset(ToolServerConfig.__annotations__)
_LANGUAGE_KEYS = frozenset(LanguageConfig.__annotations__)
_ADDITIONAL_LANGUAGE_KEYS = frozenset(AdditionalLanguage.__annotations__)
_IVR_KEYS = frozenset(IvrConfig.__annotations__)
_TELEPHONY_KEYS = frozenset(TelephonyConfig.__annotations__)


def _validate_config_sections(
    identity: Any,
    behavior: Any,
    tools: Any,
    language: Any,
    telephony: Any,
) -> None:
    _check_keys("identity", identity, _IDENTITY_KEYS)
    _check_keys("behavior", behavior, _BEHAVIOR_KEYS)
    if tools is not None:
        for i, tool in enumerate(tools):
            _check_keys(f"tools[{i}]", tool, _TOOL_KEYS)
            if isinstance(tool, dict):
                _check_keys(f"tools[{i}].server", tool.get("server"), _TOOL_SERVER_KEYS)
    if language is not None:
        _check_keys("language", language, _LANGUAGE_KEYS)
        if isinstance(language, dict):
            for i, entry in enumerate(language.get("additional_languages") or []):
                _check_keys(f"language.additional_languages[{i}]", entry, _ADDITIONAL_LANGUAGE_KEYS)
    if telephony is not None:
        _check_keys("telephony", telephony, _TELEPHONY_KEYS)
        if isinstance(telephony, dict):
            _check_keys("telephony.ivr_config", telephony.get("ivr_config"), _IVR_KEYS)


# LiveKit plugins live under livekit.plugins.<provider>, e.g.
# livekit.plugins.deepgram.stt.STT → "deepgram".
_MODEL_ATTRS = ("model", "_model", "model_name", "_opts.model", "opts.model")
_VOICE_ATTRS = (
    "voice",
    "_voice",
    "voice_id",
    "_voice_id",
    "_opts.voice",
    "opts.voice",
    "_opts.voice_id",
    "opts.voice_id",
)
_LANGUAGE_ATTRS = ("language", "_language", "_opts.language", "opts.language")
_TEMPERATURE_ATTRS = ("temperature", "_temperature", "_opts.temperature", "opts.temperature")
_MAX_TOKENS_ATTRS = ("max_tokens", "_max_tokens", "_opts.max_tokens", "opts.max_tokens")


def _read_attr_chain(source: Any, *names: str) -> Any:
    """First non-empty attribute along dotted candidate paths (never raises)."""
    for name in names:
        cur: Any = source
        for part in name.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur not in (None, ""):
            return cur
    return None


def _provider_from_plugin(obj: Any) -> str | None:
    """Extract the provider from a livekit.plugins.<provider>.* module path."""
    module = type(obj).__module__ or ""
    parts = module.split(".")
    if "plugins" in parts:
        idx = parts.index("plugins")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _extract_tools_from_agent(agent: Any) -> list[ToolConfig] | None:
    """Manifest tool entries from the agent's registered function tools.

    Reads ``agent.tools`` (falling back to ``agent._tools``). Tools created
    with ``@function_tool`` carry a ``__livekit_tool_info`` dataclass
    (name/description); raw tools carry ``__livekit_raw_tool_info`` with the
    provider-format ``raw_schema``. Plain callables fall back to
    ``__name__`` / ``__doc__``.
    """
    tools = getattr(agent, "tools", None)
    if tools is None:
        tools = getattr(agent, "_tools", None)
    if not tools:
        return None

    entries: list[ToolConfig] = []
    for tool in tools:
        entry: ToolConfig = {}

        raw_info = getattr(tool, "__livekit_raw_tool_info", None)
        raw_schema = getattr(raw_info, "raw_schema", None)
        if isinstance(raw_schema, dict):
            fn = raw_schema.get("function") if raw_schema.get("type") == "function" else raw_schema
            if isinstance(fn, dict):
                if isinstance(fn.get("name"), str) and fn["name"]:
                    entry["name"] = fn["name"]
                if isinstance(fn.get("description"), str) and fn["description"]:
                    entry["description"] = fn["description"]
                if fn.get("parameters") is not None:
                    entry["schema"] = fn["parameters"]

        if not entry:
            info = getattr(tool, "__livekit_tool_info", None)
            name = getattr(info, "name", None) or getattr(tool, "__name__", None)
            description = getattr(info, "description", None) or getattr(tool, "__doc__", None)
            if isinstance(name, str) and name:
                entry["name"] = name
            if isinstance(description, str) and description.strip():
                entry["description"] = description.strip()

        if entry:
            entries.append(entry)

    return entries or None


def _extract_component_fields(obj: Any, role: str) -> dict[str, Any]:
    """Read {provider, model, ...} fields from a single (unwrapped) plugin object."""
    block: dict[str, Any] = {}

    provider = _provider_from_plugin(obj)
    if provider:
        block["provider"] = provider

    model = _read_attr_chain(obj, *_MODEL_ATTRS)
    if isinstance(model, str) and model:
        block["model"] = model

    if role == "llm":
        temperature = _read_attr_chain(obj, *_TEMPERATURE_ATTRS)
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
            block["temperature"] = temperature
        max_tokens = _read_attr_chain(obj, *_MAX_TOKENS_ATTRS)
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
            block["max_tokens"] = max_tokens

    if role == "stt":
        language = _read_attr_chain(obj, *_LANGUAGE_ATTRS)
        if isinstance(language, str) and language:
            block["language"] = language

    if role == "tts":
        voice = _read_attr_chain(obj, *_VOICE_ATTRS)
        if isinstance(voice, str) and voice:
            block["voice_id"] = voice

    return block


def _extract_component(obj: Any, role: str) -> dict[str, Any] | None:
    """Extract a {provider, model, ...} block from a LiveKit component.

    Components are often not the plugin itself but a wrapper —
    ``FallbackAdapter``, ``StreamAdapter``, or a custom class holding the
    real plugin in an inner attribute. Those wrappers carry no provider
    module path and no model/voice attributes, so extraction on the wrapper
    yields nothing. Unwrap to the base plugin first (cycle-safe), and for
    ``FallbackAdapter`` report the first non-primary instance in the
    manifest's ``fallback`` sub-block.
    """
    if obj is None:
        return None

    inner_attrs = _INNER_ATTRS_BY_ROLE[role]
    base = _unwrap_to_base_component(obj, inner_attrs)
    block = _extract_component_fields(base if base is not None else obj, role)

    fallbacks = _fallback_instances(obj, inner_attrs)
    if fallbacks:
        fallback_base = _unwrap_to_base_component(fallbacks[0], inner_attrs)
        fallback_block = _extract_component_fields(
            fallback_base if fallback_base is not None else fallbacks[0], role
        )
        if fallback_block:
            block["fallback"] = fallback_block

    return block or None


def build_manifest_from_agent(
    agent: Any,
    *,
    source: str = "livekit",
    identity: IdentityConfig | None = None,
    behavior: BehaviorConfig | None = None,
    tools: list[ToolConfig] | None = None,
    language: LanguageConfig | None = None,
    telephony: TelephonyConfig | None = None,
    policy_guardrails: str | None = None,
    additional_details: str | None = None,
    concurrency_calls: int | None = None,
) -> dict[str, Any]:
    """Build an AgentSyncManifest dict from a LiveKit Agent / AgentSession.

    Reads ``agent.llm`` / ``agent.stt`` / ``agent.tts`` (accessors on both
    ``Agent`` and ``AgentSession``) to fill the top-level pipeline blocks,
    ``agent.instructions`` as the behavior prompt, and ``agent.tools`` as
    ``config.tools`` — unless explicit ``behavior`` / ``tools`` overrides
    are given, which always win. Everything the runtime genuinely can't
    know (identity, telephony, guardrails, concurrency, ...) is supplied
    through the keyword overrides.

    Override shapes mirror the canonical manifest schema exactly (see the
    TypedDicts at the top of this module):

    - ``identity``: ``name``, ``type`` ("inbound"/"outbound"),
      ``agent_modality`` ("voice"/"chat"), ``description``, ``pain_point``,
      ``gender``, ``age``, ``dob``
    - ``behavior``: ``prompt``, ``flow``
    - ``tools``: list of ``{name, description, schema, server}`` where
      ``server`` is ``{type: http|mcp|native, url}``
    - ``language``: ``primary_language``, ``additional_languages``
      (list of ``{code, priority}``)
    - ``telephony``: ``phone_number``, ``ivr_config``
      (``{enabled, number}``)

    Unknown keys in these overrides raise ``ValueError`` here — the sync
    endpoint validates strictly, so this surfaces typos at the call site
    instead of as an HTTP 400.

    All manifest fields are optional — extraction failures degrade to a
    sparser manifest, never an error.
    """
    _validate_config_sections(identity, behavior, tools, language, telephony)

    manifest: dict[str, Any] = {"source": source}

    try:
        for role in ("llm", "stt", "tts"):
            component = getattr(agent, role, None)
            block = _extract_component(component, role)
            if block:
                manifest[role] = block
    except Exception as exc:  # noqa: BLE001 — never break the customer's agent
        logger.debug("agent extraction failed: %s", exc)

    tts_block = manifest.get("tts")
    if isinstance(tts_block, dict) and tts_block.get("voice_id"):
        voice_block = {
            k: v
            for k, v in (
                ("provider", tts_block.get("provider")),
                ("voice_id", tts_block["voice_id"]),
            )
            if v
        }
        tts_fallback = tts_block.get("fallback")
        if isinstance(tts_fallback, dict) and tts_fallback.get("voice_id"):
            voice_fallback = {
                k: v
                for k, v in (
                    ("provider", tts_fallback.get("provider")),
                    ("voice_id", tts_fallback["voice_id"]),
                )
                if v
            }
            if voice_fallback:
                voice_block["fallback"] = voice_fallback
        manifest["voice"] = voice_block

    if behavior is None:
        try:
            instructions = getattr(agent, "instructions", None)
            if isinstance(instructions, str) and instructions.strip():
                behavior = {"prompt": instructions}
        except Exception as exc:  # noqa: BLE001 — never break the customer's agent
            logger.debug("instructions extraction failed: %s", exc)

    if tools is None:
        try:
            tools = _extract_tools_from_agent(agent)
        except Exception as exc:  # noqa: BLE001 — never break the customer's agent
            logger.debug("tool extraction failed: %s", exc)

    config: dict[str, Any] = {}
    if identity is not None:
        config["identity"] = identity
    if behavior is not None:
        config["behavior"] = behavior
    if tools is not None:
        config["tools"] = tools
    if language is not None:
        config["language"] = language
    if telephony is not None:
        config["telephony"] = telephony
    if policy_guardrails is not None:
        config["policy_guardrails"] = policy_guardrails
    if additional_details is not None:
        config["additional_details"] = additional_details
    if concurrency_calls is not None:
        config["concurrency_calls"] = concurrency_calls
    if config:
        manifest["config"] = config

    return manifest


def _resolve_endpoint(base_url: str | None) -> str:
    base = (base_url or BASE_URL or "").rstrip("/")
    if not base:
        raise ValueError(
            "SuperBryn base URL is not configured — set SUPERBRYN_BASE_URL "
            "or pass base_url= to sync_config()"
        )
    return base + SYNC_PATH


def sync_manifest(
    manifest: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Push a prebuilt manifest to SuperBryn (blocking).

    Returns the parsed JSON response: ``{status: "noop", ...}`` for a
    hash-identical sync, or the accepted-draft body with ``agent_row_id``,
    ``approval_status``, ``verification_status``, ``hash`` and
    ``change_types``. Raises on transport errors and non-2xx responses so
    deploy pipelines fail loudly.
    """
    key = api_key or WEBHOOK_CONFIG.get("api_key") or ""
    if not key:
        raise ValueError("SuperBryn API key missing — set SUPERBRYN_API_KEY or pass api_key=")

    url = _resolve_endpoint(base_url)
    body = json.dumps(manifest).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": key,
            "User-Agent": "livekit-evals",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error("SUPERBRYN_SYNC_FAILED: HTTP %s — %s", exc.code, detail)
        raise
    logger.info("SUPERBRYN_SYNC_OK: %s", payload.get("status") or payload.get("approval_status"))
    return payload


def sync_config(
    agent: Any,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    **manifest_overrides: Any,
) -> dict[str, Any]:
    """One-liner: build a manifest from the agent and push it.

    ``manifest_overrides`` are forwarded to
    :func:`build_manifest_from_agent` (``identity=``, ``behavior=``,
    ``policy_guardrails=``, ...).
    """
    manifest = build_manifest_from_agent(agent, **manifest_overrides)
    return sync_manifest(manifest, api_key=api_key, base_url=base_url, timeout=timeout)


async def async_sync_config(
    agent: Any,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    **manifest_overrides: Any,
) -> dict[str, Any]:
    """Async variant of :func:`sync_config` using aiohttp."""
    import aiohttp

    key = api_key or WEBHOOK_CONFIG.get("api_key") or ""
    if not key:
        raise ValueError("SuperBryn API key missing — set SUPERBRYN_API_KEY or pass api_key=")

    manifest = build_manifest_from_agent(agent, **manifest_overrides)
    url = _resolve_endpoint(base_url)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": key,
        "User-Agent": "livekit-evals",
    }
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, json=manifest, headers=headers) as response:
            payload = await response.json()
            if response.status >= 400:
                logger.error("SUPERBRYN_SYNC_FAILED: HTTP %s — %s", response.status, payload)
                response.raise_for_status()
    logger.info("SUPERBRYN_SYNC_OK: %s", payload.get("status") or payload.get("approval_status"))
    return payload
