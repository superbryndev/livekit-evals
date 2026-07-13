"""Manifest extraction: direct plugins, wrappers, fallbacks, secrets, tools, overrides."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from livekit_evals.config_sync import (
    _LANGUAGE_ATTRS,
    _MAX_TOKENS_ATTRS,
    _MODEL_ATTRS,
    _TEMPERATURE_ATTRS,
    _VOICE_ATTRS,
    build_manifest_from_agent,
)

from .conftest import (
    FakeAgent,
    FakeCustomWrapper,
    FakeFallbackAdapter,
    FakeStreamAdapter,
    make_llm,
    make_stt,
    make_tts,
)

# ── direct components ────────────────────────────────────────────────────


def test_direct_components_full_pipeline():
    agent = FakeAgent(llm=make_llm(), stt=make_stt(), tts=make_tts(), instructions="x" * 50)
    manifest = build_manifest_from_agent(agent)

    assert manifest["llm"] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.4,
        "max_tokens": 512,
    }
    assert manifest["stt"] == {"provider": "deepgram", "model": "nova-3", "language": "en"}
    assert manifest["tts"] == {
        "provider": "elevenlabs",
        "model": "eleven_turbo_v2",
        "voice_id": "voice-abc",
    }
    assert manifest["voice"] == {"provider": "elevenlabs", "voice_id": "voice-abc"}
    assert manifest["config"]["behavior"]["prompt"] == "x" * 50
    assert manifest["source"] == "livekit"


def test_missing_components_are_omitted():
    manifest = build_manifest_from_agent(FakeAgent())
    for key in ("llm", "stt", "tts", "voice"):
        assert key not in manifest


# ── wrapped components ───────────────────────────────────────────────────


def test_stream_adapter_wrapper_is_unwrapped():
    agent = FakeAgent(tts=FakeStreamAdapter(make_tts()))
    manifest = build_manifest_from_agent(agent)
    assert manifest["tts"]["provider"] == "elevenlabs"
    assert manifest["tts"]["voice_id"] == "voice-abc"
    assert manifest["voice"]["voice_id"] == "voice-abc"


def test_nested_custom_wrappers_are_unwrapped():
    agent = FakeAgent(tts=FakeCustomWrapper(FakeCustomWrapper(make_tts(provider="cartesia"))))
    manifest = build_manifest_from_agent(agent)
    assert manifest["tts"]["provider"] == "cartesia"


def test_cyclic_wrapper_chain_terminates():
    a = FakeCustomWrapper(None)
    b = FakeCustomWrapper(a)
    a._inner = b  # cycle
    agent = FakeAgent(tts=a)
    manifest = build_manifest_from_agent(agent)  # must not hang or raise
    assert "voice" not in manifest


# ── fallback adapters ────────────────────────────────────────────────────


def test_tts_fallback_adapter_primary_and_fallback():
    primary = make_tts(provider="elevenlabs", voice_id="v-primary")
    secondary = make_tts(provider="cartesia", model="sonic-2", voice_id="v-fallback")
    agent = FakeAgent(tts=FakeFallbackAdapter([primary, secondary], "_tts_instances"))
    manifest = build_manifest_from_agent(agent)

    assert manifest["tts"]["provider"] == "elevenlabs"
    assert manifest["tts"]["voice_id"] == "v-primary"
    assert manifest["tts"]["fallback"] == {
        "provider": "cartesia",
        "model": "sonic-2",
        "voice_id": "v-fallback",
    }
    assert manifest["voice"] == {
        "provider": "elevenlabs",
        "voice_id": "v-primary",
        "fallback": {"provider": "cartesia", "voice_id": "v-fallback"},
    }


def test_llm_fallback_adapter():
    primary = make_llm(provider="openai")
    secondary = make_llm(provider="anthropic", model="claude-sonnet-4-5")
    agent = FakeAgent(llm=FakeFallbackAdapter([primary, secondary], "_llm_instances"))
    manifest = build_manifest_from_agent(agent)

    assert manifest["llm"]["provider"] == "openai"
    assert manifest["llm"]["fallback"]["provider"] == "anthropic"
    assert manifest["llm"]["fallback"]["model"] == "claude-sonnet-4-5"


def test_fallback_adapter_nested_inside_custom_wrapper():
    adapter = FakeFallbackAdapter(
        [make_stt(provider="deepgram"), make_stt(provider="sarvam", model="saarika")],
        "_stt_instances",
    )
    agent = FakeAgent(stt=FakeCustomWrapper(adapter))
    manifest = build_manifest_from_agent(agent)

    assert manifest["stt"]["provider"] == "deepgram"
    assert manifest["stt"]["fallback"]["provider"] == "sarvam"


def test_single_instance_fallback_adapter_has_no_fallback_block():
    agent = FakeAgent(tts=FakeFallbackAdapter([make_tts()], "_tts_instances"))
    manifest = build_manifest_from_agent(agent)
    assert manifest["tts"]["provider"] == "elevenlabs"
    assert "fallback" not in manifest["tts"]


# ── secret non-extraction ────────────────────────────────────────────────

SECRET = "sk-live-EXTREMELY-SECRET-VALUE"


def test_secrets_never_reach_the_manifest():
    llm = make_llm(api_key=SECRET, _api_key=SECRET)
    llm._opts.api_key = SECRET
    stt = make_stt(_credentials=SECRET)
    tts = make_tts(token=SECRET)
    tts._opts.api_key = SECRET
    agent = FakeAgent(llm=llm, stt=stt, tts=tts, instructions="be helpful " * 5)

    manifest = build_manifest_from_agent(agent)
    assert SECRET not in json.dumps(manifest)


def test_attribute_allow_lists_contain_no_credential_names():
    forbidden = {
        "key",
        "apikey",
        "token",
        "secret",
        "password",
        "credential",
        "credentials",
        "auth",
    }
    for attrs in (
        _MODEL_ATTRS,
        _VOICE_ATTRS,
        _LANGUAGE_ATTRS,
        _TEMPERATURE_ATTRS,
        _MAX_TOKENS_ATTRS,
    ):
        for candidate in attrs:
            segments = candidate.lower().replace(".", "_").split("_")
            assert not (set(segments) & forbidden), f"{candidate!r} looks credential-shaped"


# ── tools ────────────────────────────────────────────────────────────────


def _decorated_tool(name: str, description: str):
    def fn():
        pass

    fn.__livekit_tool_info = SimpleNamespace(name=name, description=description)
    return fn


def _raw_tool(schema: dict):
    def fn():
        pass

    fn.__livekit_raw_tool_info = SimpleNamespace(raw_schema=schema)
    return fn


def test_tools_extracted_from_agent():
    raw_schema = {
        "type": "function",
        "function": {
            "name": "book_slot",
            "description": "Book an appointment slot",
            "parameters": {"type": "object", "properties": {"slot": {"type": "string"}}},
        },
    }
    agent = FakeAgent(
        llm=make_llm(),
        tools=[_decorated_tool("lookup_order", "Find an order"), _raw_tool(raw_schema)],
    )
    manifest = build_manifest_from_agent(agent)
    tools = manifest["config"]["tools"]

    assert tools[0] == {"name": "lookup_order", "description": "Find an order"}
    assert tools[1]["name"] == "book_slot"
    assert tools[1]["schema"] == raw_schema["function"]["parameters"]


# ── override precedence ──────────────────────────────────────────────────


def test_behavior_override_beats_runtime_instructions():
    agent = FakeAgent(llm=make_llm(), instructions="runtime prompt " * 5)
    manifest = build_manifest_from_agent(agent, behavior={"prompt": "explicit override"})
    assert manifest["config"]["behavior"]["prompt"] == "explicit override"


def test_tools_override_beats_runtime_tools():
    agent = FakeAgent(tools=[_decorated_tool("runtime_tool", "from runtime")])
    manifest = build_manifest_from_agent(agent, tools=[{"name": "explicit_tool"}])
    assert manifest["config"]["tools"] == [{"name": "explicit_tool"}]


def test_extraction_failure_degrades_not_raises():
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    manifest = build_manifest_from_agent(Exploding())
    assert manifest["source"] == "livekit"


# ── server-schema fixtures ───────────────────────────────────────────────
#
# Mirrors ManifestConfig in the orchestration service's agent-sync.types.ts.
# If these fail, the server schema and the client TypedDicts have drifted —
# update both together.

CANONICAL_IDENTITY_FIELDS = {
    "name",
    "type",
    "agent_modality",
    "description",
    "pain_point",
    "gender",
    "age",
    "dob",
}
CANONICAL_BEHAVIOR_FIELDS = {"prompt", "flow"}
CANONICAL_TOOL_FIELDS = {"name", "description", "schema", "server"}
CANONICAL_LANGUAGE_FIELDS = {"primary_language", "additional_languages"}
CANONICAL_TELEPHONY_FIELDS = {"phone_number", "ivr_config"}


def test_typeddicts_match_canonical_schema():
    from livekit_evals.config_sync import (
        BehaviorConfig,
        IdentityConfig,
        LanguageConfig,
        TelephonyConfig,
        ToolConfig,
    )

    assert set(IdentityConfig.__annotations__) == CANONICAL_IDENTITY_FIELDS
    assert set(BehaviorConfig.__annotations__) == CANONICAL_BEHAVIOR_FIELDS
    assert set(ToolConfig.__annotations__) == CANONICAL_TOOL_FIELDS
    assert set(LanguageConfig.__annotations__) == CANONICAL_LANGUAGE_FIELDS
    assert set(TelephonyConfig.__annotations__) == CANONICAL_TELEPHONY_FIELDS


def test_unknown_override_keys_raise_locally():
    with pytest.raises(ValueError, match="unknown identity field"):
        build_manifest_from_agent(FakeAgent(), identity={"nickname": "Bob"})
    with pytest.raises(ValueError, match="unknown tools"):
        build_manifest_from_agent(FakeAgent(), tools=[{"name": "t", "endpoint": "x"}])
    with pytest.raises(ValueError, match="unknown telephony.ivr_config field"):
        build_manifest_from_agent(
            FakeAgent(), telephony={"phone_number": "+15550001111", "ivr_config": {"digits": "1"}}
        )
    with pytest.raises(TypeError):
        build_manifest_from_agent(FakeAgent(), identity="not-a-dict")
