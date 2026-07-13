"""Shared fakes for manifest-extraction tests.

Provider plugins are simulated by setting ``__module__`` on fake classes to
``livekit.plugins.<provider>.*`` — that is the only thing extraction keys on,
so tests don't need any real plugin packages installed.
"""

from __future__ import annotations

from types import SimpleNamespace


def make_plugin_class(name: str, module: str) -> type:
    cls = type(name, (), {})
    cls.__module__ = module
    return cls


def make_llm(provider: str = "openai", model: str = "gpt-4o-mini", **attrs):
    cls = make_plugin_class("FakeLLM", f"livekit.plugins.{provider}.llm")
    obj = cls()
    obj._opts = SimpleNamespace(model=model, temperature=0.4, max_tokens=512)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def make_stt(provider: str = "deepgram", model: str = "nova-3", language: str = "en", **attrs):
    cls = make_plugin_class("FakeSTT", f"livekit.plugins.{provider}.stt")
    obj = cls()
    obj._opts = SimpleNamespace(model=model, language=language)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def make_tts(
    provider: str = "elevenlabs",
    model: str = "eleven_turbo_v2",
    voice_id: str = "voice-abc",
    **attrs,
):
    cls = make_plugin_class("FakeTTS", f"livekit.plugins.{provider}.tts")
    obj = cls()
    obj._opts = SimpleNamespace(model=model, voice_id=voice_id)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


class FakeStreamAdapter:
    """Mimics livekit.agents.tts.StreamAdapter — module path is NOT a plugin."""

    def __init__(self, wrapped) -> None:
        self._wrapped_tts = wrapped


class FakeCustomWrapper:
    """Mimics customer wrappers like SanitizedTTS / NetworkGlitchTTS."""

    def __init__(self, inner) -> None:
        self._inner = inner


class FakeFallbackAdapter:
    """Mimics livekit.agents FallbackAdapter list-holding attribute."""

    def __init__(self, instances, attr: str) -> None:
        setattr(self, attr, list(instances))


class FakeAgent:
    def __init__(self, *, llm=None, stt=None, tts=None, instructions=None, tools=None) -> None:
        self.llm = llm
        self.stt = stt
        self.tts = tts
        if instructions is not None:
            self.instructions = instructions
        if tools is not None:
            self.tools = tools
