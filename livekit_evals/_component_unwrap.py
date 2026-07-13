"""
Cycle-safe unwrapping of TTS/STT/LLM wrapper chains for config sync.

Used by ``config_sync`` (manifest extraction). Mirrors the walker
``webhook_handler`` uses internally for metrics/provider reporting, but is
kept as a separate dependency-free module so ``config_sync`` stays
importable without the full LiveKit runtime.
"""

from __future__ import annotations

from typing import Any

# Attribute names commonly used by TTS/STT/LLM wrappers to reference the
# underlying instance.  Listed in priority order — the first attribute present
# on the wrapper wins.  ``_tts_instances`` / ``_stt_instances`` /
# ``_llm_instances`` are lists used by LiveKit's ``FallbackAdapter`` (we take
# the first entry as the "primary" base provider for reporting purposes).
_TTS_INNER_ATTRS: tuple[str, ...] = (
    "_tts_instances",  # livekit.agents.tts.FallbackAdapter
    "_wrapped_tts",  # livekit.agents.tts.StreamAdapter
    "_inner_tts",  # MixedAudioTTS, etc.
    "_inner",  # NetworkGlitchTTS, SanitizedTTS, VolumeTTS, etc.
    "_base_tts",
    "tts",
)
_STT_INNER_ATTRS: tuple[str, ...] = (
    "_stt_instances",  # livekit.agents.stt.FallbackAdapter
    "_wrapped_stt",  # livekit.agents.stt.StreamAdapter
    "_inner_stt",
    "_inner",
    "_base_stt",
    "stt",
)
_LLM_INNER_ATTRS: tuple[str, ...] = (
    "_llm_instances",  # livekit.agents.llm.FallbackAdapter
    "_inner_llm",
    "_inner",
    "_base_llm",
    "llm",
)

_INNER_ATTRS_BY_ROLE: dict[str, tuple[str, ...]] = {
    "llm": _LLM_INNER_ATTRS,
    "stt": _STT_INNER_ATTRS,
    "tts": _TTS_INNER_ATTRS,
}


def _unwrap_to_base_component(component: Any, inner_attrs: tuple[str, ...]) -> Any:
    """Recursively descend through TTS/STT/LLM wrappers to find the base provider.

    Some agents wrap their TTS/STT/LLM with adapters (FallbackAdapter, StreamAdapter)
    or custom behavioural wrappers (network-glitch, mixed-audio, volume, sanitiser,
    etc.).  The module path of the wrapper is not the actual provider, so we walk
    down ``inner_attrs`` until we either:

      * reach a component whose module path is ``livekit.plugins.<provider>.*``
        (a real provider plugin), or
      * run out of wrapper attributes to follow.

    For list-valued attributes (``_tts_instances`` on ``FallbackAdapter``), the
    first instance is used as the representative base provider.
    """
    if component is None:
        return None

    visited: set[int] = set()
    current = component
    while current is not None and id(current) not in visited:
        visited.add(id(current))

        module_path = getattr(current, "__module__", "") or ""
        if module_path.startswith("livekit.plugins."):
            return current

        next_inner: Any = None
        for attr in inner_attrs:
            if not hasattr(current, attr):
                continue
            candidate = getattr(current, attr)
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple)):
                if not candidate:
                    continue
                candidate = candidate[0]
            if candidate is current:
                continue
            next_inner = candidate
            break

        if next_inner is None:
            return current
        current = next_inner

    return current


def _fallback_instances(component: Any, inner_attrs: tuple[str, ...]) -> list[Any]:
    """Non-primary instances of a ``FallbackAdapter`` (empty for other wrappers).

    ``FallbackAdapter`` keeps its candidates in a list attribute
    (``_tts_instances`` / ``_stt_instances`` / ``_llm_instances``). The first
    entry is the primary; the rest are fallbacks. Walks the wrapper chain the
    same way as :func:`_unwrap_to_base_component`, so a ``FallbackAdapter``
    nested inside a custom wrapper is still found.
    """
    visited: set[int] = set()
    current = component
    while current is not None and id(current) not in visited:
        visited.add(id(current))

        next_inner: Any = None
        for attr in inner_attrs:
            if not hasattr(current, attr):
                continue
            candidate = getattr(current, attr)
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple)):
                if len(candidate) > 1:
                    return list(candidate[1:])
                if not candidate:
                    continue
                candidate = candidate[0]
            if candidate is current:
                continue
            next_inner = candidate
            break

        current = next_inner

    return []
