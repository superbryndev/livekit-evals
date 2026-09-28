"""
Webhook Handler for LiveKit Agent

Captures events during agent session and sends webhook payload to Supabase edge function.
Designed to work with the webhooks-livekit edge function.
"""

import asyncio
import contextlib
import logging
import os
import platform
import re
import json
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Optional

import aiohttp
from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    metrics,
)
from livekit.agents.voice.events import AgentState, UserState
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import (
    LLMMetrics,
    STTMetrics,
    TTSMetrics,
    # VADMetrics,
    # EOUMetrics,
    RealtimeModelMetrics,
)
from livekit.rtc import Room

try:
    from livekit.rtc import ConnectionQuality, DisconnectReason
except ImportError:  # older livekit-rtc may not export these
    ConnectionQuality = None
    DisconnectReason = None

from .config import CREDENTIALS_CONFIG, WEBHOOK_CONFIG, LIVEKIT_CONFIG, AGENT_CONFIG
from .recording_manager import RecordingManager

logger = logging.getLogger("webhook_handler")

# Extended-capture payload bounds — keep the webhook body small on long calls
_MAX_EVENT_LIST = 50
_MAX_EOU_EVENTS = 200
_MAX_DTMF_EVENTS = 100
_MAX_ERROR_EVENTS = 50
_MAX_RTC_SAMPLES = 90
_RTC_POLL_INTERVAL_S = 10.0
_ERROR_MSG_MAX_LEN = 500

_QUALITY_RANK = {"excellent": 0, "good": 1, "poor": 2, "lost": 3}


def _enum_name(enum_cls: Any, value: Any) -> str:
    """Readable name for an enum value ("QUALITY_POOR" -> "poor").

    Handles both shapes used across livekit-rtc versions: Python IntEnum
    members (``value.name``) and proto enum wrappers (``EnumCls.Name(value)``).
    """
    try:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            name = enum_cls.Name(value) if enum_cls is not None else str(value)
    except Exception:  # noqa: BLE001
        name = str(value)
    for prefix in ("QUALITY_", "DISCONNECT_REASON_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.lower()


def _mask_api_key(api_key: Optional[str]) -> str:
    """Return a redacted representation of an API key safe for logging."""
    if not api_key:
        return "<not-set>"
    key_len = len(api_key)
    if key_len <= 8:
        return "*" * key_len
    return f"{api_key[:4]}...{api_key[-4:]} (len={key_len})"


# Attribute names commonly used by TTS/STT/LLM wrappers to reference the
# underlying instance.  Listed in priority order — the first attribute present
# on the wrapper wins.  ``_tts_instances`` / ``_stt_instances`` /
# ``_llm_instances`` are lists used by LiveKit's ``FallbackAdapter`` (we take
# the first entry as the "primary" base provider for reporting purposes).
_TTS_INNER_ATTRS: tuple[str, ...] = (
    "_tts_instances",   # livekit.agents.tts.FallbackAdapter
    "_wrapped_tts",     # livekit.agents.tts.StreamAdapter
    "_inner_tts",       # MixedAudioTTS, etc.
    "_inner",           # NetworkGlitchTTS, SanitizedTTS, VolumeTTS, etc.
    "_base_tts",
    "tts",
)
_STT_INNER_ATTRS: tuple[str, ...] = (
    "_stt_instances",   # livekit.agents.stt.FallbackAdapter
    "_wrapped_stt",     # livekit.agents.stt.StreamAdapter
    "_inner_stt",
    "_inner",
    "_base_stt",
    "stt",
)
_LLM_INNER_ATTRS: tuple[str, ...] = (
    "_llm_instances",   # livekit.agents.llm.FallbackAdapter
    "_inner_llm",
    "_inner",
    "_base_llm",
    "llm",
)


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


def _extract_provider_from_module(component: Any) -> Optional[str]:
    """Extract the provider name from a component's module path.

    For ``livekit.plugins.<provider>.<module>`` we return ``<provider>``;
    otherwise we fall back to the last segment of the module path.
    """
    if component is None:
        return None
    module_path = getattr(component, "__module__", "") or ""
    if not module_path:
        return None
    parts = module_path.split(".")
    if len(parts) >= 3 and parts[0] == "livekit" and parts[1] == "plugins":
        return parts[2]
    return parts[-1] if parts else None


def _extract_project_id_from_url(livekit_url: str) -> Optional[str]:
    """
    Extract project ID from LiveKit URL.

    Args:
        livekit_url: LiveKit URL (e.g., "wss://tara-agent-bt2d90rn.livekit.cloud")

    Returns:
        Project ID (e.g., "tara-agent-bt2d90rn") or None if not found
    """
    # Match pattern: wss://PROJECT_ID.livekit.cloud or ws://PROJECT_ID.livekit.cloud
    match = re.match(r"wss?://([^.]+)\.livekit\.cloud", livekit_url)
    if match:
        return match.group(1)
    
    # Also handle custom domains or localhost
    # For custom domains, just use the full hostname as project ID
    match = re.match(r"wss?://([^:/]+)", livekit_url)
    if match:
        hostname = match.group(1)
        # Remove 'livekit.cloud' suffix if present
        if hostname.endswith(".livekit.cloud"):
            return hostname.replace(".livekit.cloud", "")
        return hostname
    
    return None


class WebhookHandler:
    """
    Handles webhook payload construction and delivery for LiveKit agent sessions.
    
    Listens to session events and aggregates data to send to the webhook endpoint
    when the session closes.
    """
    
    def __init__(
        self,
        webhook_url: str,
        api_key: str,
        room: Room,
        is_deployed_on_lk_cloud: bool,
        livekit_project_id: Optional[str] = None,
        call_rate_usd: Optional[float] = None,
        recording_manager: Optional["RecordingManager"] = None,
        disable_recording: bool = False,
        stereo_recording: bool = False,
        defer_recording: bool = False,
        custom_data: Optional[dict[str, Any]] = None,
        extended_capture: bool = True,
    ):
        """
        Initialize webhook handler.
        
        Args:
            webhook_url: URL of the webhook endpoint
            api_key: API key for the webhook endpoint
            room: LiveKit room instance
            is_deployed_on_lk_cloud: Whether agent is deployed on LiveKit Cloud ($0.014/min)
            livekit_project_id: LiveKit project ID for agent uniqueness (optional)
            call_rate_usd: Custom telephony rate per minute ($/min) for cost calculation (optional)
            recording_manager: RecordingManager instance for handling recordings (optional)
            disable_recording: Set to True to disable call recording (default: False, recording enabled)
            stereo_recording: If True, record in dual-channel stereo (L=agent, R=others).
                Implies recording is enabled (overrides disable_recording).
            defer_recording: If True, recording does NOT start automatically in
                ``attach_to_session``.  The caller must invoke ``start_recording()``
                explicitly (e.g. when the remote participant connects).
            custom_data: Arbitrary JSON-serializable dict that is forwarded as-is
                in the ``call.custom_data`` field of every webhook payload.  Use
                ``update_custom_data()`` / ``set_custom_data()`` at any point
                during the session to change it before the webhook fires.
            extended_capture: When True (default), also capture turn-detection
                (EOU) latency, VAD/interruption analytics, provider errors,
                network quality + WebRTC stats, SIP attributes and DTMF, and
                emit them as additive payload sections. Set False for the
                legacy payload shape.
        """
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.room = room
        self.is_deployed_on_lk_cloud = is_deployed_on_lk_cloud
        self.livekit_project_id = livekit_project_id
        self.call_rate_usd = call_rate_usd
        self.recording_manager = recording_manager
        self.disable_recording = disable_recording
        self.stereo_recording = stereo_recording
        self.defer_recording = defer_recording
        self.custom_data: dict[str, Any] = dict(custom_data) if custom_data else {}
        
        # These will be auto-detected
        self.agent_id: Optional[str] = None
        self.version_id: Optional[str] = None
        self.system_prompt: Optional[str] = None
        self.sip_trunking_enabled: bool = False
        self.egress_enabled: bool = False
        self.phone_number: Optional[str] = None
        
        # Recording URLs (set externally by RecordingManager)
        self.recording_url: Optional[str] = None
        self.stereo_recording_url: Optional[str] = None
        self.egress_id: Optional[str] = None

        # "superbryn_s3" (managed) | "external" (dev-supplied); reachable = probe result
        self.recording_url_source: Optional[str] = None
        self.recording_url_reachable: Optional[bool] = None
        
        # Session tracking
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        self.ring_started_at: Optional[datetime] = None
        self.call_end_reason: Optional[str] = None
        self._preferred_call_end_reason: Optional[str] = None
        
        # Transcript tracking
        self.transcript_turns: list[dict[str, Any]] = []
        self.call_start_time_ms: Optional[int] = None
        self.last_user_turn_time_ms: Optional[int] = None

        # Tool call tracking
        self.tool_calls: list[dict[str, Any]] = []

        # Agent handoff tracking (non-message conversation items)
        self.agent_handoffs: list[dict[str, Any]] = []

        # Usage metrics tracking
        self.usage_metrics = {
            "llm_model": None,
            "llm_provider": None,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_total_tokens": 0,
            "stt_provider": None,
            "stt_model": None,
            "stt_duration_seconds": 0,
            "audio_duration_seconds": 0,
            "tts_provider": None,
            "tts_model": None,
            "tts_voice_id": None,
            "tts_characters": 0,
            "tts_audio_duration_seconds": 0,
        }
        
        # Latency tracking (aggregated)
        self.latency_metrics = {
            "llm_ms": [],
            "stt_ms": [],
            "tts_ms": [],
        }
        
        # Latest cumulative usage from the (non-deprecated) session_usage_updated
        # event; used to backfill usage when the per-plugin metrics path can't
        # observe it (notably realtime models). See _backfill_usage_from_session.
        self._latest_session_usage: Any = None

        # Speech events tracking
        self.speech_events: list[dict[str, Any]] = []

        # Defensive base-component metrics subscriptions.
        # Custom user wrappers (e.g. ``SanitizedTTS``, ``VolumeTTS``,
        # ``MixedAudioTTS``, ``NetworkGlitchTTS``) frequently forget to forward
        # the inner plugin's ``metrics_collected`` event, which causes
        # ``tts_characters`` / ``tts_audio_duration_seconds`` / TTS latency to
        # silently report zero.  ``attach_to_session`` subscribes directly to
        # the base TTS/STT/LLM components as a safety net, and these structures
        # let us dedup re-emitted events and unsubscribe on session close.
        self._base_metric_subscriptions: list[tuple[Any, Any]] = []
        self._seen_metrics: set[tuple[Optional[str], Optional[str], Optional[str]]] = set()

        # Extended capture (additive payload sections; see _build_extended_sections)
        self.extended_capture = extended_capture
        self.eou_events: list[dict[str, Any]] = []
        self.eou_delays_ms: list[float] = []
        self.transcription_delays_ms: list[float] = []
        self.eot_stats = {"inference_count": 0, "prediction_total_s": 0.0}
        self.vad_stats = {"inference_count": 0, "inference_total_s": 0.0, "idle_time_s": 0.0}
        self.interruption_stats: dict[str, Any] = {}
        self.false_interruption_count = 0
        self.error_events: list[dict[str, Any]] = []
        self.lk_close_info: dict[str, Any] = {}
        self.dtmf_events: list[dict[str, Any]] = []
        self.sip_attributes: dict[str, str] = {}
        self.connection_quality_events: list[dict[str, Any]] = []
        self.connection_events: list[dict[str, Any]] = []
        self.track_subscription_failures = 0
        self.room_disconnect_reason: Optional[str] = None
        self._worst_quality: Optional[str] = None
        self.extra_usage: dict[str, Any] = {
            "llm_prompt_cached_tokens": 0,
            "llm_tokens_per_second": [],
            "stt_acquire_time_ms": [],
            "tts_acquire_time_ms": [],
            "realtime_input_text_tokens": 0,
            "realtime_input_audio_tokens": 0,
            "realtime_input_cached_tokens": 0,
            "realtime_output_text_tokens": 0,
            "realtime_output_audio_tokens": 0,
        }
        self._rtc_samples: list[dict[str, Any]] = []
        self._rtc_agg: dict[str, Any] = {}
        self._rtc_poll_task: Optional[asyncio.Task] = None
        self._room_subscriptions: list[tuple[str, Any]] = []

        logger.info(
            "WebhookHandler initialized: is_deployed_on_lk_cloud=%s, livekit_project_id=%s (agent_id and version_id will be auto-detected)",
            is_deployed_on_lk_cloud,
            livekit_project_id or "not-set",
        )

        logger.info(
            "SUPERBRYN_CONFIG_LOADED: webhook_url=%s | api_key=%s | "
            "credentials_url=%s | base_url=%s | livekit_project_id=%s | "
            "is_deployed_on_lk_cloud=%s | call_rate_usd=%s | "
            "disable_recording=%s | stereo_recording=%s | defer_recording=%s | "
            "recording_manager=%s | agent_id_default=%s | version_id_default=%s",
            self.webhook_url,
            _mask_api_key(self.api_key),
            CREDENTIALS_CONFIG.get("url"),
            os.getenv("SUPERBRYN_BASE_URL", "https://api.superbryn.com"),
            self.livekit_project_id or "not-set",
            self.is_deployed_on_lk_cloud,
            self.call_rate_usd,
            self.disable_recording,
            self.stereo_recording,
            self.defer_recording,
            "enabled" if self.recording_manager else "disabled",
            AGENT_CONFIG.get("id"),
            AGENT_CONFIG.get("version_id"),
        )
    
    def set_recording_url(
        self,
        recording_url: Optional[str],
        egress_id: Optional[str] = None,
        stereo_recording_url: Optional[str] = None,
    ) -> None:
        """
        Set the recording URL for this session.
        
        Args:
            recording_url: Primary recording URL
            egress_id: LiveKit egress ID for the recording
            stereo_recording_url: Stereo recording URL (optional)
        """
        self.recording_url = recording_url
        self.egress_id = egress_id
        self.stereo_recording_url = stereo_recording_url
        
        if recording_url:
            self.egress_enabled = True
            logger.info("Recording URL set: %s (egress_id: %s)", recording_url, egress_id)

    def set_call_end_reason(self, reason: Optional[str]) -> None:
        """Override the reason sent in the final webhook payload."""
        if reason and reason.strip():
            self._preferred_call_end_reason = reason.strip()
            self.call_end_reason = reason.strip()

    def set_custom_data(self, data: dict[str, Any]) -> None:
        """Replace the ``call.custom_data`` field in the webhook payload entirely.

        Call this at any point before the session ends to attach arbitrary
        JSON-serializable data to the outgoing webhook.  The value is forwarded
        verbatim in ``payload["call"]["custom_data"]``.

        Example::

            webhook_handler.set_custom_data({
                "ticket_id": "TKT-9001",
                "customer_tier": "enterprise",
                "resolved": True,
            })
        """
        self.custom_data = dict(data)

    def update_custom_data(self, data: dict[str, Any]) -> None:
        """Shallow-merge *data* into the existing ``call.custom_data`` dict.

        Existing keys not present in *data* are preserved.  Use this to
        incrementally enrich the payload as events unfold during the call
        (e.g. after a tool call resolves or an intent is detected).

        Example::

            # Set initial context at session start
            webhook_handler.update_custom_data({"lead_source": "website"})

            # Later, after a tool call:
            webhook_handler.update_custom_data({"appointment_booked": True, "slot": "2026-06-05T10:00"})
        """
        self.custom_data.update(data)

    async def start_recording(self) -> None:
        """Start recording the call.

        When ``defer_recording=True`` was passed to the factory, this must be
        called explicitly (e.g. when the remote participant connects).
        When ``defer_recording=False`` (default), recording starts
        automatically inside ``attach_to_session`` and calling this is a no-op.
        """
        if self.recording_url:
            return

        if self.disable_recording and not self.stereo_recording:
            return

        if not self.recording_manager:
            return

        mode = "stereo" if self.stereo_recording else "mono"
        logger.info("Starting %s recording for room %s", mode, self.room.name)
        recording_url, egress_id = await self.recording_manager.start_recording(
            room_name=self.room.name,
            phone_number=self.phone_number,
        )
        if recording_url:
            self.set_recording_url(
                recording_url=recording_url,
                egress_id=egress_id,
                stereo_recording_url=recording_url if self.stereo_recording else None,
            )
            # Managed egress writes to SuperBryn's bucket — no mirroring needed
            self.recording_url_source = "superbryn_s3"
            self.recording_url_reachable = True
            logger.info("Recording started successfully (%s): %s", mode, recording_url)
        else:
            logger.warning("Failed to start recording")

    async def stop_egress(self) -> None:
        """Stop the active egress so the recording file is finalised on S3.

        Call this **before** deleting the room. Safe to call even if no
        recording is active (no-op in that case).
        """
        if self.recording_manager:
            await self.recording_manager.stop_egress()

    async def set_external_recording_url(
        self,
        recording_url: str,
        *,
        probe: bool = True,
    ) -> bool:
        """Attach a recording URL produced by the developer's own egress.

        Use when you run your own egress (typically with ``disable_recording=True``)
        and want the URL forwarded in the webhook. Call it once the recording
        exists, any time before ``send_webhook`` fires on shutdown.

        Only public or pre-signed URLs are supported. When ``probe=True`` the URL
        is validated at call time with a ranged GET (200/206 reachable, 401/403
        private, 404 not-yet/wrong-path). Reachability is checked *at call time*,
        so sign pre-signed URLs for a long-enough TTL if mirroring happens later.

        Returns True if probed and reachable, else False (the URL is stored
        either way; ``probe=False`` skips the check and returns False).
        """
        if not recording_url or not recording_url.strip():
            logger.warning("set_external_recording_url called with empty URL — ignored")
            return False

        recording_url = recording_url.strip()

        reachable: Optional[bool] = None
        if probe:
            reachable, detail = await self._probe_recording_url(recording_url)
            if reachable:
                logger.info(
                    "SUPERBRYN_EXTERNAL_RECORDING_URL_OK: %s (%s)",
                    recording_url,
                    detail,
                )
            else:
                logger.error(
                    "SUPERBRYN_EXTERNAL_RECORDING_URL_UNREACHABLE: %s — %s. "
                    "Only public or pre-signed URLs are supported; mirroring will be skipped.",
                    recording_url,
                    detail,
                )

        self.set_recording_url(recording_url=recording_url)
        self.recording_url_source = "external"
        self.recording_url_reachable = reachable
        return bool(reachable)

    async def _probe_recording_url(self, url: str) -> tuple[bool, str]:
        """Probe *url* with a ranged GET (``bytes=0-0``); return (reachable, detail)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"Range": "bytes=0-0"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    status = resp.status
                    if status in (200, 206):
                        return True, f"reachable (status={status})"
                    if status in (401, 403):
                        return False, (
                            f"not publicly fetchable (status={status}) — private "
                            "object or bad/expired signature"
                        )
                    if status == 404:
                        return False, (
                            f"not found (status={status}) — object may not be "
                            "uploaded yet or the path is wrong"
                        )
                    return False, f"unexpected status={status}"
        except asyncio.TimeoutError:
            return False, "probe timed out after 10s"
        except Exception as e:  # noqa: BLE001
            return False, f"probe failed: {e}"

    def _extract_session_config(self, session: AgentSession) -> None:
        """Extract model/provider info from session configuration using Whispey's approach.

        Components (TTS in particular) may be wrapped by adapters
        (``FallbackAdapter``, ``StreamAdapter``) or custom behavioural wrappers
        (network-glitch, mixed-audio, volume, sanitiser, etc.).  In those cases
        the wrapper's module path is not the real provider name, so we descend
        through known inner-instance attributes to find the underlying plugin
        (e.g. ``cartesia``, ``sarvam``) before reading model/provider info.
        """
        try:
            if hasattr(session, 'llm') and session.llm:
                wrapper_llm = session.llm
                llm_obj = _unwrap_to_base_component(wrapper_llm, _LLM_INNER_ATTRS) or wrapper_llm
                if llm_obj is not wrapper_llm:
                    logger.info(
                        "Unwrapped LLM: %s -> %s",
                        getattr(wrapper_llm, "__module__", type(wrapper_llm).__name__),
                        getattr(llm_obj, "__module__", type(llm_obj).__name__),
                    )

                if hasattr(llm_obj, 'model'):
                    self.usage_metrics["llm_model"] = llm_obj.model

                provider_name = _extract_provider_from_module(llm_obj)
                if provider_name:
                    self.usage_metrics["llm_provider"] = provider_name

                if hasattr(llm_obj, '_opts') and llm_obj._opts:
                    opts = llm_obj._opts
                    if hasattr(opts, 'model') and not self.usage_metrics["llm_model"]:
                        self.usage_metrics["llm_model"] = opts.model

            if hasattr(session, 'stt') and session.stt:
                wrapper_stt = session.stt
                stt_obj = _unwrap_to_base_component(wrapper_stt, _STT_INNER_ATTRS) or wrapper_stt
                if stt_obj is not wrapper_stt:
                    logger.info(
                        "Unwrapped STT: %s -> %s",
                        getattr(wrapper_stt, "__module__", type(wrapper_stt).__name__),
                        getattr(stt_obj, "__module__", type(stt_obj).__name__),
                    )

                if hasattr(stt_obj, 'model'):
                    self.usage_metrics["stt_model"] = stt_obj.model

                provider_name = _extract_provider_from_module(stt_obj)
                if provider_name:
                    self.usage_metrics["stt_provider"] = provider_name

                if hasattr(stt_obj, '_opts') and stt_obj._opts:
                    opts = stt_obj._opts
                    if hasattr(opts, 'model') and not self.usage_metrics["stt_model"]:
                        self.usage_metrics["stt_model"] = opts.model

                    # Speechmatics uses operating_point instead of model
                    if hasattr(opts, 'operating_point') and not self.usage_metrics["stt_model"]:
                        self.usage_metrics["stt_model"] = opts.operating_point

            if hasattr(session, 'tts') and session.tts:
                wrapper_tts = session.tts
                tts_obj = _unwrap_to_base_component(wrapper_tts, _TTS_INNER_ATTRS) or wrapper_tts
                if tts_obj is not wrapper_tts:
                    logger.info(
                        "Unwrapped TTS: %s -> %s",
                        getattr(wrapper_tts, "__module__", type(wrapper_tts).__name__),
                        getattr(tts_obj, "__module__", type(tts_obj).__name__),
                    )

                if hasattr(tts_obj, 'voice_id'):
                    self.usage_metrics["tts_voice_id"] = tts_obj.voice_id
                elif hasattr(tts_obj, 'voice'):
                    self.usage_metrics["tts_voice_id"] = tts_obj.voice

                if hasattr(tts_obj, 'model'):
                    self.usage_metrics["tts_model"] = tts_obj.model

                provider_name = _extract_provider_from_module(tts_obj)
                if provider_name:
                    self.usage_metrics["tts_provider"] = provider_name

                if hasattr(tts_obj, '_opts') and tts_obj._opts:
                    opts = tts_obj._opts
                    if hasattr(opts, 'voice_id') and not self.usage_metrics.get("tts_voice_id"):
                        self.usage_metrics["tts_voice_id"] = opts.voice_id
                    elif hasattr(opts, 'voice') and not self.usage_metrics.get("tts_voice_id"):
                        self.usage_metrics["tts_voice_id"] = opts.voice
                    if hasattr(opts, 'model') and not self.usage_metrics["tts_model"]:
                        self.usage_metrics["tts_model"] = opts.model

                # Fallback: if no voice_id found but model exists, use model as voice_id.
                # This is common for providers like Sarvam where model name IS the voice.
                if not self.usage_metrics["tts_voice_id"] and self.usage_metrics["tts_model"]:
                    self.usage_metrics["tts_voice_id"] = self.usage_metrics["tts_model"]
            
            # Apply provider detection based on model names
            if self.usage_metrics["llm_model"]:
                detected_provider = self._detect_provider_from_model_name(self.usage_metrics["llm_model"])
                if detected_provider != 'unknown':
                    self.usage_metrics["llm_provider"] = detected_provider
            
            if self.usage_metrics["stt_model"]:
                detected_provider = self._detect_provider_from_model_name(self.usage_metrics["stt_model"])
                if detected_provider != 'unknown':
                    self.usage_metrics["stt_provider"] = detected_provider
            
            if self.usage_metrics["tts_model"] or self.usage_metrics.get("tts_voice_id"):
                model_or_voice = self.usage_metrics["tts_model"] or self.usage_metrics.get("tts_voice_id")
                detected_provider = self._detect_provider_from_model_name(model_or_voice)
                if detected_provider != 'unknown':
                    self.usage_metrics["tts_provider"] = detected_provider
            
            logger.info("Extracted session config: %s", self.usage_metrics)
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to extract session config: %s", e)
    
    def _detect_provider_from_model_name(self, model_name: str) -> str:
        """Detect provider from model name with comprehensive provider support."""
        if not model_name:
            return 'unknown'
        
        model_lower = model_name.lower()
        
        # LLM Providers
        if any(x in model_lower for x in ['gpt', 'openai', 'whisper', 'tts-1', 'o1-', 'o3-']):
            return 'openai'
        elif any(x in model_lower for x in ['claude', 'anthropic']):
            return 'anthropic'
        elif any(x in model_lower for x in ['gemini', 'palm', 'bard', 'gemma']):
            return 'google'
        elif any(x in model_lower for x in ['llama', 'meta-llama']):
            return 'meta'
        elif any(x in model_lower for x in ['mistral', 'mixtral']):
            return 'mistral'
        elif any(x in model_lower for x in ['cohere', 'command']):
            return 'cohere'
        elif any(x in model_lower for x in ['perplexity', 'pplx']):
            return 'perplexity'
        elif any(x in model_lower for x in ['groq']):
            return 'groq'
        elif any(x in model_lower for x in ['together', 'togethercomputer']):
            return 'together'
        elif any(x in model_lower for x in ['replicate']):
            return 'replicate'
        elif any(x in model_lower for x in ['huggingface', 'hf-']):
            return 'huggingface'
        
        # TTS Providers
        elif any(x in model_lower for x in ['eleven', 'elevenlabs']):
            return 'elevenlabs'
        elif any(x in model_lower for x in ['cartesia', 'sonic']):
            return 'cartesia'
        elif any(x in model_lower for x in ['playht', 'play.ht', 'play-ht']):
            return 'playht'
        elif any(x in model_lower for x in ['resemble', 'resembleai']):
            return 'resemble'
        elif any(x in model_lower for x in ['murf', 'murf.ai']):
            return 'murf'
        elif any(x in model_lower for x in ['wellsaid', 'wellsaidlabs']):
            return 'wellsaid'
        elif any(x in model_lower for x in ['speechify']):
            return 'speechify'
        elif any(x in model_lower for x in ['saarika', 'sarvam', 'bulbul']):
            return 'sarvam'
        elif any(x in model_lower for x in ['azure', 'microsoft']):
            return 'azure'
        elif any(x in model_lower for x in ['aws', 'polly', 'amazon']):
            return 'aws'
        elif any(x in model_lower for x in ['gcloud', 'google-cloud']):
            return 'google-cloud'
        
        # STT Providers
        elif any(x in model_lower for x in ['deepgram', 'nova', 'aura']):
            return 'deepgram'
        elif any(x in model_lower for x in ['assemblyai', 'assembly']):
            return 'assemblyai'
        elif any(x in model_lower for x in ['rev.ai', 'revai']):
            return 'rev'
        elif any(x in model_lower for x in ['speechmatics']):
            return 'speechmatics'
        elif any(x in model_lower for x in ['gladia']):
            return 'gladia'
        
        # Multi-modal/Realtime Providers
        elif any(x in model_lower for x in ['livekit']):
            return 'livekit'
        elif any(x in model_lower for x in ['twilio']):
            return 'twilio'
        elif any(x in model_lower for x in ['vonage']):
            return 'vonage'
        
        else:
            return 'unknown'
    
    def _detect_sip_trunking(self) -> None:
        """Detect if SIP trunking is enabled by checking for SIP participants and extract phone number."""
        try:
            for participant in self.room.remote_participants.values():
                # Check if participant has SIP-related attributes
                if hasattr(participant, 'attributes'):
                    attributes = participant.attributes
                    # Check for any sip.* attributes
                    if any(key.startswith('sip.') for key in attributes.keys()):
                        self.sip_trunking_enabled = True
                        logger.info("SIP trunking detected from participant attributes")
                        
                        # Extract phone number from sip.phoneNumber attribute if available
                        if 'sip.phoneNumber' in attributes and not self.phone_number:
                            self.phone_number = attributes['sip.phoneNumber']
                            logger.info("Extracted phone number from SIP attributes: %s", self.phone_number)
                        
                        return
                
                # Also check participant identity for SIP patterns
                if hasattr(participant, 'identity'):
                    identity = participant.identity
                    # SIP participants often have phone number-like identities
                    if identity and (identity.startswith('+') or identity.startswith('sip:')):
                        self.sip_trunking_enabled = True
                        logger.info("SIP trunking detected from participant identity: %s", identity)
                        
                        # Extract phone number from identity if not already set
                        if not self.phone_number and identity.startswith('+'):
                            self.phone_number = identity
                            logger.info("Extracted phone number from participant identity: %s", self.phone_number)
                        
                        return
            
            logger.info("No SIP participants detected, SIP trunking disabled")
        except Exception as e:
            logger.warning("Failed to detect SIP trunking: %s", e)
    
    async def attach_to_session(self, session: AgentSession) -> None:
        """
        Attach event listeners to the agent session and start recording if configured.
        
        Args:
            session: AgentSession to attach to
        """
        # Mark session start
        self.started_at = datetime.now(timezone.utc)
        self.call_start_time_ms = int(self.started_at.timestamp() * 1000)
        
        # Capture room creation time as ring_started_at (when the call was initiated)
        try:
            self.ring_started_at = self.room.creation_time
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to get room.creation_time: %s", e)
        
        # agent_id / version_id come from AGENT_CONFIG (env AGENT_ID / VERSION_ID,
        # with defaults). phone_number, if any, is detected from SIP participant
        # attributes below.
        self.agent_id = AGENT_CONFIG['id']
        self.version_id = AGENT_CONFIG['version_id']

        # Ensure agent_id is never None or empty (required by webhook endpoint)
        if not self.agent_id:
            self.agent_id = "livekit-agent-default"
            logger.warning("agent_id not configured - using fallback: %s", self.agent_id)
        
        # Start recording unless deferred or disabled
        if self.defer_recording:
            logger.info("Recording deferred — call start_recording() explicitly")
        else:
            try:
                await self.start_recording()
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to start recording: %s", e, exc_info=True)
        
        # Extract system prompt from session's current agent
        if hasattr(session, 'current_agent') and session.current_agent:
            current_agent = session.current_agent
            if hasattr(current_agent, 'instructions'):
                self.system_prompt = current_agent.instructions
                logger.info("Extracted system prompt from session.current_agent: %s...",
                          self.system_prompt[:100] if self.system_prompt else "None")
        
        # Detect SIP trunking by checking for SIP participants
        self._detect_sip_trunking()
        
        # Extract model/provider info from session configuration
        self._extract_session_config(session)
        
        # Listen to conversation items for transcript
        session.on("conversation_item_added")(self._on_conversation_item_added)
        
        # Metrics: subscribe on the per-plugin metrics_collected surface (the
        # session-level metrics_collected event is deprecated). See
        # _subscribe_to_metrics for the outer+base subscription strategy.
        self._subscribe_to_metrics(session)

        # Realtime models emit metrics on an internal session we can't reach, so
        # also track cumulative usage via the (non-deprecated) session_usage_updated
        # event and use it to backfill usage at payload time.
        session.on("session_usage_updated")(self._on_session_usage_updated)
        
        # Listen to user input for additional transcript metadata
        session.on("user_input_transcribed")(self._on_user_input_transcribed)
        
        # Listen to state changes for precise timing
        session.on("agent_state_changed")(self._on_agent_state_changed)
        session.on("user_state_changed")(self._on_user_state_changed)
        
        # Listen to speech creation for additional tracking
        session.on("speech_created")(self._on_speech_created)

        # Listen to tool/function call execution
        session.on("function_tools_executed")(self._on_function_tools_executed)

        # Listen to session close for cleanup
        session.on("close")(self._on_session_close)

        if self.extended_capture:
            self._attach_extended_listeners(session)

        logger.info("Event listeners attached to session")

    def _attach_extended_listeners(self, session: AgentSession) -> None:
        """Subscribe to the extended session/room surfaces, feature-detecting each.

        Every subscription is wrapped individually: an event name that doesn't
        exist on an older SDK must never take down the core capture path.
        """
        # The session-level metrics_collected re-emit is the only surface that
        # carries VAD/EOU/EOT/interruption metrics; provider metrics arriving
        # here as duplicates are already handled by _seen_metrics dedup.
        session_events = [
            ("metrics_collected", self._on_metrics_collected),
            ("error", self._on_error_event),
            ("agent_false_interruption", self._on_false_interruption),
        ]
        for event_name, handler in session_events:
            try:
                session.on(event_name)(handler)
            except Exception as e:  # noqa: BLE001
                logger.debug("Extended session event %s unavailable: %s", event_name, e)

        room_events = [
            ("connection_quality_changed", self._on_connection_quality_changed),
            ("reconnecting", self._on_room_reconnecting),
            ("reconnected", self._on_room_reconnected),
            ("disconnected", self._on_room_disconnected),
            ("participant_disconnected", self._on_participant_disconnected),
            ("track_subscription_failed", self._on_track_subscription_failed),
            ("sip_dtmf_received", self._on_sip_dtmf),
            ("participant_attributes_changed", self._on_participant_attributes_changed),
        ]
        for event_name, handler in room_events:
            try:
                self.room.on(event_name, handler)
                self._room_subscriptions.append((event_name, handler))
            except Exception as e:  # noqa: BLE001
                logger.debug("Extended room event %s unavailable: %s", event_name, e)

        if hasattr(self.room, "get_rtc_stats"):
            self._rtc_poll_task = asyncio.create_task(self._rtc_stats_poll_loop())
        else:
            logger.info("room.get_rtc_stats unavailable — skipping network stats polling")

        logger.info("Extended capture listeners attached (room events + metrics + rtc stats)")
    
    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        """Handle conversation item added event - fills in text for existing turns.

        The conversation stream can carry non-message items (e.g. ``AgentHandoff``
        emitted on receptionist/agent handoffs), which have no ``role`` or
        ``text_content``. Only ``ChatMessage`` items hold transcript text, so
        gate on that (matching LiveKit's own handlers) and record handoffs
        separately for observability instead of crashing on ``item.role``.
        """
        item = event.item

        if not isinstance(item, ChatMessage):
            item_type = getattr(item, "type", type(item).__name__)
            if item_type == "agent_handoff":
                self._track_agent_handoff(item)
            else:
                logger.info("Skipping non-message conversation item: type=%s", item_type)
            return

        # Determine speaker role
        speaker = "user" if item.role == "user" else "assistant"
        
        # Get text content
        text = item.text_content or ""
        
        # Find the most recent turn for this speaker WITHOUT text and fill it in
        turn_found = False
        for turn in reversed(self.transcript_turns):
            if turn["speaker"] == speaker and not turn["text"]:
                turn["text"] = text
                turn["interrupted"] = item.interrupted if hasattr(item, "interrupted") else False
                turn["turn_latency"] = self._get_turn_latency(speaker)
                turn["confidence_score"] = self._get_confidence_score(item)
                
                # Calculate response delay for assistant turns
                if speaker == "assistant" and self.last_user_turn_time_ms is not None:
                    turn["response_delay_ms"] = turn["start_time_ms"] - self.last_user_turn_time_ms
                
                # Update last user turn time
                if speaker == "user":
                    self.last_user_turn_time_ms = turn["end_time_ms"] if turn["end_time_ms"] else turn["start_time_ms"]
                
                logger.debug("✓ Filled text for %s turn: %s...", speaker, text[:200])
                turn_found = True
                break
        
        if not turn_found:
            # Usually benign: the STT path (user_input_transcribed) already filled
            # this turn before the message item arrived, so there's no empty turn
            # left to fill and the text is already captured. Only note it (at debug)
            # when the text isn't present anywhere — no noisy warning either way.
            text_norm = text.strip()
            already_captured = any(
                turn["speaker"] == speaker and (turn.get("text") or "").strip() == text_norm
                for turn in self.transcript_turns
            )
            if already_captured:
                logger.debug("%s text already captured by another path; nothing to fill", speaker)
            else:
                logger.debug("No open %s turn to attach text to: %s...", speaker, text[:200])

        logger.debug("Transcript turn text updated: %s - %s...", speaker, text[:500])

    def _track_agent_handoff(self, item: Any) -> None:
        """Record an ``AgentHandoff`` conversation item for observability.

        Handoffs carry ``old_agent_id`` -> ``new_agent_id`` (and ``created_at``
        in epoch seconds) but no transcript text; capture them separately so a
        handoff is visible in the payload without touching transcript turns.
        """
        try:
            created_at = getattr(item, "created_at", None)
            if created_at and self.call_start_time_ms:
                timestamp_ms = int(created_at * 1000) - self.call_start_time_ms
            else:
                current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                timestamp_ms = current_ms - self.call_start_time_ms if self.call_start_time_ms else None

            entry = {
                "id": getattr(item, "id", None),
                "old_agent_id": getattr(item, "old_agent_id", None),
                "new_agent_id": getattr(item, "new_agent_id", None),
                "timestamp_ms": timestamp_ms,
            }
            self.agent_handoffs.append(entry)
            logger.info(
                "Agent handoff captured: %s -> %s",
                entry["old_agent_id"],
                entry["new_agent_id"],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to track agent handoff: %s", e)

    def _on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        """Handle user input transcribed event (final STT text + metadata).

        NOTE: Some LiveKit agent flows emit user STT via this event but do NOT emit a
        corresponding user ChatMessage via conversation_item_added. If we don't copy
        event.transcript into the most recent user turn, the webhook payload will
        drop user turns (because we filter out turns with empty text).
        """
        if not event.is_final:
            return

        transcript_text = getattr(event, "transcript", None)
        if transcript_text is None:
            # Defensive fallback for any SDK shape differences
            transcript_text = getattr(event, "text", None)
        transcript_text = (transcript_text or "").strip()

        # Find the most recent user turn without text and fill it in.
        target_turn = None
        for turn in reversed(self.transcript_turns):
            if turn.get("speaker") == "user" and not (turn.get("text") or "").strip():
                target_turn = turn
                break

        # Fallback: if we didn't observe a user_state_changed start turn (rare),
        # append a minimal user turn so we don't lose the transcript entirely.
        if target_turn is None:
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            state_time_ms = current_time_ms - self.call_start_time_ms if self.call_start_time_ms else 0
            timestamp = datetime.now(timezone.utc).isoformat()
            target_turn = {
                "speaker": "user",
                "text": "",
                "timestamp": timestamp,
                "start_timestamp": timestamp,
                "end_timestamp": timestamp,
                "start_time_ms": state_time_ms,
                "end_time_ms": state_time_ms,
                "response_delay_ms": None,
                "interrupted": False,
                "turn_latency": None,
                "confidence_score": None,
                "language": None,
                "speaker_id": None,
            }
            self.transcript_turns.append(target_turn)

        # Fill in text from STT if we have it and the turn is empty.
        if transcript_text and not (target_turn.get("text") or "").strip():
            target_turn["text"] = transcript_text
            target_turn["turn_latency"] = self._get_turn_latency("user")

            # Keep response_delay computation consistent with conversation_item_added path
            self.last_user_turn_time_ms = (
                target_turn.get("end_time_ms") if target_turn.get("end_time_ms") is not None else target_turn.get("start_time_ms")
            )

            logger.debug("✓ Filled text for user turn from STT: %s...", transcript_text[:200])

        # Always store metadata if available
        target_turn["language"] = getattr(event, "language", None)
        speaker_id = getattr(event, "speaker_id", None)
        if speaker_id:
            target_turn["speaker_id"] = speaker_id
    
    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        """Handle agent state changes for precise timing."""
        try:
            old_state: AgentState = event.old_state
            new_state: AgentState = event.new_state
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            state_time_ms = current_time_ms - self.call_start_time_ms if self.call_start_time_ms else 0
            timestamp = datetime.now(timezone.utc).isoformat()

            # logger.info("Agent state changed: %s -> %s at %dms (turns count: %d)",
            #            old_state, new_state, state_time_ms, len(self.transcript_turns))
            
            # START: non-speaking -> speaking
            if new_state == 'speaking' and old_state != 'speaking':
                logger.debug("Agent STARTED speaking at %dms", state_time_ms)
                turn = {
                    "speaker": "assistant",
                    "text": "",  # Will be filled by conversation_item_added
                    "timestamp": timestamp,
                    "start_timestamp": timestamp,
                    "end_timestamp": None,
                    "start_time_ms": state_time_ms,
                    "end_time_ms": None,
                    "response_delay_ms": None,
                    "interrupted": False,
                    "turn_latency": None,
                    "confidence_score": None,
                    "language": None,
                    "speaker_id": None,
                }
                self.transcript_turns.append(turn)
                logger.debug("✓ Created assistant turn at start")
            
            # END: speaking -> non-speaking
            elif old_state == 'speaking' and new_state != 'speaking':
                # logger.info("Agent STOPPED speaking at %dms", state_time_ms)
                # Find the last assistant turn without an end time
                for turn in reversed(self.transcript_turns):
                    if turn["speaker"] == "assistant" and turn["end_time_ms"] is None:
                        turn["end_time_ms"] = state_time_ms
                        turn["end_timestamp"] = timestamp
                        # duration_ms = state_time_ms - turn["start_time_ms"]
                        # logger.info("✓ Updated assistant turn end time: duration=%dms", duration_ms)
                        break
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle agent state change: %s", e)
    
    def _on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        """Handle user state changes for precise timing."""
        try:
            old_state: UserState = event.old_state
            new_state: UserState = event.new_state
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            state_time_ms = current_time_ms - self.call_start_time_ms if self.call_start_time_ms else 0
            timestamp = datetime.now(timezone.utc).isoformat()

            # logger.info("User state changed: %s -> %s at %dms (turns count: %d)",
            #            old_state, new_state, state_time_ms, len(self.transcript_turns))
            
            # START: non-speaking -> speaking
            if new_state == 'speaking' and old_state != 'speaking':
                logger.debug("User STARTED speaking at %dms", state_time_ms)
                turn = {
                    "speaker": "user",
                    "text": "",  # Will be filled by user_input_transcribed/conversation_item_added
                    "timestamp": timestamp,
                    "start_timestamp": timestamp,
                    "end_timestamp": None,
                    "start_time_ms": state_time_ms,
                    "end_time_ms": None,
                    "response_delay_ms": None,
                    "interrupted": False,
                    "turn_latency": None,
                    "confidence_score": None,
                    "language": None,
                    "speaker_id": None,
                }
                self.transcript_turns.append(turn)
                # logger.info("✓ Created user turn at start")
            
            # END: speaking -> non-speaking
            elif old_state == 'speaking' and new_state != 'speaking':
                # logger.info("User STOPPED speaking at %dms", state_time_ms)
                # Find the last user turn without an end time
                for turn in reversed(self.transcript_turns):
                    if turn["speaker"] == "user" and turn["end_time_ms"] is None:
                        turn["end_time_ms"] = state_time_ms
                        turn["end_timestamp"] = timestamp
                        # duration_ms = state_time_ms - turn["start_time_ms"]
                        # logger.info("✓ Updated user turn end time: duration=%dms", duration_ms)
                        break
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle user state change: %s", e)
    
    def _on_speech_created(self, event: SpeechCreatedEvent) -> None:
        """Handle speech creation event for additional tracking."""
        try:
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            speech_time_ms = current_time_ms - self.call_start_time_ms if self.call_start_time_ms else 0
            
            # logger.debug("Speech created: source=%s, user_initiated=%s at %dms",
            #             event.source, event.user_initiated, speech_time_ms)
            
            # Track speech creation for analytics
            if not hasattr(self, 'speech_events'):
                self.speech_events = []
            
            self.speech_events.append({
                'source': event.source,
                'user_initiated': event.user_initiated,
                'timestamp_ms': speech_time_ms,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle speech created event: %s", e)
    
    def _on_function_tools_executed(self, event: FunctionToolsExecutedEvent) -> None:
        """Handle function/tool calls and their outputs."""
        try:
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            for fc, fco in event.zipped():
                # Derive start_ms from the FunctionCall's created_at (epoch seconds).
                # Fall back to now if created_at is zero/missing.
                fc_created_at = getattr(fc, "created_at", None)
                if fc_created_at:
                    fc_abs_ms = int(fc_created_at * 1000)
                    start_ms = fc_abs_ms - self.call_start_time_ms if self.call_start_time_ms else None
                else:
                    start_ms = current_time_ms - self.call_start_time_ms if self.call_start_time_ms else None

                # Derive end_ms from the FunctionCallOutput's created_at when present.
                end_ms: Optional[int] = None
                if fco is not None:
                    fco_created_at = getattr(fco, "created_at", None)
                    if fco_created_at:
                        fco_abs_ms = int(fco_created_at * 1000)
                        end_ms = fco_abs_ms - self.call_start_time_ms if self.call_start_time_ms else None

                entry: dict[str, Any] = {
                    "id": fc.call_id,
                    "function_name": fc.name,
                    "arguments": fc.arguments,  # raw JSON string
                    "result": fco.output if fco is not None else None,
                    "is_error": fco.is_error if fco is not None else None,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "timestamp_ms": start_ms,  # backward-compat alias
                }
                self.tool_calls.append(entry)
                logger.info(
                    "Tool call captured: function=%s call_id=%s start_ms=%s",
                    fc.name,
                    fc.call_id,
                    start_ms,
                )

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle function_tools_executed: %s", e)

    def _on_session_close(self, event: CloseEvent) -> None:
        """Handle session close event for cleanup."""
        try:
            logger.info("Session closed")
            if event.error:
                logger.error("Session closed with error: %s", event.error)
            
            # Capture end reason
            if self._preferred_call_end_reason:
                self.call_end_reason = self._preferred_call_end_reason
            elif hasattr(event, 'reason') and event.reason is not None:
                self.call_end_reason = event.reason.value if hasattr(event.reason, 'value') else str(event.reason)
            elif event.error:
                self.call_end_reason = "error"

            # LiveKit's own close info, kept separately: call_end_reason may be
            # overridden by the app via set_call_end_reason.
            try:
                reason = getattr(event, "reason", None)
                err = getattr(event, "error", None)
                self.lk_close_info = {
                    "reason": reason.value if hasattr(reason, "value") else (str(reason) if reason is not None else None),
                    "error_type": getattr(err, "type", None) if err is not None else None,
                    "error_message": str(getattr(err, "error", None) or err)[:_ERROR_MSG_MAX_LEN] if err is not None else None,
                }
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to capture close info: %s", e)

            # Mark session as ended
            self.ended_at = datetime.now(timezone.utc)

            for event_name, callback in self._room_subscriptions:
                try:
                    self.room.off(event_name, callback)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to unsubscribe room listener %s: %s", event_name, e)
            self._room_subscriptions.clear()

            if self._rtc_poll_task is not None and not self._rtc_poll_task.done():
                self._rtc_poll_task.cancel()

            # Unsubscribe from base TTS/STT/LLM components so the handler isn't
            # invoked after we've sent the webhook (and to release references).
            for component, callback in self._base_metric_subscriptions:
                try:
                    if hasattr(component, "off"):
                        component.off("metrics_collected", callback)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to unsubscribe base metrics listener: %s", e)
            self._base_metric_subscriptions.clear()

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle session close: %s", e)

    def _rel_ms(self, epoch_s: Optional[float] = None) -> Optional[int]:
        """Offset in ms from call start, from an epoch-seconds stamp (default: now)."""
        if self.call_start_time_ms is None:
            return None
        if epoch_s:
            return int(epoch_s * 1000) - self.call_start_time_ms
        return int(datetime.now(timezone.utc).timestamp() * 1000) - self.call_start_time_ms

    def _on_error_event(self, event: Any) -> None:
        """Capture provider errors (LLMError / STTError / TTSError / RealtimeModelError)."""
        try:
            err = getattr(event, "error", None)
            source = getattr(event, "source", None)
            inner = getattr(err, "error", None) if err is not None else None
            entry = {
                "timestamp_ms": self._rel_ms(getattr(event, "created_at", None)),
                "error_type": getattr(err, "type", None) or type(err).__name__,
                "message": str(inner if inner is not None else err)[:_ERROR_MSG_MAX_LEN],
                "recoverable": getattr(err, "recoverable", None),
                "label": getattr(err, "label", None),
                "source_model": getattr(source, "model", None),
                "source_provider": _extract_provider_from_module(source),
            }
            if len(self.error_events) < _MAX_ERROR_EVENTS:
                self.error_events.append(entry)
            logger.warning(
                "SUPERBRYN_PROVIDER_ERROR captured: type=%s recoverable=%s message=%s",
                entry["error_type"], entry["recoverable"], entry["message"][:120],
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to capture error event: %s", e)

    def _on_false_interruption(self, event: Any) -> None:
        # Only count — the event's message/extra_instructions fields are deprecated
        self.false_interruption_count += 1

    def _on_connection_quality_changed(self, participant: Any, quality: Any) -> None:
        try:
            quality_name = _enum_name(ConnectionQuality, quality)
            entry = {
                "timestamp_ms": self._rel_ms(),
                "participant": getattr(participant, "identity", None),
                "quality": quality_name,
            }
            if len(self.connection_quality_events) < _MAX_EVENT_LIST:
                self.connection_quality_events.append(entry)
            rank = _QUALITY_RANK.get(quality_name, -1)
            if rank > _QUALITY_RANK.get(self._worst_quality or "", -1):
                self._worst_quality = quality_name
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to capture connection quality: %s", e)

    def _append_connection_event(self, event_type: str, **extra: Any) -> None:
        if len(self.connection_events) < _MAX_EVENT_LIST:
            self.connection_events.append(
                {"type": event_type, "timestamp_ms": self._rel_ms(), **extra}
            )

    def _on_room_reconnecting(self) -> None:
        self._append_connection_event("reconnecting")

    def _on_room_reconnected(self) -> None:
        self._append_connection_event("reconnected")

    def _on_room_disconnected(self, reason: Any = None) -> None:
        try:
            self.room_disconnect_reason = _enum_name(DisconnectReason, reason) if reason is not None else None
            self._append_connection_event("disconnected", reason=self.room_disconnect_reason)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to capture room disconnect: %s", e)

    def _on_participant_disconnected(self, participant: Any) -> None:
        try:
            reason = getattr(participant, "disconnect_reason", None)
            self._append_connection_event(
                "participant_disconnected",
                participant=getattr(participant, "identity", None),
                reason=_enum_name(DisconnectReason, reason) if reason is not None else None,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to capture participant disconnect: %s", e)

    def _on_track_subscription_failed(self, *args: Any) -> None:
        self.track_subscription_failures += 1

    def _on_sip_dtmf(self, dtmf: Any) -> None:
        try:
            if len(self.dtmf_events) < _MAX_DTMF_EVENTS:
                self.dtmf_events.append({
                    "timestamp_ms": self._rel_ms(),
                    "digit": getattr(dtmf, "digit", None),
                    "code": getattr(dtmf, "code", None),
                })
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to capture DTMF: %s", e)

    def _on_participant_attributes_changed(self, changed_attributes: Any, participant: Any) -> None:
        try:
            attrs = dict(changed_attributes or {})
            sip_attrs = {k: v for k, v in attrs.items() if isinstance(k, str) and k.startswith("sip.")}
            if sip_attrs:
                self.sip_attributes.update(sip_attrs)
                self.sip_trunking_enabled = True
                if not self.phone_number and sip_attrs.get("sip.phoneNumber"):
                    self.phone_number = sip_attrs["sip.phoneNumber"]
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to capture participant attributes: %s", e)

    async def _rtc_stats_poll_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_RTC_POLL_INTERVAL_S)
                await self._collect_rtc_sample()
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.debug("RTC stats poll loop stopped: %s", e)

    async def _stop_rtc_poller(self, final_sample: bool = True) -> None:
        if self._rtc_poll_task is not None:
            self._rtc_poll_task.cancel()
            with contextlib.suppress(Exception):
                await self._rtc_poll_task
            self._rtc_poll_task = None
        if final_sample and hasattr(self.room, "get_rtc_stats"):
            await self._collect_rtc_sample()

    async def _collect_rtc_sample(self) -> None:
        """Poll WebRTC stats and fold them into one per-interval audio sample.

        Counters (packets, bytes, concealed samples) are cumulative per WebRTC
        spec — each sample records the latest value; deltas are the consumer's
        job. Gauges (jitter, RTT, audio level) are instantaneous.
        """
        try:
            stats = await self.room.get_rtc_stats()
        except Exception as e:  # noqa: BLE001
            logger.debug("get_rtc_stats failed: %s", e)
            return

        try:
            sample: dict[str, Any] = {"t_ms": self._rel_ms()}
            agg = self._rtc_agg

            for s in list(getattr(stats, "subscriber_stats", []) or []) + list(
                getattr(stats, "publisher_stats", []) or []
            ):
                which = s.WhichOneof("stats") if hasattr(s, "WhichOneof") else None

                if which == "inbound_rtp" and getattr(s.inbound_rtp.stream, "kind", "") == "audio":
                    received = s.inbound_rtp.received
                    inbound = s.inbound_rtp.inbound
                    jitter_ms = round(getattr(received, "jitter", 0.0) * 1000, 1)
                    sample.update({
                        "jitter_ms": jitter_ms,
                        "packets_received": getattr(received, "packets_received", 0),
                        "packets_lost": getattr(received, "packets_lost", 0),
                        "concealed_samples": getattr(inbound, "concealed_samples", 0),
                        "silent_concealed_samples": getattr(inbound, "silent_concealed_samples", 0),
                        "audio_level": round(getattr(inbound, "audio_level", 0.0), 3),
                        "bytes_received": getattr(inbound, "bytes_received", 0),
                        "nack_count": getattr(inbound, "nack_count", 0),
                    })
                    agg["jitter_sum_ms"] = agg.get("jitter_sum_ms", 0.0) + jitter_ms
                    agg["jitter_n"] = agg.get("jitter_n", 0) + 1
                    agg["jitter_max_ms"] = max(agg.get("jitter_max_ms", 0.0), jitter_ms)
                    agg["jitter_buffer_delay_s"] = getattr(inbound, "jitter_buffer_delay", 0.0)
                    agg["jitter_buffer_emitted"] = getattr(inbound, "jitter_buffer_emitted_count", 0)

                elif which == "remote_inbound_rtp":
                    remote = s.remote_inbound_rtp.remote_inbound
                    rtt_ms = round(getattr(remote, "round_trip_time", 0.0) * 1000, 1)
                    fraction_lost = round(getattr(remote, "fraction_lost", 0.0), 4)
                    if rtt_ms > 0:
                        sample["rtt_ms"] = rtt_ms
                        agg["rtt_sum_ms"] = agg.get("rtt_sum_ms", 0.0) + rtt_ms
                        agg["rtt_n"] = agg.get("rtt_n", 0) + 1
                        agg["rtt_max_ms"] = max(agg.get("rtt_max_ms", 0.0), rtt_ms)
                    sample["remote_fraction_lost"] = fraction_lost
                    agg["remote_fraction_lost_max"] = max(
                        agg.get("remote_fraction_lost_max", 0.0), fraction_lost
                    )

                elif which == "outbound_rtp" and getattr(s.outbound_rtp.stream, "kind", "") == "audio":
                    sent = s.outbound_rtp.sent
                    sample.update({
                        "packets_sent": getattr(sent, "packets_sent", 0),
                        "bytes_sent": getattr(sent, "bytes_sent", 0),
                    })

            if len(sample) > 1:
                agg["last_sample"] = sample
                agg["samples_collected"] = agg.get("samples_collected", 0) + 1
                if len(self._rtc_samples) < _MAX_RTC_SAMPLES:
                    self._rtc_samples.append(sample)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to parse rtc stats: %s", e)

    def _on_metrics_collected(self, event: MetricsCollectedEvent) -> None:
        """Handle metrics collected event."""
        metrics_obj = event.metrics

        # Dedup metrics that may be observed both via ``session`` (wrapper-
        # forwarded) and via our defensive base-component subscription.  Only
        # provider-emitted types participate; framework-emitted VAD/EOU events
        # never flow through the base subscription and don't have a stable
        # request_id, so we skip dedup for them.
        if isinstance(metrics_obj, (LLMMetrics, STTMetrics, TTSMetrics, RealtimeModelMetrics)):
            key = (
                getattr(metrics_obj, "type", None),
                getattr(metrics_obj, "request_id", None),
                getattr(metrics_obj, "segment_id", None),
            )
            if key in self._seen_metrics:
                return
            self._seen_metrics.add(key)

        # Aggregation into usage/latency happens below; just log here.
        metrics.log_metrics(metrics_obj)
        
        # Handle different metric types using the discriminator field
        if isinstance(metrics_obj, LLMMetrics):
            # LLM metrics
            self.usage_metrics["llm_input_tokens"] += metrics_obj.prompt_tokens
            self.usage_metrics["llm_output_tokens"] += metrics_obj.completion_tokens
            self.usage_metrics["llm_total_tokens"] += metrics_obj.total_tokens
            
            # Latency (ttft in seconds, convert to ms)
            if metrics_obj.ttft > 0:
                self.latency_metrics["llm_ms"].append(metrics_obj.ttft * 1000)
            elif metrics_obj.duration > 0:
                self.latency_metrics["llm_ms"].append(metrics_obj.duration * 1000)

            self.extra_usage["llm_prompt_cached_tokens"] += getattr(metrics_obj, "prompt_cached_tokens", 0) or 0
            tps = getattr(metrics_obj, "tokens_per_second", 0.0) or 0.0
            if tps > 0:
                self.extra_usage["llm_tokens_per_second"].append(tps)

        elif isinstance(metrics_obj, STTMetrics):
            # STT metrics
            self.usage_metrics["stt_duration_seconds"] += metrics_obj.audio_duration
            self.usage_metrics["audio_duration_seconds"] += metrics_obj.audio_duration

            # Latency (duration in seconds, convert to ms)
            if metrics_obj.duration > 0:
                self.latency_metrics["stt_ms"].append(metrics_obj.duration * 1000)

            acquire = getattr(metrics_obj, "acquire_time", 0.0) or 0.0
            if acquire > 0:
                self.extra_usage["stt_acquire_time_ms"].append(acquire * 1000)

        elif isinstance(metrics_obj, TTSMetrics):
            # TTS metrics
            self.usage_metrics["tts_characters"] += metrics_obj.characters_count
            self.usage_metrics["tts_audio_duration_seconds"] += metrics_obj.audio_duration

            # Latency (ttfb in seconds, convert to ms)
            if metrics_obj.ttfb > 0:
                self.latency_metrics["tts_ms"].append(metrics_obj.ttfb * 1000)
            elif metrics_obj.duration > 0:
                self.latency_metrics["tts_ms"].append(metrics_obj.duration * 1000)

            acquire = getattr(metrics_obj, "acquire_time", 0.0) or 0.0
            if acquire > 0:
                self.extra_usage["tts_acquire_time_ms"].append(acquire * 1000)

        elif isinstance(metrics_obj, RealtimeModelMetrics):
            # Realtime model metrics (e.g., OpenAI Realtime API)
            self.usage_metrics["llm_input_tokens"] += metrics_obj.input_tokens
            self.usage_metrics["llm_output_tokens"] += metrics_obj.output_tokens
            self.usage_metrics["llm_total_tokens"] += metrics_obj.total_tokens

            # Latency (ttft in seconds, convert to ms)
            if metrics_obj.ttft > 0:
                self.latency_metrics["llm_ms"].append(metrics_obj.ttft * 1000)

            input_details = getattr(metrics_obj, "input_token_details", None)
            if input_details is not None:
                self.extra_usage["realtime_input_text_tokens"] += getattr(input_details, "text_tokens", 0) or 0
                self.extra_usage["realtime_input_audio_tokens"] += getattr(input_details, "audio_tokens", 0) or 0
                self.extra_usage["realtime_input_cached_tokens"] += getattr(input_details, "cached_tokens", 0) or 0
            output_details = getattr(metrics_obj, "output_token_details", None)
            if output_details is not None:
                self.extra_usage["realtime_output_text_tokens"] += getattr(output_details, "text_tokens", 0) or 0
                self.extra_usage["realtime_output_audio_tokens"] += getattr(output_details, "audio_tokens", 0) or 0

        else:
            # VAD/EOU/EOT/interruption metrics — matched by their `type`
            # discriminator so this works on SDKs where the classes don't exist
            self._collect_extended_metrics(metrics_obj)

    def _collect_extended_metrics(self, metrics_obj: Any) -> None:
        try:
            mtype = getattr(metrics_obj, "type", None)

            if mtype == "vad_metrics":
                self.vad_stats["inference_count"] += getattr(metrics_obj, "inference_count", 0) or 0
                self.vad_stats["inference_total_s"] += getattr(metrics_obj, "inference_duration_total", 0.0) or 0.0
                self.vad_stats["idle_time_s"] = getattr(metrics_obj, "idle_time", 0.0) or 0.0

            elif mtype == "eou_metrics":
                eou_delay_ms = (getattr(metrics_obj, "end_of_utterance_delay", 0.0) or 0.0) * 1000
                transcription_delay_ms = (getattr(metrics_obj, "transcription_delay", 0.0) or 0.0) * 1000
                if eou_delay_ms > 0:
                    self.eou_delays_ms.append(eou_delay_ms)
                if transcription_delay_ms > 0:
                    self.transcription_delays_ms.append(transcription_delay_ms)
                if len(self.eou_events) < _MAX_EOU_EVENTS:
                    self.eou_events.append({
                        "timestamp_ms": self._rel_ms(getattr(metrics_obj, "timestamp", None)),
                        "eou_delay_ms": round(eou_delay_ms, 1),
                        "transcription_delay_ms": round(transcription_delay_ms, 1),
                        "on_user_turn_completed_delay_ms": round(
                            (getattr(metrics_obj, "on_user_turn_completed_delay", 0.0) or 0.0) * 1000, 1
                        ),
                    })

            elif mtype == "eot_inference_metrics":
                self.eot_stats["inference_count"] += getattr(metrics_obj, "num_requests", 1) or 1
                self.eot_stats["prediction_total_s"] += getattr(metrics_obj, "prediction_duration", 0.0) or 0.0

            elif mtype == "interruption_metrics":
                # Counters are cumulative on the SDK side — latest event wins
                self.interruption_stats = {
                    "detected_interruptions": getattr(metrics_obj, "num_interruptions", 0) or 0,
                    "backchannels": getattr(metrics_obj, "num_backchannels", 0) or 0,
                    "detection_delay_ms": round((getattr(metrics_obj, "detection_delay", 0.0) or 0.0) * 1000, 1),
                }
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to collect extended metrics: %s", e)

    def _subscribe_to_metrics(self, session: AgentSession) -> None:
        """Subscribe to per-plugin ``metrics_collected`` events for STT/LLM/TTS.

        This is the non-deprecated metrics surface — LiveKit's own AgentActivity
        subscribes to the plugin instances the same way; the session-level
        ``metrics_collected`` event is deprecated.

        For each component we attach to two targets:

          * the instance as held by the session (``session.tts`` etc.) — adapters
            such as ``FallbackAdapter`` / ``StreamAdapter`` re-emit their inner
            instances' metrics here, and
          * the unwrapped base provider — to still observe metrics when a custom
            wrapper (``SanitizedTTS``, ``VolumeTTS``, ``MixedAudioTTS``,
            ``NetworkGlitchTTS``, ...) doesn't forward the event.

        Both may deliver the same event; ``_on_metrics_collected`` de-duplicates
        via ``self._seen_metrics`` on ``(type, request_id, segment_id)``.
        """
        seen_targets: set[int] = set()
        for outer, inner_attrs, label in (
            (getattr(session, "tts", None), _TTS_INNER_ATTRS, "tts"),
            (getattr(session, "stt", None), _STT_INNER_ATTRS, "stt"),
            (getattr(session, "llm", None), _LLM_INNER_ATTRS, "llm"),
        ):
            if outer is None:
                continue

            targets = [outer]
            try:
                base = _unwrap_to_base_component(outer, inner_attrs)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to unwrap %s base component: %s", label, e)
                base = None
            if base is not None and base is not outer:
                targets.append(base)

            for target in targets:
                if target is None or not hasattr(target, "on") or id(target) in seen_targets:
                    continue
                seen_targets.add(id(target))
                try:
                    target.on("metrics_collected", self._on_base_metrics_collected)
                    self._base_metric_subscriptions.append((target, self._on_base_metrics_collected))
                    logger.info(
                        "Subscribed to %s metrics on %s",
                        label,
                        getattr(target, "__module__", type(target).__name__),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to subscribe to %s metrics: %s", label, e)

    def _on_base_metrics_collected(self, metrics_obj: Any) -> None:
        """Route a per-plugin ``metrics_collected`` event through the aggregator.

        Per-plugin events deliver the raw metrics object; wrap it in a
        ``MetricsCollectedEvent`` (mirroring the ``AgentSession`` shape) and hand
        it to ``_on_metrics_collected``, which de-duplicates so a metric observed
        on both the outer and the base target is only counted once.
        """
        try:
            self._on_metrics_collected(MetricsCollectedEvent(metrics=metrics_obj))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle base-component metrics event: %s", e)

    def _on_session_usage_updated(self, event: Any) -> None:
        """Store the latest cumulative ``AgentSessionUsage`` (non-deprecated).

        Backfills usage at payload time when the per-plugin path can't observe
        it — notably realtime models, whose metrics are emitted on an internal
        session we don't hold. See ``_backfill_usage_from_session``.
        """
        try:
            self._latest_session_usage = getattr(event, "usage", None)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to store session usage: %s", e)

    def _backfill_usage_from_session(self) -> None:
        """Fill zero usage fields from the latest ``AgentSessionUsage``.

        Only fills fields still at zero, so the per-plugin path (which also
        yields latency) stays authoritative whenever it observed the metrics.
        ``AgentSessionUsage.model_usage`` is a list of per-model entries tagged
        ``llm_usage`` / ``tts_usage`` / ``stt_usage``.
        """
        usage = self._latest_session_usage
        if usage is None:
            return

        llm_in = llm_out = tts_chars = 0
        tts_audio = stt_audio = 0.0
        for m in getattr(usage, "model_usage", None) or []:
            mtype = getattr(m, "type", None)
            if mtype == "llm_usage":
                llm_in += getattr(m, "input_tokens", 0) or 0
                llm_out += getattr(m, "output_tokens", 0) or 0
            elif mtype == "tts_usage":
                tts_chars += getattr(m, "characters_count", 0) or 0
                tts_audio += getattr(m, "audio_duration", 0.0) or 0.0
            elif mtype == "stt_usage":
                stt_audio += getattr(m, "audio_duration", 0.0) or 0.0

        um = self.usage_metrics
        if not um["llm_input_tokens"] and llm_in:
            um["llm_input_tokens"] = llm_in
        if not um["llm_output_tokens"] and llm_out:
            um["llm_output_tokens"] = llm_out
        if not um["llm_total_tokens"] and (llm_in or llm_out):
            um["llm_total_tokens"] = llm_in + llm_out
        if not um["tts_characters"] and tts_chars:
            um["tts_characters"] = tts_chars
        if not um["tts_audio_duration_seconds"] and tts_audio:
            um["tts_audio_duration_seconds"] = tts_audio
        if not um["stt_duration_seconds"] and stt_audio:
            um["stt_duration_seconds"] = stt_audio
            if not um["audio_duration_seconds"]:
                um["audio_duration_seconds"] = stt_audio

    def _calculate_average_latency(self, latencies: list[float]) -> float:
        """Calculate average latency from a list of measurements."""
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)
    
    def _build_webhook_payload(self) -> dict[str, Any]:
        """
        Build the webhook payload in the expected format.
        
        Returns:
            Dictionary matching LivekitWebhookPayload interface
        """
        # Mark end time
        self.ended_at = datetime.now(timezone.utc)
        
        # No post-processing needed - all timestamps come from actual state change events
        # Log any turns with missing end times (shouldn't happen)
        for i, turn in enumerate(self.transcript_turns):
            if turn["end_time_ms"] is None:
                logger.warning("Turn %d (%s) has no end_time_ms - state change event may not have fired",
                              i, turn["speaker"])
        
        # Calculate duration
        duration_seconds = 0
        if self.started_at and self.ended_at:
            duration_seconds = int((self.ended_at - self.started_at).total_seconds())
        
        # Calculate average latencies
        avg_latency = {
            "llm_ms": self._calculate_average_latency(self.latency_metrics["llm_ms"]),
            "stt_ms": self._calculate_average_latency(self.latency_metrics["stt_ms"]),
            "tts_ms": self._calculate_average_latency(self.latency_metrics["tts_ms"]),
        }
        avg_latency["total_ms"] = sum(avg_latency.values())

        if self.extended_capture and self.eou_delays_ms:
            avg_latency["eou_ms"] = self._calculate_average_latency(self.eou_delays_ms)
            avg_latency["transcription_ms"] = self._calculate_average_latency(self.transcription_delays_ms)
            # Perceived response latency: turn-end detection + LLM TTFT + TTS TTFB
            avg_latency["e2e_ms"] = avg_latency["eou_ms"] + avg_latency["llm_ms"] + avg_latency["tts_ms"]

        # Backfill usage from session_usage_updated when the per-plugin path saw
        # nothing (e.g. realtime models emit metrics on an internal session).
        self._backfill_usage_from_session()

        # Debug: Log final usage metrics
        logger.info("Final usage metrics: %s", self.usage_metrics)
        
        # Filter out turns without text
        turns_with_text = [turn for turn in self.transcript_turns if turn.get("text") and turn["text"].strip()]
        logger.info("Filtered transcript: %d total turns, %d turns with text",
                   len(self.transcript_turns), len(turns_with_text))
        
        # Ensure agent_id is set (required by webhook endpoint)
        if not self.agent_id:
            self.agent_id = "livekit-agent-default"
            logger.error("agent_id was None when building webhook payload - using fallback: %s", self.agent_id)
        
        # ``custom_data`` is the only developer-supplied, arbitrary part of the
        # payload.  Round-trip it through JSON (with ``default=str`` for exotic
        # values) so a single non-serialisable entry can't make the whole
        # ``send_webhook`` POST raise and drop the entire payload (transcript,
        # recording, usage, ...).  Natively-serialisable structures pass through
        # unchanged; only exotic objects (datetime, set, custom classes) are
        # coerced to their string form.
        safe_custom_data = self._sanitize_custom_data()

        usage_payload: dict[str, Any] = dict(self.usage_metrics)
        if self.extended_capture:
            usage_payload.update(self._build_usage_extras())

        # Build payload
        payload = {
            "event": "call.ended",
            "call": {
                "id": self.room.name,  # Use room name as call ID
                "room_name": self.room.name,
                "participant_identity": self._get_participant_identity(),
                "ring_started_at": self.ring_started_at.isoformat() if self.ring_started_at else None,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "ended_at": self.ended_at.isoformat() if self.ended_at else None,
                "duration_seconds": duration_seconds,
                "call_end_reason": self.call_end_reason,
                "transcript": {
                    "turns": turns_with_text,
                },
                "tool_calls": self.tool_calls,
                "agent_handoffs": self.agent_handoffs,
                "recording_url": self._get_recording_url(),
                "stereo_recording_url": self._get_stereo_recording_url(),
                # provenance for the consumer's mirror decision (+ call-time probe result)
                "recording_url_source": self.recording_url_source,
                "recording_url_reachable": self.recording_url_reachable,
                "metadata": {
                    "agent_id": self.agent_id,
                    "livekit_project_id": self.livekit_project_id,
                    # Include model/provider info for version config tracking
                    "llm_model": self.usage_metrics["llm_model"],
                    "llm_provider": self.usage_metrics["llm_provider"],
                    "stt_model": self.usage_metrics["stt_model"],
                    "stt_provider": self.usage_metrics["stt_provider"],
                    "tts_model": self.usage_metrics["tts_model"],
                    "tts_provider": self.usage_metrics["tts_provider"],
                    "tts_voice_id": self.usage_metrics["tts_voice_id"],
                    "system_prompt": self.system_prompt,
                    # LiveKit feature flags for cost calculation
                    "sip_trunking_enabled": self.sip_trunking_enabled,
                    "egress_enabled": self.egress_enabled,
                    "lk_agent_enabled": self.is_deployed_on_lk_cloud,
                    # Phone number if available
                    "phone_number": self.phone_number,
                    # Custom telephony rate ($/min) if provided
                    "call_rate_usd": self.call_rate_usd,
                },
                "usage": usage_payload,
                "latency": avg_latency,
                "custom_data": safe_custom_data,
            },
        }

        if self.extended_capture:
            payload["call"].update(self._build_extended_sections(turns_with_text, duration_seconds))

        return payload

    def _build_usage_extras(self) -> dict[str, Any]:
        xu = self.extra_usage
        return {
            "llm_prompt_cached_tokens": xu["llm_prompt_cached_tokens"],
            "llm_tokens_per_second": round(self._calculate_average_latency(xu["llm_tokens_per_second"]), 2),
            "stt_avg_acquire_time_ms": round(self._calculate_average_latency(xu["stt_acquire_time_ms"]), 1),
            "tts_avg_acquire_time_ms": round(self._calculate_average_latency(xu["tts_acquire_time_ms"]), 1),
            "realtime_input_text_tokens": xu["realtime_input_text_tokens"],
            "realtime_input_audio_tokens": xu["realtime_input_audio_tokens"],
            "realtime_input_cached_tokens": xu["realtime_input_cached_tokens"],
            "realtime_output_text_tokens": xu["realtime_output_text_tokens"],
            "realtime_output_audio_tokens": xu["realtime_output_audio_tokens"],
        }

    def _build_extended_sections(
        self, turns: list[dict[str, Any]], duration_seconds: int
    ) -> dict[str, Any]:
        """Assemble the additive payload sections; each is exception-isolated so
        one broken builder degrades to null instead of dropping the webhook."""
        builders = {
            "turn_detection": self._build_turn_detection_section,
            "speech_stats": lambda: self._build_speech_stats(turns, duration_seconds),
            "interruptions": lambda: self._build_interruptions_section(turns),
            "network": self._build_network_section,
            "vad": self._build_vad_section,
            "errors": lambda: self.error_events,
            "close": lambda: self.lk_close_info or None,
            "sip": self._build_sip_section,
            "environment": self._build_environment_section,
        }
        sections: dict[str, Any] = {}
        for name, build in builders.items():
            try:
                sections[name] = build()
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to build extended section %s: %s", name, e)
                sections[name] = None
        return self._json_safe(sections)

    def _build_turn_detection_section(self) -> dict[str, Any]:
        eot_count = self.eot_stats["inference_count"]
        return {
            "avg_eou_delay_ms": round(self._calculate_average_latency(self.eou_delays_ms), 1),
            "max_eou_delay_ms": round(max(self.eou_delays_ms), 1) if self.eou_delays_ms else 0.0,
            "avg_transcription_delay_ms": round(
                self._calculate_average_latency(self.transcription_delays_ms), 1
            ),
            "eot_inference_count": eot_count,
            "avg_eot_prediction_ms": round(
                self.eot_stats["prediction_total_s"] / eot_count * 1000, 1
            ) if eot_count else 0.0,
            "eou_events": self.eou_events,
        }

    def _build_speech_stats(
        self, turns: list[dict[str, Any]], duration_seconds: int
    ) -> dict[str, Any]:
        user_seconds = agent_seconds = 0.0
        user_turns = agent_turns = 0
        response_delays: list[float] = []
        timed: list[tuple[int, int]] = []

        for turn in turns:
            start_ms, end_ms = turn.get("start_time_ms"), turn.get("end_time_ms")
            turn_seconds = (end_ms - start_ms) / 1000 if start_ms is not None and end_ms is not None else 0.0
            if start_ms is not None and end_ms is not None:
                timed.append((start_ms, end_ms))
            if turn.get("speaker") == "user":
                user_turns += 1
                user_seconds += turn_seconds
            else:
                agent_turns += 1
                agent_seconds += turn_seconds
                if turn.get("response_delay_ms") is not None:
                    response_delays.append(float(turn["response_delay_ms"]))

        longest_gap_ms = 0
        timed.sort()
        for (_, prev_end), (next_start, _) in zip(timed, timed[1:]):
            longest_gap_ms = max(longest_gap_ms, next_start - prev_end)

        total_talk = user_seconds + agent_seconds
        return {
            "user_talk_seconds": round(user_seconds, 2),
            "agent_talk_seconds": round(agent_seconds, 2),
            "user_turn_count": user_turns,
            "agent_turn_count": agent_turns,
            "avg_user_turn_seconds": round(user_seconds / user_turns, 2) if user_turns else 0.0,
            "avg_agent_turn_seconds": round(agent_seconds / agent_turns, 2) if agent_turns else 0.0,
            "agent_talk_ratio": round(agent_seconds / total_talk, 3) if total_talk else None,
            # Approximate: overlapping speech is not subtracted
            "silence_seconds": round(max(0.0, duration_seconds - total_talk), 2),
            "longest_silence_ms": longest_gap_ms,
            "avg_response_delay_ms": round(self._calculate_average_latency(response_delays), 1),
            "max_response_delay_ms": round(max(response_delays), 1) if response_delays else 0.0,
        }

    def _build_interruptions_section(self, turns: list[dict[str, Any]]) -> dict[str, Any]:
        section = {
            "agent_turns_interrupted": sum(
                1 for t in turns if t.get("speaker") == "assistant" and t.get("interrupted")
            ),
            "false_interruptions": self.false_interruption_count,
        }
        section.update(self.interruption_stats)
        return section

    def _build_network_section(self) -> dict[str, Any]:
        return {
            "worst_connection_quality": self._worst_quality,
            "connection_quality_events": self.connection_quality_events,
            "reconnect_count": sum(
                1 for e in self.connection_events if e.get("type") == "reconnecting"
            ),
            "connection_events": self.connection_events,
            "track_subscription_failures": self.track_subscription_failures,
            "room_disconnect_reason": self.room_disconnect_reason,
            "rtc": self._build_rtc_section(),
        }

    def _build_rtc_section(self) -> Optional[dict[str, Any]]:
        agg = self._rtc_agg
        if not agg.get("samples_collected"):
            return None
        last = agg.get("last_sample", {})

        packets_received = last.get("packets_received", 0) or 0
        packets_lost = last.get("packets_lost", 0) or 0
        total_packets = packets_received + packets_lost
        emitted = agg.get("jitter_buffer_emitted", 0) or 0

        summary = {
            "samples_collected": agg["samples_collected"],
            "avg_jitter_ms": round(agg["jitter_sum_ms"] / agg["jitter_n"], 1) if agg.get("jitter_n") else None,
            "max_jitter_ms": agg.get("jitter_max_ms"),
            "avg_rtt_ms": round(agg["rtt_sum_ms"] / agg["rtt_n"], 1) if agg.get("rtt_n") else None,
            "max_rtt_ms": agg.get("rtt_max_ms"),
            "packets_received": packets_received,
            "packets_lost": packets_lost,
            "packet_loss_pct": round(packets_lost / total_packets * 100, 2) if total_packets else None,
            "concealed_samples": last.get("concealed_samples"),
            "silent_concealed_samples": last.get("silent_concealed_samples"),
            "nack_count": last.get("nack_count"),
            "bytes_received": last.get("bytes_received"),
            "bytes_sent": last.get("bytes_sent"),
            "avg_jitter_buffer_delay_ms": round(
                agg["jitter_buffer_delay_s"] / emitted * 1000, 1
            ) if emitted else None,
            "remote_fraction_lost_max": agg.get("remote_fraction_lost_max"),
        }
        return {"summary": summary, "samples": self._rtc_samples}

    def _build_vad_section(self) -> dict[str, Any]:
        count = self.vad_stats["inference_count"]
        return {
            "inference_count": count,
            "avg_inference_ms": round(
                self.vad_stats["inference_total_s"] / count * 1000, 2
            ) if count else 0.0,
            "total_inference_seconds": round(self.vad_stats["inference_total_s"], 3),
            "idle_time_seconds": round(self.vad_stats["idle_time_s"], 3),
        }

    def _build_sip_section(self) -> dict[str, Any]:
        attributes = dict(self.sip_attributes)
        try:
            for participant in self.room.remote_participants.values():
                participant_attrs = getattr(participant, "attributes", None) or {}
                attributes.update({
                    k: v for k, v in participant_attrs.items()
                    if isinstance(k, str) and k.startswith("sip.")
                })
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to snapshot SIP attributes: %s", e)
        return {"attributes": attributes, "dtmf": self.dtmf_events}

    def _build_environment_section(self) -> dict[str, Any]:
        def _pkg_version(name: str) -> Optional[str]:
            try:
                return importlib_metadata.version(name)
            except Exception:  # noqa: BLE001
                return None

        return {
            "livekit_evals_version": _pkg_version("livekit-evals"),
            "livekit_agents_version": _pkg_version("livekit-agents"),
            "livekit_rtc_version": _pkg_version("livekit"),
            "python_version": platform.python_version(),
            "rtc_stats_supported": hasattr(self.room, "get_rtc_stats"),
        }

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        try:
            return json.loads(json.dumps(obj, default=str))
        except (TypeError, ValueError) as e:
            logger.warning("Extended sections not JSON-serializable, omitting them: %s", e)
            return {}

    def _sanitize_custom_data(self) -> dict[str, Any]:
        """Return a JSON-safe copy of ``custom_data``.

        Coerces non-serialisable values to strings via ``default=str`` and, as a
        last resort (e.g. circular references), drops ``custom_data`` entirely
        with a warning rather than letting the webhook POST fail.
        """
        if not self.custom_data:
            return {}
        try:
            return json.loads(json.dumps(self.custom_data, default=str))
        except (TypeError, ValueError) as e:
            logger.warning(
                "custom_data is not JSON-serializable, omitting it from payload: %s",
                e,
            )
            return {}
    
    def _get_participant_identity(self) -> str:
        """Get the first non-agent participant identity."""
        for participant in self.room.remote_participants.values():
            return participant.identity
        return "unknown"
    
    def _get_recording_url(self) -> str | None:
        """Get recording URL if available."""
        # Return the recording URL set by RecordingManager
        if self.recording_url:
            return self.recording_url
        
        # Fallback: Check if room has recording info
        if hasattr(self.room, 'recording_url') and self.room.recording_url:
            return self.room.recording_url
        
        # Fallback: Check if room has recording status
        if hasattr(self.room, 'recording_status') and self.room.recording_status:
            # Try to construct URL from room name and LiveKit project
            if self.livekit_project_id and self.livekit_project_id != "not-set":
                return f"https://cloud.livekit.io/projects/{self.livekit_project_id}/recordings/{self.room.name}"
        
        return None
    
    def _get_stereo_recording_url(self) -> str | None:
        """Get stereo recording URL if available."""
        # Return the stereo recording URL set by RecordingManager
        if self.stereo_recording_url:
            return self.stereo_recording_url
        
        # Fallback: Check if room has stereo recording info
        if hasattr(self.room, 'stereo_recording_url') and self.room.stereo_recording_url:
            return self.room.stereo_recording_url
        
        # For now, return None as stereo recording is less common
        return None
    
    def _get_turn_latency(self, speaker: str) -> dict[str, float] | None:
        """Get turn-level latency breakdown if available."""
        if speaker == "assistant":
            # For assistant turns, we can estimate latency from recent metrics
            return {
                "llm_ms": self._calculate_average_latency(self.latency_metrics["llm_ms"]) if self.latency_metrics["llm_ms"] else 0,
                "tts_ms": self._calculate_average_latency(self.latency_metrics["tts_ms"]) if self.latency_metrics["tts_ms"] else 0,
                "total_ms": self._calculate_average_latency(self.latency_metrics["llm_ms"]) + self._calculate_average_latency(self.latency_metrics["tts_ms"]),
            }
        elif speaker == "user":
            # For user turns, STT latency
            return {
                "stt_ms": self._calculate_average_latency(self.latency_metrics["stt_ms"]) if self.latency_metrics["stt_ms"] else 0,
                "total_ms": self._calculate_average_latency(self.latency_metrics["stt_ms"]) if self.latency_metrics["stt_ms"] else 0,
            }
        return None
    
    def _get_confidence_score(self, item: ChatMessage) -> float | None:
        """Get confidence score if available from the message item."""
        # Check if the item has confidence information
        if hasattr(item, 'confidence') and item.confidence is not None:
            return float(item.confidence)
        
        # Check for STT confidence in metadata
        if hasattr(item, 'metadata') and item.metadata:
            if 'confidence' in item.metadata:
                return float(item.metadata['confidence'])
            if 'stt_confidence' in item.metadata:
                return float(item.metadata['stt_confidence'])
        
        return None
    
    async def send_webhook(self) -> None:
        """
        Build and send the webhook payload to the endpoint.
        
        This should be called when the session ends.
        Authenticates using API key from parameter or environment (SUPERBRYN_API_KEY).
        Cleans up recording resources after sending webhook.
        
        Args:
            api_key_override: Optional API key to override environment variable
        """
        try:
            # Final network sample + poller shutdown before the payload is built
            if self.extended_capture:
                try:
                    await self._stop_rtc_poller(final_sample=True)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to stop rtc poller: %s", e)

            # Clean up recording resources first
            if self.recording_manager:
                try:
                    await self.recording_manager.stop_recording()
                    logger.info("Recording manager cleanup completed")
                except Exception as e:  # noqa: BLE001
                    logger.error("Error during recording cleanup: %s", e, exc_info=True)
            
            # Build webhook payload (egress_enabled is set by set_recording_url if recording is active)
            payload = self._build_webhook_payload()
            
            # Get API key from parameter or environment
            api_key = self.api_key
            if not api_key:
                logger.error("SUPERBRYN_API_KEY not configured and no api_key provided, webhook disabled")
                return
            
            logger.info(
                "Sending webhook to %s for call %s",
                self.webhook_url,
                payload["call"]["id"],
            )
            logger.debug("Webhook payload: %s", payload)
            
            # Send webhook via HTTP POST with API key authentication
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": api_key,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        logger.info("SUPERBRYN_WEBHOOK_SENT: %s", response_text)
                    elif response.status == 401:
                        logger.error(
                            "SUPERBRYN_WEBHOOK_UNAUTHORIZED: %s - Check SUPERBRYN_API_KEY",
                            response_text,
                        )
                    elif response.status == 403:
                        logger.error(
                            "SUPERBRYN_WEBHOOK_FORBIDDEN: %s - API key may be expired or disabled",
                            response_text,
                        )
                    else:
                        logger.error(
                            "SUPERBRYN_WEBHOOK_FAILED: status %s: %s",
                            response.status,
                            response_text,
                        )
        
        except Exception as e:  # noqa: BLE001
            logger.error("SUPERBRYN_WEBHOOK_ERROR: %s", e, exc_info=True)


def create_webhook_handler(
    room: Room,
    is_deployed_on_lk_cloud: bool,
    livekit_project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    call_rate_usd: Optional[float] = None,
    disable_recording: bool = False,
    stereo_recording: bool = False,
    defer_recording: bool = False,
    custom_data: Optional[dict[str, Any]] = None,
    extended_capture: bool = True,
) -> Optional[WebhookHandler]:
    """
    Factory function to create a webhook handler from environment variables.
    
    Auto-detects agent_id, version_id, system_prompt, phone_number, SIP trunking,
    and egress recording from session context and room participants.
    
    Recording is ENABLED by default. Temporary S3 credentials are fetched
    per-session using the SUPERBRYN_API_KEY -- no S3 keys need to be configured.
    
    When ``stereo_recording=True``, the egress uses ``DUAL_CHANNEL_AGENT`` audio
    mixing which places the agent on the left channel and all other participants
    on the right channel. This also auto-populates ``stereo_recording_url`` in
    the webhook payload.
    
    When ``defer_recording=True``, recording does NOT start automatically in
    ``attach_to_session``.  Call ``webhook_handler.start_recording()`` when ready
    (e.g. when a remote participant connects).
    
    Requires SUPERBRYN_API_KEY in environment or as parameter for webhook authentication.
    
    Args:
        room: LiveKit room instance
        is_deployed_on_lk_cloud: Whether agent is deployed on LiveKit Cloud ($0.014/min) - REQUIRED
        livekit_project_id: LiveKit project ID (defaults to env var or extracted from LIVEKIT_URL)
        api_key: Override API key (defaults to env var SUPERBRYN_API_KEY)
        call_rate_usd: Custom telephony rate per minute ($/min) for cost calculation (optional)
        disable_recording: Set to True to disable call recording (default: False, recording enabled)
        stereo_recording: If True, record in dual-channel stereo (L=agent, R=others).
            Implies recording is enabled (overrides disable_recording).
        defer_recording: If True, recording will not start in attach_to_session.
            Call start_recording() explicitly when the remote participant connects.
        custom_data: Arbitrary JSON-serializable dict forwarded as-is in
            ``payload["call"]["custom_data"]``.  Useful for attaching
            session-level context known at startup (e.g. ticket ID, user tier).
            Use ``webhook_handler.update_custom_data()`` later to add fields
            discovered during the call.
        extended_capture: When True (default), also capture turn-detection (EOU)
            latency, VAD/interruption analytics, provider errors, network
            quality + WebRTC stats, SIP attributes and DTMF as additive payload
            sections. Set False for the legacy payload shape.

    Returns:
        WebhookHandler instance or None if webhook is disabled
    """
    # Get configuration from config.py (with parameter overrides)
    webhook_url = WEBHOOK_CONFIG["url"]
    livekit_project_id = livekit_project_id or LIVEKIT_CONFIG["project_id"]
    
    # Check for API key (parameter override, then config)
    resolved_api_key = api_key or WEBHOOK_CONFIG["api_key"]
    if not resolved_api_key:
        logger.warning("SUPERBRYN_API_KEY not configured in config.py, webhook disabled")
        return None
    
    # If project ID not explicitly set, try to extract from LIVEKIT_URL
    if not livekit_project_id:
        livekit_url = os.getenv("LIVEKIT_URL") or os.getenv("LIVEKIT_WS_URL") or os.getenv("LIVEKIT_WSS_URL")
        if livekit_url:
            livekit_project_id = _extract_project_id_from_url(livekit_url)
            if livekit_project_id:
                logger.info("Extracted project ID from LIVEKIT_URL: %s", livekit_project_id)
    
    # Skip webhook if URL not configured
    if not webhook_url:
        logger.warning("WEBHOOK_URL not configured, webhook disabled")
        return None
    
    recording_manager = None
    should_record = stereo_recording or not disable_recording

    if not should_record:
        logger.info("Recording disabled by disable_recording=True flag")
    else:
        try:
            recording_manager = RecordingManager(
                credentials_url=CREDENTIALS_CONFIG["url"],
                api_key=resolved_api_key,
                stereo=stereo_recording,
            )
            logger.info("Recording manager initialized (credentials fetched per-session)")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to initialize recording manager: %s", e, exc_info=True)
            logger.warning("Continuing without recording functionality")
    
    handler = WebhookHandler(
        webhook_url=webhook_url,
        api_key=resolved_api_key,
        room=room,
        is_deployed_on_lk_cloud=is_deployed_on_lk_cloud,
        livekit_project_id=livekit_project_id,
        call_rate_usd=call_rate_usd,
        recording_manager=recording_manager,
        disable_recording=disable_recording,
        stereo_recording=stereo_recording,
        defer_recording=defer_recording,
        custom_data=custom_data,
        extended_capture=extended_capture,
    )
    
    mode = "stereo" if stereo_recording else ("disabled" if not should_record else ("enabled" if recording_manager else "unavailable"))
    logger.info("SUPERBRYN_WEBHOOK_HANDLER_CREATED: is_deployed_on_lk_cloud=%s, recording=%s",
               is_deployed_on_lk_cloud, mode)
    return handler
