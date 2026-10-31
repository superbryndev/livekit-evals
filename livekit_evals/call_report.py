"""
One report per call: attach(ctx, session) sends the agent's own record of each call (transcript, tool calls, latency,
configuration, usage and failures) to SuperBryn. A SuperBryn test call is reported as surface "testing"; every other
call as "observability", with a stereo recording (unless record=False), unless production=False. Everything runs in
the background.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import re
import stat
import tempfile
import time
import uuid
import weakref
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import aiohttp
from livekit import rtc
from livekit.agents import llm

from .prompt_sync import prompt_ref, push_prompt
from .recording_manager import RecordingManager

if TYPE_CHECKING:
    from livekit.agents import AgentSession, JobContext

logger = logging.getLogger("livekit_evals")

_DEFAULT_BASE_URL = "https://api.superbryn.com"
_REPORT_PATH = "/public-api/v1/calls/agent-data"
_CALLER_NUMBERS_PATH = "/public-api/v1/test-calls/caller-numbers"
_CREDENTIALS_PATH = "/api/recording-credentials"
_TEST_ID_KEY_SUFFIXES = ("x-superbryn-call-id", "superbryn_call_id")

_SIP_WAIT_S = 15.0
_POLL_S = 0.25
# sip.callStatus once the phone call is connected ("automation": connected, still dialling DTMF)
_ANSWERED = ("active", "automation")
# why a phone call never connected, as LiveKit names it (DisconnectReason)
_NOT_CONNECTED = ("USER_UNAVAILABLE", "USER_REJECTED", "SIP_TRUNK_FAILURE")
_SID_WAIT_S = 5.0
_CALLER_NUMBERS_TTL_S = 3600.0
_CALLER_NUMBERS_RETRY_S = 60.0
_ATTEMPTS = 3
_BACKOFF_S = (1.0, 2.0)
_STARTED_TIMEOUT_S = 5.0
_ENDED_TIMEOUT_S = 6.0
# all the shutdown work, call.ended included: LiveKit kills the job's process 10 s into its shutdown
_SHUTDOWN_S = 8.0
_MAX_TURNS = 5000
_MAX_TOOL_CALLS = 1000
_MAX_TEXT = 20000
_SCHEMA_VERSION = 1
_MAX_TURN_LATENCIES = 5000
_MAX_TOOLS = 200
_MAX_ID = 200
_MAX_NAME = 200
_MAX_DESCRIPTION = 4000
_MAX_LLM_REQUESTS = 2000
_MAX_ERRORS = 500
_MAX_EVENTS = 5000
_MAX_DIGIT = 8
_MAX_URL = 2000
_MAX_SYSTEM_PROMPT = 200_000
# The server takes times between the years 2000 and 2100.
_EARLIEST_S, _LATEST_S = 946_684_800.0, 4_102_444_800.0
# An LLM error this long after a request's metrics were stamped still belongs to it.
_FAILURE_SLACK_S = 1.0
# A realtime response or error stamped before this (2001) is on the monotonic clock, as Nova Sonic stamps them.
_EPOCH_FLOOR = 1e9
# LiveKit marks the caller speaking a moment after the provider stamps the line's start.
_SPEECH_SLACK_S = 0.25
# A caller line still open when the session closed is ended at the close by LiveKit; one ending this near it is checked.
_CLOSE_SLACK_S = 2.0
_MAX_ERROR_MESSAGE = 1000
# LiveKit's error types by pipeline step; a realtime model is the LLM step
_ERROR_STEPS = {"stt_error": "stt", "llm_error": "llm", "tts_error": "tts", "realtime_model_error": "llm"}
# The metrics a working step records, by step: the first after a failure is when that step came back.
_METRIC_STEPS = {"stt_metrics": "stt", "llm_metrics": "llm", "tts_metrics": "tts", "realtime_model_metrics": "llm"}
# Where LiveKit's FallbackAdapter / StreamAdapter / RealtimeModelFallbackAdapter (1.x) and common wrappers keep
# the component they wrap.
_WRAPPED_ATTRS = {
    "stt": ("_stt_instances", "_stt", "_wrapped_stt", "_inner", "_wrapped"),
    "llm": ("_llm_instances", "_llm", "_wrapped_llm", "_inner", "_wrapped", "_models"),
    "tts": ("_tts_instances", "_wrapped_tts", "_tts", "_inner", "_wrapped"),
}
_KNOWN_VENDORS = {"openai", "google", "xai", "anthropic", "deepgram", "cartesia", "elevenlabs", "assemblyai", "rime",
                  "inworld", "mistral", "meta", "groq", "cerebras", "baseten", "sarvam", "speechmatics", "gladia",
                  "aws", "azure", "phonic"}
# A failing model's vendor by a part of its name, first match wins (as 0.3.x named error sources)
_MODEL_VENDORS = (
    ("openai", ("gpt", "openai", "whisper", "tts-1", "o1-", "o3-")), ("anthropic", ("claude", "anthropic")),
    ("google", ("gemini", "palm", "bard", "gemma")), ("meta", ("llama",)), ("mistral", ("mistral", "mixtral")),
    ("cohere", ("cohere", "command")), ("perplexity", ("perplexity", "pplx")), ("groq", ("groq",)),
    ("together", ("together",)), ("replicate", ("replicate",)), ("huggingface", ("huggingface", "hf-")),
    ("elevenlabs", ("eleven",)), ("cartesia", ("cartesia", "sonic")), ("playht", ("playht", "play.ht", "play-ht")),
    ("resemble", ("resemble",)), ("murf", ("murf",)), ("wellsaid", ("wellsaid",)), ("speechify", ("speechify",)),
    ("sarvam", ("saarika", "sarvam", "bulbul")), ("azure", ("azure", "microsoft")),
    ("aws", ("aws", "polly", "amazon")), ("google-cloud", ("gcloud", "google-cloud")),
    ("deepgram", ("deepgram", "nova", "aura")), ("assemblyai", ("assembly",)), ("rev", ("rev.ai", "revai")),
    ("speechmatics", ("speechmatics",)), ("gladia", ("gladia",)), ("livekit", ("livekit",)), ("twilio", ("twilio",)),
    ("vonage", ("vonage",)),
)
_LEFT = getattr(rtc, "DisconnectReason", None)  # why a participant left (livekit-rtc 1.x)

# base URL and key -> (monotonic fetch time, caller numbers); a temp file shares them with later jobs' processes
_caller_numbers_cache: dict[str, tuple[float, list[str]]] = {}
_attached_jobs: weakref.WeakSet = weakref.WeakSet()  # job contexts a report is attached to
_no_key_logged = False
_bad_agent_logged = False
_numbers_refused_logged = False


def attach(
    ctx: JobContext,
    session: AgentSession,
    *,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    production: bool = True,
    record: bool = True,
    custom_data: Optional[dict[str, Any]] = None,
    prompt: Optional[str] = None,
    prompt_version: Optional[str] = None,
    base_url: Optional[str] = None,
) -> CallReport:
    """Report this call to SuperBryn; call once per job, before session.start(). Returns at once with the call's
    handle: recognition, recording and sending run in a background task and one shutdown callback."""
    global _no_key_logged
    report = CallReport()
    report.set_custom_data(custom_data if custom_data is not None else {})
    try:
        if callable(getattr(ctx, "is_fake_job", None)) and ctx.is_fake_job():
            logger.info("SUPERBRYN_CALL_REPORT: console run, nothing is sent")
            return report
        key = api_key or os.getenv("SUPERBRYN_API_KEY")
        if not key:
            if not _no_key_logged:
                _no_key_logged = True
                logger.warning("SUPERBRYN_CALL_REPORT_DISABLED: no API key, set SUPERBRYN_API_KEY")
            return report
        if ctx in _attached_jobs:  # one report per job: a second one would reuse its event IDs
            logger.warning("SUPERBRYN_CALL_REPORT_ERROR: attach already ran in this job, only the first session is "
                           "reported")
            return report
        base = (base_url or os.getenv("SUPERBRYN_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        version = str(prompt_version) if prompt_version is not None else None
        report._attach(ctx, session, key, _resolve_agent_id(agent_id), base, production, prompt, version, record)
        _attached_jobs.add(ctx)
    except Exception as e:  # noqa: BLE001
        logger.warning("SUPERBRYN_CALL_REPORT_ERROR: could not attach: %s", e)
    return report


class CallReport:
    """The handle attach() returns, and the call's state. What is known only during the call goes in through
    set_custom_data(), update_custom_data(), set_call_end_reason() and set_recording_url() until the call ends."""

    def __init__(self) -> None:
        self._custom_data: dict[str, Any] = {}
        self._end_reason: Optional[str] = None
        self._external_recording: Optional[str] = None
        self._ctx: Any = None
        self._session: Any = None
        self._api_key = ""
        self._agent_id: Optional[str] = None
        self._base_url = _DEFAULT_BASE_URL
        self._production = True
        self._record = True
        self._prompt: Optional[str] = None
        self._prompt_version: Optional[str] = None
        self._room_name = ""
        self._attached_at = time.time()
        self._task: Optional[asyncio.Task] = None
        self._caller: Any = None  # the participant on the other end: SIP, else a web caller
        self._surface: Optional[str] = None  # "testing" or "observability" once recognised
        self._test_id: Optional[str] = None
        self._is_test: Optional[bool] = None
        self._call_id = ""
        self._start_fields: dict[str, Any] = {}
        self._started_ts: Optional[float] = None  # when the call connected: every *_ms counts from it
        self._ring_ts: Optional[float] = None  # when an outbound call's phone leg joined: dialling began
        self._answered = True  # False for an outbound phone call that never connected
        self._outbound = False  # a phone call the agent dialled
        self._answer_unknown = False  # its answer couldn't be watched: counted as connected
        # the server refused call.started, so it would refuse call.ended too
        self._refused = False
        self._close_reason: Optional[str] = None
        self._close_error: Optional[dict[str, Any]] = None
        # when the session closed, i.e. the caller hung up; the job shuts down only once the room goes
        self._closed_at: Optional[float] = None
        self._agent_seen = False
        self._system_prompt: Optional[str] = None  # the first agent's, read when it became active
        self._tools_seen: dict[str, Optional[str]] = {}  # tool name -> first description seen
        self._turn_detection: Optional[dict[str, Any]] = None  # the running agent's, sampled during the call
        self._interruptions: Optional[dict[str, Any]] = None  # the running agent's, sampled during the call
        self._sample_failed = False
        self._dtmf: list[tuple[float, str]] = []  # (when, digit) of each key the caller pressed
        self._recorder: Optional[RecordingManager] = None
        self._recording_url: Optional[str] = None

    def set_custom_data(self, data: dict[str, Any]) -> None:
        """Replace custom_data: your own keys for this call, sent as given (same names, same value types)."""
        given = _as_dict(data, "set_custom_data")
        if given is not None:
            self._custom_data = given

    def update_custom_data(self, data: dict[str, Any]) -> None:
        """Shallow-merge data into custom_data; keys it doesn't name are kept."""
        given = _as_dict(data, "update_custom_data")
        if given is not None:
            self._custom_data.update(given)

    def set_call_end_reason(self, reason: Optional[str]) -> None:
        """Send this as ended_reason instead of LiveKit's close reason (e.g. "transfer_to_human")."""
        if _is_text(reason, "set_call_end_reason"):
            self._end_reason = _cut(reason.strip(), _MAX_ID)

    def set_recording_url(self, url: Optional[str]) -> None:
        """A production call's recording from your own egress (a public or pre-signed URL), sent instead of
        SuperBryn's; set before the call is recognised, SuperBryn's recording is not started."""
        if _is_text(url, "set_recording_url"):
            self._external_recording = url.strip()

    def _attach(self, ctx: Any, session: Any, api_key: str, agent_id: Optional[str], base_url: str,
                production: bool, prompt: Optional[str], prompt_version: Optional[str], record: bool = True) -> None:
        """Listen to the session and the room, then start recognising the call in the background."""
        loop = asyncio.get_running_loop()
        self._ctx, self._session = ctx, session
        self._api_key, self._agent_id, self._base_url = api_key, agent_id, base_url
        self._production = bool(production)
        # record=False: production calls are reported without SuperBryn's recording (e.g. for compliance).
        self._record = bool(record)
        self._prompt = prompt if isinstance(prompt, str) and prompt else None
        self._prompt_version = prompt_version
        # the room is usually not connected yet, so its name comes from the job
        job_room = getattr(getattr(ctx, "job", None), "room", None)
        self._room_name = getattr(job_room, "name", "") or getattr(ctx.room, "name", "") or ""
        for event, callback in (("close", self._on_close), ("conversation_item_added", self._sample),
                                ("agent_state_changed", self._sample)):
            try:
                session.on(event, callback)
            except Exception as e:  # noqa: BLE001 - not an event emitter: those fields are left out
                logger.debug("SUPERBRYN_CALL_REPORT: cannot listen to %s: %s", event, e)
        try:
            ctx.room.on("sip_dtmf_received", self._on_dtmf)
        except Exception as e:  # noqa: BLE001 - keypresses are left out
            logger.debug("SUPERBRYN_CALL_REPORT: cannot listen to keypresses: %s", e)
        self._sample()  # attached after session.start(): the first agent is already running
        self._task = loop.create_task(self._run(), name="superbryn_call_report")
        ctx.add_shutdown_callback(self._on_shutdown)

    def _confirmed(self) -> bool:
        """A recognised call, unless the server said it isn't a test or refused call.started."""
        if self._is_test is not None:
            return self._is_test
        return bool(self._call_id and self._start_fields) and not self._refused

    async def _run(self) -> bool:
        """Recognise the call, start its recording (production) and send call.started; True when it is reported."""
        try:
            await self._recognise_and_start()
        except Exception as e:  # noqa: BLE001
            logger.warning("SUPERBRYN_CALL_REPORT_ERROR: %s", e)
        return self._confirmed()

    async def _recognise_and_start(self) -> None:
        caller = self._caller = await _wait_for_caller(self._ctx.room)
        attributes = dict(getattr(caller, "attributes", None) or {})
        sip = getattr(caller, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        inbound = bool(attributes.get("sip.ruleID"))  # a dispatch rule matched: the call came in to the agent
        self._outbound = sip and not inbound
        # a dialled call's answer is watched from here, so the lookups below can't make its start late; an inbound one
        # counts from when it joined, as before
        answer = asyncio.ensure_future(_wait_for_answer(self._ctx.room, caller)) if self._outbound else None
        try:
            await self._start(caller, attributes, sip, inbound, answer)
        finally:
            if answer is not None:
                if not answer.done():
                    answer.cancel()
                if answer.cancelled() or not answer.done():
                    self._answered = False  # the call ended while it still rang
                elif answer.exception() is None:
                    self._take_answer(*answer.result())

    def _take_answer(self, answered: bool, answered_at: Optional[float]) -> None:
        """An outbound call's answer: whether it connected, and its start moved to when it did."""
        self._answered = answered
        if answered_at is not None and self._started_ts is not None and self._start_fields:
            self._started_ts = max(answered_at, self._started_ts)
            self._start_fields["started_at"] = _iso(self._started_ts)

    async def _start(self, caller: Any, attributes: dict[str, Any], sip: bool, inbound: bool,
                     answer: Optional[asyncio.Future]) -> None:
        """Pick the surface, then record (production) and send call.started."""
        self._test_id = _test_id(self._ctx, attributes)
        remote = _e164(attributes.get("sip.phoneNumber") or None) if sip else None
        if self._test_id or await _is_caller_number(remote, self._base_url, self._api_key):
            self._surface = "testing"
        elif self._production:
            self._surface = "observability"
        else:
            logger.debug("not a SuperBryn test call and production=False: nothing is sent")
            return

        # event_id adds "lk:" and ":started" to it, and both stay within the server's 200
        self._call_id = _cut(await _room_sid(self._ctx) or self._room_name, _MAX_ID - len("lk::started"))
        if not self._call_id:
            logger.warning("SUPERBRYN_CALL_REPORT_ERROR: no room SID or name, nothing is sent")
            return
        trunk = _e164(attributes.get("sip.trunkPhoneNumber") or None) if sip else None
        self._started_ts = _joined_at(caller) or (time.time() if caller is not None else self._attached_at)
        if self._outbound:  # an outbound call rings from when it is dialled; an inbound one from its room's creation
            self._ring_ts = self._started_ts
        block = ({"superbryn_call_id": self._test_id} if self._surface == "testing"
                 else {"direction": ("inbound" if inbound else "outbound") if sip else None})
        self._start_fields = _compact({
            "agent_id": self._agent_id,
            "from": _cut(remote if inbound else trunk, _MAX_ID),
            "to": _cut(trunk if inbound else remote, _MAX_ID),
            "started_at": _iso(self._started_ts),
            self._surface: _compact(block) or None,
        })
        if answer is not None:
            # the start fields are set first: a call cancelled while it still rings is reported from them
            try:
                answered = await answer
            except Exception as e:  # noqa: BLE001 - counted from when the caller joined, as without a call status
                logger.warning("SUPERBRYN_CALL_REPORT_ERROR: could not watch for the answer: %s", e)
                answered, self._answer_unknown = (True, None), True
            self._take_answer(*answered)
        self._sample()
        if self._surface == "observability" and self._record and self._answered and not self._external_recording:
            await self._start_recording()
        status, answer = await _send("POST", self._base_url + _REPORT_PATH, self._api_key,
                                     self._event("started"), timeout=_STARTED_TIMEOUT_S)
        self._refused = status is not None and answer is None
        self._read_answer("call.started", answer)
        if self._prompt and self._confirmed():  # once per process per version; never raises
            await push_prompt(api_key=self._api_key, api_base_url=self._base_url, template=self._prompt,
                              version=self._prompt_version, agent_id=self._agent_id)

    async def _start_recording(self) -> None:
        """SuperBryn's stereo recording of a production call, the agent and the caller on their own channels as
        RecordingManager lays them out; a failure leaves the call unrecorded, never unreported."""
        try:
            self._recorder = RecordingManager(credentials_url=self._base_url + _CREDENTIALS_PATH, api_key=self._api_key)
            self._recording_url = await self._recorder.start_recording(room_name=self._room_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("SUPERBRYN_CALL_REPORT_ERROR: recording not started: %s", e)
            return
        if self._recording_url is None:  # the reason is logged by livekit_evals.recording_manager just before
            logger.warning("SUPERBRYN_CALL_REPORT_ERROR: recording not started for call %s", self._call_id)

    async def _stop_recording(self) -> None:
        """Stop SuperBryn's egress so the recording's file is finished; never raises."""
        if self._recorder is None:
            return
        try:
            await self._recorder.stop_recording()
        except Exception as e:  # noqa: BLE001
            logger.warning("SUPERBRYN_CALL_REPORT_ERROR: recording not stopped: %s", e)

    async def _on_shutdown(self, reason: str = "") -> None:
        """Send call.ended for a reported call and stop its recording, all within 8 s of the job's shutdown; never
        raises."""
        try:
            await asyncio.wait_for(self._end(reason), _SHUTDOWN_S)
        except asyncio.TimeoutError:
            logger.error("SUPERBRYN_CALL_REPORT_FAILED: call.ended for call %s not sent within %g s of shutdown",
                         self._call_id, _SHUTDOWN_S)
        except Exception as e:  # noqa: BLE001
            logger.warning("SUPERBRYN_CALL_REPORT_ERROR: %s", e)

    async def _end(self, reason: str) -> None:
        """Stop an unfinished recognition, then send call.ended while the recording stops."""
        if self._task is not None and not self._task.done():
            # still waiting for the caller, the caller numbers, the recording or call.started's answer: stop it;
            # call.ended repeats the start fields, so once they are set it can be sent alone
            self._task.cancel()
            await asyncio.wait({self._task})
        with contextlib.suppress(Exception):
            self._ctx.room.off("sip_dtmf_received", self._on_dtmf)
        await asyncio.gather(self._send_ended(reason), self._stop_recording())

    async def _send_ended(self, reason: str) -> None:
        """call.ended with the full record, for a recognised call."""
        if not (self._confirmed() and self._call_id):
            return
        self._sample()
        session, base = self._session, self._started_ts
        items = list(getattr(getattr(session, "history", None), "items", None) or [])
        # every event the session emitted, as its own session report reads them
        recorded = _attr(session, "_recorded_events")
        events = list(recorded or ())
        requests = _section("llm_requests", _request_log, events) or []
        # a realtime model hears the caller itself: when they talked comes from the session's user states
        speech = (_section("caller_speech", _caller_speech, events) or []
                  if _section("pipeline", _pipeline, session) == "realtime" else None)
        # LiveKit Inference STT says 1.0 when its gateway gives no confidence, as realtime models do
        stand_in = _inference_stt(session)
        # a realtime model that marks the caller stopped only once its reply starts has no real caller times
        caller_times = speech is None or _caller_times_real(session)
        # the caller's turns from the session's user states, for a caller line LiveKit ended at the close
        states = speech if speech is not None else (_section("caller_speech", _caller_speech, events) or [])
        closed = self._closed_at if self._closed_at is not None else time.time()
        transcript, tool_calls = _history_payload(items, base, speech, stand_in, caller_times, closed, states)
        llm_requests = _section("llm_requests", _llm_requests, requests, events, items, base)
        # failures come from the session's event record; an SDK without one can't say there were none
        errors = _section("errors", _errors, events, base) if recorded is not None else None
        never_connected = (_not_connected(self._caller, self._answered)
                           if self._outbound and not self._answer_unknown else None)
        ended_reason = (self._end_reason or never_connected or self._close_reason
                        or (reason.strip() if isinstance(reason, str) else ""))
        body = self._event("ended", **_compact({
            "ended_at": self._ended_at(),
            "ended_reason": _cut_utf16(ended_reason, _MAX_NAME) or None,
            "ended_by": _section("ended_by", _ended_by, self._close_reason, self._caller),
            "transcript": _strict_json("transcript", transcript),
            "tool_calls": _strict_json("tool_calls", tool_calls),
            "diagnostics": _section("diagnostics", _diagnostics, session, items, transcript, self._tools_seen,
                                    self._turn_detection, llm_requests, errors, requests, speech, self._interruptions,
                                    _section("eou_events", _eou_events, events, base),
                                    _section("dtmf", _dtmf, self._dtmf, base)),
            "context": _section("context", self._context),
            "custom_data": _section("custom_data", _custom_data, self._custom_data),
        }))
        if self._surface == "observability":  # the direction, with the recording
            body["observability"] = {**(self._start_fields.get("observability") or {}), **self._recording()}
            if not body["observability"]:
                del body["observability"]
        _, answer = await _send("POST", self._base_url + _REPORT_PATH, self._api_key, body, timeout=_ENDED_TIMEOUT_S)
        self._read_answer("call.ended", answer)

    def _recording(self) -> dict[str, Any]:
        """The recording sent with a production call: one set with set_recording_url(), else SuperBryn's stereo one
        (both URLs name its one file); a URL over the server's limit is left out rather than cut."""
        url = self._external_recording or self._recording_url
        if not url:
            return {}
        if len(url) > _MAX_URL:
            logger.warning("SUPERBRYN_CALL_REPORT_ERROR: recording URL over %d characters left out", _MAX_URL)
            return {}
        if self._external_recording:
            return {"recording_url": url, "recording_url_source": "external"}
        return {"recording_url": url, "stereo_recording_url": url, "recording_url_source": "superbryn_s3"}

    def _context(self) -> Optional[dict[str, Any]]:
        """Call-level facts: the first agent's prompt, the room and the caller, when ringing began, the versions
        running, and the session's error when it closed on one."""
        identity = _attr(self._caller, "identity") or _first_identity(self._ctx.room)
        ref = prompt_ref(self._prompt, self._prompt_version) if self._prompt else None
        return _compact({
            "system_prompt": _cut_utf16(self._system_prompt, _MAX_SYSTEM_PROMPT) if self._system_prompt else None,
            "prompt_ref": {k: _cut(v, _MAX_NAME) for k, v in ref.items()} if ref else None,
            "room_name": _cut(self._room_name, _MAX_NAME) or None,
            "participant_identity": _cut(identity, _MAX_NAME) if isinstance(identity, str) and identity else None,
            "ring_started_at": _iso(self._ring_ts) if self._ring_ts else _ring_started_at(self._ctx),
            "environment": _environment(),
            "close": self._close_error,
        }) or None

    def _on_close(self, event: Any) -> None:
        """Keep the session's close reason (a CloseReason value) for ended_reason, its time, and its error when it
        closed on one; never raises."""
        try:
            reason = getattr(event, "reason", None)
            if reason is not None:
                self._close_reason = str(getattr(reason, "value", reason))
            created = getattr(event, "created_at", None)
            if self._closed_at is None and isinstance(created, (int, float)) and math.isfinite(created):
                self._closed_at = float(created)
            error = getattr(event, "error", None)
            if error is not None:
                self._close_error = _compact({"error_type": _name(_attr(error, "type")) or _name(type(error).__name__),
                                              "error_message": _error_message(error)})
        except Exception as e:  # noqa: BLE001
            logger.debug("SUPERBRYN_CALL_REPORT: unreadable close event: %s", e)

    def _on_dtmf(self, dtmf: Any) -> None:
        """Keep each key the caller pressed, and when; never raises."""
        try:
            digit = getattr(dtmf, "digit", None)
            if isinstance(digit, str) and digit and len(self._dtmf) < _MAX_EVENTS:
                self._dtmf.append((time.time(), digit))
        except Exception as e:  # noqa: BLE001
            logger.debug("SUPERBRYN_CALL_REPORT: unreadable keypress: %s", e)

    def _ended_at(self) -> str:
        """When the session closed, else now; never before the start."""
        if self._closed_at is None:
            return _now_iso()
        return _iso(max(self._closed_at, self._started_ts or self._closed_at))

    def _sample(self, _event: Any = None) -> None:
        """Remember the first agent's prompt, the tools the running agent offers (MCP ones only exist while it runs)
        and its turn-taking; runs on every conversation item and agent state change, so it stays a cheap walk; never
        raises."""
        try:
            activity, agent = _attr(self._session, "_activity"), _attr(self._session, "current_agent")
            if agent is not None and not self._agent_seen:
                # the first agent's instructions as rendered when it became active; later agents' are not sent
                self._agent_seen = True
                self._system_prompt = _section("system_prompt", _rendered, _attr(agent, "instructions"))
            for name, description in _tool_infos([*(_attr(activity, "tools") or ()),
                                                  *(_attr(self._session, "tools") or ()),
                                                  *(_attr(agent, "tools") or ())]):
                self._tools_seen.setdefault(name, description)
            self._turn_detection = _live_turn_detection(activity) or self._turn_detection
            self._interruptions = _live_interruptions(activity) or self._interruptions
        except Exception as e:  # noqa: BLE001
            if not self._sample_failed:
                self._sample_failed = True
                logger.warning("SUPERBRYN_CALL_REPORT_ERROR: could not read tools or turn detection: %s", e)

    def _event(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Contract body for call.<kind>; every event repeats the start fields."""
        return {"schema_version": _SCHEMA_VERSION, "event_id": f"lk:{self._call_id}:{kind}", "event": f"call.{kind}",
                "surface": self._surface, "source": "livekit", "call_id": self._call_id, **self._start_fields,
                **fields}

    def _read_answer(self, event: str, answer: Optional[dict[str, Any]]) -> None:
        if answer is None:
            return
        if isinstance(answer.get("is_test"), bool):
            self._is_test = answer["is_test"]
        said = " ".join(f"{key}={answer[key]}" for key in ("status", "ok", "is_test", "reason", "dropped") if key in answer)
        logger.info("SUPERBRYN_CALL_REPORT_SENT: %s (%s) for call %s: %s", event, self._surface, self._call_id, said)


async def _http_request(method: str, url: str, headers: dict[str, str],
                        body: Optional[dict[str, Any]]) -> tuple[int, Any]:
    """Default transport (aiohttp); tests replace it. Returns (status, parsed JSON or None)."""
    data = None if body is None else json.dumps(body, allow_nan=False)  # NaN / Infinity are not JSON
    async with aiohttp.ClientSession() as http:
        async with http.request(method, url, data=data, headers=headers) as resp:
            try:
                data = await resp.json(content_type=None)
            except ValueError:
                data = None
            return resp.status, data


async def _send(method: str, url: str, api_key: str, body: Optional[dict[str, Any]], *,
                timeout: float, refusal: int = logging.ERROR) -> tuple[Optional[int], Optional[dict[str, Any]]]:
    """Up to 3 attempts with the same body (so the same event_id): (status, the JSON of a 2xx);
    the status is None when no answer came, and there is no JSON when it was refused (logged at refusal)."""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    for attempt in range(_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_BACKOFF_S[attempt - 1])
        try:
            status, data = await asyncio.wait_for(_http_request(method, url, headers, body), timeout)
        except Exception as e:  # noqa: BLE001 - timeouts and network errors are retried
            logger.warning("SUPERBRYN_CALL_REPORT_RETRY: %s %s attempt %d: %r", method, url, attempt + 1, e)
            continue
        if 200 <= status < 300:
            return status, data if isinstance(data, dict) else {}
        if status not in (408, 429) and status < 500:
            logger.log(refusal, "SUPERBRYN_CALL_REPORT_REJECTED: %s %s -> %s %s%s", method, url, status, data,
                       _refusal_hint(status, data))
            return status, None
        logger.warning("SUPERBRYN_CALL_REPORT_RETRY: %s %s attempt %d -> %s", method, url, attempt + 1, status)
    logger.error("SUPERBRYN_CALL_REPORT_FAILED: %s %s gave up", method, url)
    return None, None


def _refusal_hint(status: int, data: Any) -> str:
    """What to change when SuperBryn refused a report."""
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if code == "agent_id_required":
        return " (set SUPERBRYN_AGENT_ID to the agent's SuperBryn ID, or use the agent's own API key)"
    if code in ("agent_key_mismatch", "agent_outside_project"):
        return " (SUPERBRYN_AGENT_ID is not this key's agent, or not in the key's organization)"
    if status in (401, 403):
        return " (a SuperBryn test call needs a key with test_data:write; check the key is valid)"
    return ""


async def _wait_for_caller(room: Any) -> Any:
    """The room's SIP participant, polling up to 15 s for one to join; a web caller (a standard participant) when
    one is there and no SIP one is; None if neither joins."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SIP_WAIT_S
    kinds = (rtc.ParticipantKind.PARTICIPANT_KIND_SIP, rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD)
    while True:
        participants = list(getattr(room, "remote_participants", {}).values())
        for kind in kinds:
            found = next((p for p in participants if getattr(p, "kind", None) == kind), None)
            if found is not None:
                return found
        if loop.time() >= deadline:
            return None
        await asyncio.sleep(_POLL_S)


async def _wait_for_answer(room: Any, caller: Any) -> tuple[bool, Optional[float]]:
    """An outbound phone call's answer: (True, when sip.callStatus turned active), or (False, None) when the caller
    left, or the line went to hangup, before it connected. A call already connected counts from now (attach ran after the
    answer); one with no call status (an older SIP service) from when the caller joined: (True, None)."""
    status = (_attr(caller, "attributes") or {}).get("sip.callStatus")
    if not status:
        return True, None
    if status in _ANSWERED:
        return True, time.time()
    answered: list[float] = []

    def on_change(changed: Any, participant: Any) -> None:
        if participant is caller and (changed or {}).get("sip.callStatus") in _ANSWERED and not answered:
            answered.append(time.time())

    try:
        room.on("participant_attributes_changed", on_change)
    except Exception:  # noqa: BLE001 - polled instead
        pass
    try:
        while True:
            status = (_attr(caller, "attributes") or {}).get("sip.callStatus")
            if answered or status in _ANSWERED:
                return True, answered[0] if answered else time.time()
            left = _attr(caller, "identity") not in (_attr(room, "remote_participants") or {})
            if left or status == "hangup" or _attr(caller, "disconnect_reason") is not None:
                return False, None
            await asyncio.sleep(_POLL_S)
    finally:
        with contextlib.suppress(Exception):
            room.off("participant_attributes_changed", on_change)


def _not_connected(caller: Any, answered: bool) -> Optional[str]:
    """Why a phone call never connected, as LiveKit names it ("user_unavailable": no answer, "user_rejected",
    "sip_trunk_failure"), else "not_answered"; None for a call that connected."""
    status = (_attr(caller, "attributes") or {}).get("sip.callStatus")
    if answered and status not in ("dialing", "ringing"):
        return None
    left = _attr(caller, "disconnect_reason")
    for name in _NOT_CONNECTED:
        if left is not None and left == _attr(_LEFT, name):
            return name.lower()
    return "not_answered"


def _joined_at(participant: Any) -> Optional[float]:
    """When the participant joined the room, as LiveKit stamped it; None when unknown or implausible."""
    joined = _attr(participant, "joined_at")
    stamp = joined.timestamp() if isinstance(joined, datetime) else None
    return stamp if _plausible(stamp) and stamp <= time.time() + 60 else None


def _first_identity(room: Any) -> Optional[str]:
    """The first remote participant's identity, when no caller was recognised."""
    for participant in list((_attr(room, "remote_participants") or {}).values()):
        return _attr(participant, "identity")
    return None


def _ring_started_at(ctx: Any) -> Optional[str]:
    """When the room was created, i.e. ringing began: the connected room's time, else the job's; None when unknown."""
    created = _attr(ctx.room, "creation_time")
    job_room = _attr(_attr(ctx, "job"), "room")
    ms = _attr(job_room, "creation_time_ms")
    stamps = (created.timestamp() if isinstance(created, datetime) else None,
              ms / 1000 if _is_num(ms) else None, _attr(job_room, "creation_time"))
    return next((_iso(t) for t in stamps if _plausible(t)), None)


def _plausible(stamp: Any) -> bool:
    """A time the server takes: between the years 2000 and 2100."""
    return _is_num(stamp) and _EARLIEST_S <= stamp < _LATEST_S


def _environment() -> dict[str, str]:
    """The versions this agent runs: this package, livekit-agents, livekit-rtc (the "livekit" package) and Python."""
    from . import __version__

    def version(package: str) -> Optional[str]:
        try:
            return _name(importlib_metadata.version(package))
        except Exception:  # noqa: BLE001
            return None

    return _compact({"livekit_evals_version": __version__, "livekit_agents_version": version("livekit-agents"),
                     "livekit_rtc_version": version("livekit"), "python_version": platform.python_version()})


def _rendered(instructions: Any) -> Optional[str]:
    """An agent's instructions as LiveKit renders them for a voice call (livekit-agents 1.8's Instructions), else
    the text itself."""
    if instructions is not None and not isinstance(instructions, str) and callable(_attr(instructions, "render")):
        instructions = instructions.render(modality="audio")
    return instructions if isinstance(instructions, str) and instructions.strip() else None


def _custom_data(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """custom_data as given, as strict JSON: a value JSON can't hold (a datetime, a set) goes as its text, as 0.3.x
    sent it; None when empty, or when it still can't be sent (NaN), with one warning line."""
    if not data:
        return None
    try:
        return json.loads(json.dumps(data, default=str, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as e:
        logger.warning("SUPERBRYN_CALL_REPORT_ERROR: custom_data left out of call.ended: %s", e)
        return None


def _ended_by(close_reason: Optional[str], caller: Any) -> Optional[str]:
    """Who ended the call: the agent (its code closed the session or finished its task, or deleted the room, as
    agents hang up), the caller (hung up, turned the call down or didn't answer), else the system (an error, a trunk
    failure, the job shutting down); None when unknown."""
    left = _attr(caller, "disconnect_reason")
    if left is not None and left in (_attr(_LEFT, "USER_UNAVAILABLE"), _attr(_LEFT, "USER_REJECTED")):
        return "caller"
    if left is not None and left == _attr(_LEFT, "SIP_TRUNK_FAILURE"):
        return "system"
    if close_reason in ("user_initiated", "task_completed"):
        return "agent"
    if close_reason == "error":
        return "system"
    if left is not None and left == _attr(_LEFT, "ROOM_DELETED"):
        return "agent"
    hung_up = left is not None and left == _attr(_LEFT, "CLIENT_INITIATED")
    if close_reason == "participant_disconnected" or hung_up:
        return "caller"
    return "system" if close_reason == "job_shutdown" else None


def _test_id(ctx: Any, attributes: dict[str, str]) -> Optional[str]:
    """The first test ID of at most 200 UTF-16 units (the server refuses longer ones); a longer one is ignored."""
    too_long = False
    for value in (*_metadata_test_ids(ctx), *_attribute_test_ids(attributes)):
        if _cut_utf16(value, _MAX_ID) == value:
            return value
        too_long = True
    if too_long:
        logger.warning("SUPERBRYN_CALL_REPORT_ERROR: superbryn_call_id over %d characters ignored", _MAX_ID)
    return None


def _metadata_test_ids(ctx: Any) -> Iterator[str]:
    """superbryn_call_id from the JSON of the job metadata or the room metadata."""
    job = getattr(ctx, "job", None)
    for raw in (getattr(job, "metadata", None), getattr(ctx.room, "metadata", None),
                getattr(getattr(job, "room", None), "metadata", None)):
        try:
            data = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            continue
        value = data.get("superbryn_call_id") if isinstance(data, dict) else None
        if isinstance(value, str) and value.strip():
            yield value.strip()


def _attribute_test_ids(attributes: dict[str, str]) -> Iterator[str]:
    """Test ID from a SIP header attribute (trunk include_headers or headers_to_attributes)."""
    for key, value in attributes.items():
        if key.lower().endswith(_TEST_ID_KEY_SUFFIXES) and isinstance(value, str) and value.strip():
            yield value.strip()


async def _is_caller_number(number: Optional[str], base_url: str, api_key: str) -> bool:
    """True when the remote SIP number is one SuperBryn tests call from or that customers dial."""
    if not number:
        return False
    numbers = await _caller_numbers(base_url, api_key)
    return any(_same_number(number, n) for n in numbers or ())


async def _caller_numbers(base_url: str, api_key: str) -> Optional[list[str]]:
    """SuperBryn's test numbers, cached for an hour per key in memory and in a temp file that every job on this host
    reads (LiveKit runs each job in a new process)."""
    global _numbers_refused_logged
    cache_key = _numbers_key(base_url, api_key)
    hit = _caller_numbers_cache.get(cache_key)
    if hit and time.monotonic() - hit[0] < _CALLER_NUMBERS_TTL_S:
        return hit[1]
    shared = _read_numbers_file(base_url, api_key)
    if shared is not None:
        age, numbers = shared
        _caller_numbers_cache[cache_key] = (time.monotonic() - age, numbers)
        return numbers
    # a key refused the numbers (not a valid key) gets one warning, not an error a call
    status, answer = await _send("GET", base_url + _CALLER_NUMBERS_PATH, api_key, None, timeout=_STARTED_TIMEOUT_S,
                                 refusal=logging.DEBUG)
    numbers = answer.get("numbers") if answer else None
    if not isinstance(numbers, list):
        if status is not None and answer is None and not _numbers_refused_logged:
            _numbers_refused_logged = True
            logger.warning("SUPERBRYN_CALL_REPORT_WARNING: SuperBryn's caller numbers were refused (HTTP %s), so its "
                           "phone tests can't be told from real calls; check the API key", status)
        # a refused key stays refused: asked again in an hour; an outage in a minute
        refused = status is not None
        _remember_numbers(base_url, api_key, [], age=0.0 if refused else _CALLER_NUMBERS_TTL_S - _CALLER_NUMBERS_RETRY_S)
        return None
    numbers = [n for n in numbers if isinstance(n, str)]
    _remember_numbers(base_url, api_key, numbers)
    return numbers


def _numbers_key(base_url: str, api_key: str) -> str:
    """The cache's name for this API and key: another key on the host (maybe refused) keeps its own."""
    return hashlib.sha256(f"{base_url}\n{api_key}".encode()).hexdigest()[:16]


def _numbers_file(base_url: str, api_key: str) -> str:
    """This host's caller-number file for base_url and the key, in the temp dir."""
    return os.path.join(tempfile.gettempdir(), f"superbryn_caller_numbers_{_numbers_key(base_url, api_key)}.json")


def _read_numbers_file(base_url: str, api_key: str) -> Optional[tuple[float, list[str]]]:
    """(age in seconds, numbers) from the file while under an hour old; None when missing, stale or unreadable."""
    try:
        path = _numbers_file(base_url, api_key)
        info = os.lstat(path)
        # only our own regular file: no link, pipe or device that could block, nor another user's numbers
        if not stat.S_ISREG(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        age, numbers = time.time() - data["fetched_at"], data["numbers"]
        if 0 <= age < _CALLER_NUMBERS_TTL_S and isinstance(numbers, list):
            return age, [n for n in numbers if isinstance(n, str)]
    except Exception:  # noqa: BLE001 - fetched instead
        pass
    return None


def _remember_numbers(base_url: str, api_key: str, numbers: list[str], age: float = 0.0) -> None:
    """Cache the numbers (a miss is []) in memory and in the file, to expire an hour after age; never raises."""
    _caller_numbers_cache[_numbers_key(base_url, api_key)] = (time.monotonic() - age, numbers)
    tmp = None
    try:
        path = _numbers_file(base_url, api_key)
        fd, tmp = tempfile.mkstemp(prefix=".superbryn_caller_numbers_", dir=os.path.dirname(path))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time() - age, "numbers": numbers}, f)
        os.replace(tmp, path)  # atomic: a reader sees the old file or the new one, never half of one
    except Exception as e:  # noqa: BLE001 - the in-memory cache still has them
        logger.debug("SUPERBRYN_CALL_REPORT: caller numbers not cached on disk: %s", e)
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _e164(number: Optional[str]) -> Optional[str]:
    """A phone number with its +: LiveKit can give the trunk number as digits only."""
    return f"+{number}" if number and re.fullmatch(r"\d{8,15}", number) else number


def _same_number(a: Optional[str], b: Optional[str]) -> bool:
    """Equal digits, or equal last 10 digits when both have at least 10 (as the server matches)."""
    da, db = re.sub(r"\D", "", a or ""), re.sub(r"\D", "", b or "")
    if not da or not db:
        return False
    return da == db or (len(da) >= 10 and len(db) >= 10 and da[-10:] == db[-10:])


async def _room_sid(ctx: Any) -> str:
    """Room SID from the job, else from the room (a plain property, or awaitable in livekit-rtc 1.x)."""
    sid = getattr(getattr(getattr(ctx, "job", None), "room", None), "sid", None)
    if sid:
        return sid
    value = getattr(ctx.room, "sid", None)
    if inspect.isawaitable(value):
        try:
            # shield: cancelling a timed-out wait must not cancel the room's own SID future
            value = await asyncio.wait_for(asyncio.shield(value), _SID_WAIT_S)
        except Exception:  # noqa: BLE001
            return ""
    return value if isinstance(value, str) else ""


def _resolve_agent_id(explicit: Optional[str]) -> Optional[str]:
    """agent_id= or SUPERBRYN_AGENT_ID when it is a SuperBryn agent ID (a UUID), else None with one warning: the
    server refuses a whole report with any other value. Never AGENT_ID, which 0.3.x read as a free-form name."""
    global _bad_agent_logged
    value = explicit or os.getenv("SUPERBRYN_AGENT_ID")
    agent = _agent_uuid(value)
    if value and not agent and not _bad_agent_logged:
        _bad_agent_logged = True
        logger.warning("SUPERBRYN_CALL_REPORT_WARNING: agent_id %r is not a SuperBryn agent ID (a UUID), so it is "
                       "not sent", value)
    return agent


def _agent_uuid(value: Optional[str]) -> Optional[str]:
    """The agent ID when the server's z.uuid() accepts it (RFC 4122 variant, version 1-8), else None."""
    if not value:
        return None
    try:
        parsed = uuid.UUID(value)
        if parsed.variant == uuid.RFC_4122 and 1 <= parsed.version <= 8:
            return str(parsed)
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def _ms(at: float, base: float) -> int:
    """A moment as whole ms from base (started_at), never below 0."""
    return max(0, int(round((at - base) * 1000)))


def _eou_events(events: list[Any], base: float) -> list[dict[str, Any]]:
    """Each caller turn the turn detector closed (LiveKit's EOU metrics): when, its end-of-turn wait and its
    transcript wait; both waits are left out when LiveKit saw no end of speech (it reports 0 for them then)."""
    out = []
    for event in events:
        metrics = _attr(event, "metrics") if _attr(event, "type") == "metrics_collected" else None
        stamp = _attr(metrics, "timestamp")
        if _attr(metrics, "type") != "eou_metrics" or not (_is_num(stamp) and stamp > 0):
            continue
        waited = _attr(metrics, "end_of_utterance_delay")
        seen = _is_num(waited) and waited > 0
        out.append(_compact({
            "at_ms": _ms(stamp, base),
            "eou_delay_ms": _millis(waited) if seen else None,
            "transcription_delay_ms": _millis(_attr(metrics, "transcription_delay")) if seen else None,
        }))
    return out[:_MAX_EVENTS]


def _dtmf(presses: list[tuple[float, str]], base: float) -> list[dict[str, Any]]:
    """The keys the caller pressed, in order, ms from base."""
    return [{"at_ms": _ms(at, base), "digit": digit[:_MAX_DIGIT]} for at, digit in presses]


def _vendor(model: Optional[str], component: Any) -> Optional[str]:
    """The vendor behind a failing model: LiveKit Inference's "vendor/model", Bedrock's "amazon.", a known part of the
    model's name, else the plugin package it comes from (livekit.plugins.<vendor>)."""
    name = (model or "").lower()
    vendor, slash, _ = name.partition("/")
    if slash and vendor in _KNOWN_VENDORS:
        return vendor
    if name.startswith("amazon."):
        return "aws"
    found = next((vendor for vendor, parts in _MODEL_VENDORS if any(part in name for part in parts)), None)
    if found:
        return found
    module = (getattr(type(component), "__module__", "") or "").split(".")
    return module[2] if len(module) > 2 and module[:2] == ["livekit", "plugins"] else None


def _history_payload(items: list[Any], base: float,
                     speech: Optional[list[tuple[float, Optional[float]]]] = None, stand_in_confidence: bool = False,
                     caller_times: bool = True, closed_at: Optional[float] = None,
                     caller_states: Optional[list[tuple[float, Optional[float]]]] = None,
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Contract turns and tool calls from the history items; ms are from base (started_at). An agent line names the
    provider's request ID for it (LiveKit 1.8's message metrics). speech is set for a realtime model: caller lines take their times from it. LiveKit's 1.0 stand-in confidence is
    left out for a realtime model and when stand_in_confidence is set (LiveKit Inference STT). Without caller_times
    (a realtime model whose caller times are stand-ins), caller lines carry no times. A caller line ending near
    closed_at is checked against caller_states (_stop_before_close)."""
    def ms(t: Any) -> Optional[int]:
        return _ms(t, base) if _is_num(t) and t > 0 else None

    outputs = {o.call_id: o for o in items if getattr(o, "type", None) == "function_call_output"}
    transcript: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    last_agent: Optional[str] = None  # cleared by a user turn; tool calls without one take the next agent turn
    waiting: list[dict[str, Any]] = []
    for item in items:
        kind = getattr(item, "type", None)
        if kind == "message" and getattr(item, "role", None) in ("user", "assistant"):
            text = _message_text(item)
            if text:
                metrics = _metrics(item)
                user = item.role == "user"
                turn_id = _turn_id(item)
                confidence = getattr(item, "transcript_confidence", None)
                start = metrics.get("started_speaking_at") or getattr(item, "created_at", None)
                end = metrics.get("stopped_speaking_at")
                if user and speech is not None:
                    start, end = _caller_span(start, end, speech) if caller_times else (None, None)
                if user and _is_num(closed_at) and _is_num(end) and abs(closed_at - end) <= _CLOSE_SLACK_S:
                    end = _stop_before_close(start, end, caller_states or [])
                if user and (speech is not None or stand_in_confidence) and confidence == 1.0:
                    confidence = None
                transcript.append(_compact({
                    "id": turn_id,
                    "role": "user" if user else "agent",
                    "text": _cut_utf16(text, _MAX_TEXT),
                    "start_ms": ms(start),
                    "end_ms": ms(end),
                    "confidence": confidence if user and _is_num(confidence) and 0 <= confidence <= 1 else None,
                    "interrupted": None if user else bool(getattr(item, "interrupted", False)),
                    "provider_request_id": None if user else _provider_request_id(metrics),
                }))
                if not user:
                    for call in waiting:
                        call.update(_compact({"turn_id": turn_id}))
                    waiting.clear()
                last_agent = None if user else turn_id
        elif kind == "function_call":
            name = getattr(item, "name", None)
            if not (isinstance(name, str) and name):  # the server needs a name; a nameless call is left out
                continue
            output = outputs.get(item.call_id)
            tool_calls.append(_compact({
                "id": _cut(item.call_id, _MAX_ID),
                "turn_id": last_agent,
                "name": _cut(name, _MAX_NAME),
                "arguments": _parse_json(item.arguments),
                "result": getattr(output, "output", None),
                "is_error": bool(output.is_error) if output is not None else None,
                "start_ms": ms(getattr(item, "created_at", None)),
                "end_ms": ms(getattr(output, "created_at", None)),
            }))
            if last_agent is None:
                waiting.append(tool_calls[-1])
    return transcript[:_MAX_TURNS], tool_calls[:_MAX_TOOL_CALLS]


def _provider_request_id(metrics: dict[str, Any]) -> Optional[str]:
    """The first provider request ID LiveKit keeps on a reply, cut to the server's limit."""
    ids = metrics.get("provider_request_ids")
    first = ids[0] if isinstance(ids, (list, tuple)) and ids else None
    return _cut_utf16(first, _MAX_ID) if isinstance(first, str) and first else None


def _turn_id(item: Any) -> Optional[str]:
    """The history item's id as a contract turn id (1-200 UTF-16 units), or None."""
    value = getattr(item, "id", None)
    return _cut_utf16(value, _MAX_ID) if isinstance(value, str) and value else None


def _diagnostics(session: Any, items: list[Any], transcript: list[dict[str, Any]], tools: dict[str, Optional[str]],
                 turn_detection: Optional[dict[str, Any]] = None,
                 llm_requests: Optional[list[dict[str, Any]]] = None,
                 errors: Optional[list[dict[str, Any]]] = None,
                 requests: Optional[list[tuple[float, Any]]] = None,
                 speech: Optional[list[tuple[float, Optional[float]]]] = None,
                 interruptions: Optional[dict[str, Any]] = None,
                 eou_events: Optional[list[dict[str, Any]]] = None,
                 dtmf: Optional[list[dict[str, Any]]] = None) -> Optional[dict[str, Any]]:
    """call.ended diagnostics, section by section: a section that fails is left out alone. turn_detection and
    interruptions are what the running agent used, when seen during the call; else they come from the session's
    options. speech is set for a realtime model, whose reply waits come from it and the requests."""
    usage = _section("usage", list, _attr(_attr(session, "usage"), "model_usage") or ()) or []
    options = _attr(session, "options")
    pipeline = _section("pipeline", _pipeline, session)
    configuration = {kind: _section(kind, _component, session, kind, usage) for kind in ("stt", "llm", "tts")}
    configuration["turn_detection"] = turn_detection or _section("turn_detection", _turn_detection, options)
    configuration["interruptions"] = interruptions or _section("interruptions", _interruptions, options)
    if pipeline == "realtime":
        # the model's own speech detection interrupts it: LiveKit's duration and word gates don't apply
        configuration["interruptions"] = _compact({"enabled": (configuration["interruptions"] or {}).get("enabled")})
        configuration["tts"] = configuration["tts"] or _section("tts", _realtime_voice, session, configuration["llm"])
    configuration["tools"] = [_compact({"name": name, "description": description})
                              for name, description in list(tools.items())[:_MAX_TOOLS]]
    turn_ids = {turn["id"] for turn in transcript if "id" in turn}
    diagnostics = {
        "pipeline": pipeline,
        "configuration": {k: v for k, v in configuration.items() if v},
        "turn_latencies": (_section("turn_latencies", _turn_latencies, items, turn_ids) if speech is None else
                           _section("turn_latencies", _realtime_turn_latencies, items, turn_ids, requests or [],
                                    speech, _caller_times_real(session))),
        "usage": _section("usage", _usage, usage),
        "llm_requests": llm_requests,
        "errors": errors,
        "eou_events": eou_events,
        "dtmf": dtmf,
    }
    diagnostics = {k: _strict_json(k, v) for k, v in diagnostics.items() if _kept(k, v)}
    return {k: v for k, v in diagnostics.items() if _kept(k, v)} or None


def _kept(name: str, value: Any) -> bool:
    """A diagnostics section worth sending: anything non-empty, and an empty errors list, which says the session
    reported no failure (an error capture that failed is None and stays out, so SuperBryn reads it as not sent)."""
    return bool(value) or (name == "errors" and value == [])


def _section(name: str, build: Callable[..., Any], *args: Any) -> Any:
    """build(*args), or None with one warning line when it raises."""
    try:
        return build(*args)
    except Exception as e:  # noqa: BLE001
        logger.warning("SUPERBRYN_CALL_REPORT_ERROR: %s left out of call.ended: %s", name, e)
        return None


def _pipeline(session: Any) -> Optional[str]:
    """realtime for a RealtimeModel, stt_llm_tts for any other LLM, None without an LLM."""
    model = _running(session, "llm")
    if not model:
        return None
    return "realtime" if isinstance(model, llm.RealtimeModel) else "stt_llm_tts"


def _running(session: Any, kind: str) -> Any:
    """The current agent's own stt, llm or tts when it has one (LiveKit runs that one), else the session's."""
    return _attr(_attr(session, "current_agent"), kind) or _attr(session, kind)


def _component(session: Any, kind: str, usage: list[Any]) -> dict[str, str]:
    """Provider and model of the running stt, llm or tts (and the TTS voice); session.usage fills the gaps, and
    names the LLM when most tokens went to another model (a handoff, a fallback)."""
    base = _unwrap(_running(session, kind), kind)
    entries = _of_type(usage, f"{kind}_usage")
    fields = {key: _first_name(_attr(base, key), *(_attr(e, key) for e in entries))
              for key in ("provider", "model")}
    # a realtime model is the model that ran; usage may name its helper model (GPT Live's delegated LLM)
    top = max(entries, key=_tokens, default=None) if kind == "llm" and not isinstance(base, llm.RealtimeModel) else None
    if top is not None and _tokens(top) > 0 and _name(_attr(top, "model")) not in (None, fields["model"]):
        fields = {key: _name(_attr(top, key)) for key in ("provider", "model")}
    if kind == "tts":
        opts = _attr(base, "_opts")
        fields["voice"] = _first_name(_attr(base, "voice_id"), _attr(base, "voice"),
                                      _attr(opts, "voice_id"), _attr(opts, "voice"))
    if kind == "stt":
        # the language the STT was set to ("multi", "hi", "en-US"), when it says
        fields["language"] = _first_name(_attr(_attr(base, "_opts"), "language"), _attr(base, "language"))
    return _compact(fields)


# Realtime models whose own speech detection reports when the caller really stopped (OpenAI's server VAD, xAI's).
# Others mark the caller stopped only once their reply starts (Nova Sonic, Gemini Live) or are unverified.
_REAL_CALLER_STOPS = ("gpt-realtime", "gpt-4o-realtime", "gpt-4o-mini-realtime", "grok")


def _caller_times_real(session: Any) -> bool:
    """Whether a realtime session's caller start and stop times are real: with a VAD the code gave the agent or the
    session they are LiveKit's own; without, only for the models in _REAL_CALLER_STOPS. Quietly False when the model
    can't be read."""
    return _given_vad(session) or _model_times_real(session)


def _model_times_real(session: Any) -> bool:
    """Whether the realtime model is one in _REAL_CALLER_STOPS, whose own response timings are real too; the others
    report a ttft of -1, 0 or a few ms. Quietly False when its name can't be read."""
    try:
        model = _component(session, "llm", []).get("model") or ""
    except Exception:  # noqa: BLE001
        model = ""
    return isinstance(model, str) and any(stop in model.lower() for stop in _REAL_CALLER_STOPS)


_UNSET = object()


def _given_vad(session: Any) -> bool:
    """Whether a VAD the code passed in runs, as LiveKit picks it: the agent's own when given, else the session's.
    LiveKit 1.8 fills in a default VAD when none is passed (_using_default_vad), which realtime models with their own
    turn detection never use, so that one doesn't count."""
    agent_vad = getattr(_attr(session, "current_agent"), "vad", _UNSET)
    if agent_vad is not _UNSET and type(agent_vad).__name__ != "NotGiven":
        return agent_vad is not None
    return _attr(session, "vad") is not None and not _attr(session, "_using_default_vad")


def _inference_stt(session: Any) -> bool:
    """Whether the running STT is LiveKit Inference; quietly False when it can't be read (its section warns)."""
    try:
        return _component(session, "stt", []).get("provider") == "livekit"
    except Exception:  # noqa: BLE001
        return False


def _realtime_voice(session: Any, model: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
    """A realtime model speaks for itself: its provider and model, with the voice it was given; None without one."""
    base = _unwrap(_running(session, "llm"), "llm")
    voice = _first_name(_attr(_attr(base, "_opts"), "voice"), _attr(base, "voice"))
    return _compact({**(model or {}), "voice": voice}) if voice else None


def _unwrap(component: Any, kind: str) -> Any:
    """The provider plugin behind adapters and wrappers (a fallback list's first); the component itself if none."""
    seen: set[int] = set()
    current = component
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (getattr(type(current), "__module__", "") or "").startswith("livekit.plugins."):
            return current
        inner = None
        for name in _WRAPPED_ATTRS[kind]:
            value = getattr(current, name, None)  # a raising wrapper fails its section, with a warning
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None and value is not current:
                inner = value
                break
        if inner is None:
            return current
        current = inner
    return current


def _turn_detection(options: Any) -> dict[str, Any]:
    """Turn detector and end-of-turn delays from turn_handling, or the older option fields."""
    handling = _attr(options, "turn_handling")
    if isinstance(handling, dict):
        detector, endpointing = handling.get("turn_detection"), handling.get("endpointing") or {}
        min_s, max_s = endpointing.get("min_delay"), endpointing.get("max_delay")
        preemptive = handling.get("preemptive_generation")
    else:
        detector = None
        min_s, max_s = _attr(options, "min_endpointing_delay"), _attr(options, "max_endpointing_delay")
        preemptive = _attr(options, "preemptive_generation")
    return _compact({"type": _detector_name(detector), "min_end_of_turn_ms": _millis(min_s),
                     "max_end_of_turn_ms": _millis(max_s), "preemptive_generation": _preemptive(preemptive)})


def _preemptive(value: Any) -> Optional[bool]:
    """Whether LiveKit drafts a reply before the caller's turn is confirmed (and drops it if they go on): a bool before
    1.8, then turn_handling's options, whose "enabled" says it."""
    if isinstance(value, dict):
        value = value.get("enabled")
    return value if isinstance(value, bool) else None


def _live_turn_detection(activity: Any) -> Optional[dict[str, Any]]:
    """The turn-taking LiveKit's running AgentActivity resolved, the agent's settings over the session's;
    None without one. Before 1.8, a realtime model with server-side turn detection always takes the turns."""
    realtime = _attr(activity, "_rt_turn_detection_enabled")
    if not isinstance(realtime, bool):
        if not (hasattr(activity, "_turn_detection") and hasattr(activity, "min_endpointing_delay")):
            return None
        model = _attr(activity, "llm")
        realtime = isinstance(model, llm.RealtimeModel) and _attr(_attr(model, "capabilities"), "turn_detection") is True
    if realtime:  # the realtime model decides when the caller finished: no endpointing delays apply
        return {"type": "realtime_llm"}
    # the agent's preemptive setting over the session's from 1.8; before, only the session's options carry it
    preemptive = _attr(activity, "preemptive_generation_opts") or _attr(_attr(_attr(activity, "_session"), "options"),
                                                                        "preemptive_generation")
    return _compact({"type": _detector_name(_attr(activity, "_turn_detection")),
                     "min_end_of_turn_ms": _millis(_attr(activity, "min_endpointing_delay")),
                     "max_end_of_turn_ms": _millis(_attr(activity, "max_endpointing_delay")),
                     "preemptive_generation": _preemptive(preemptive)})


def _detector_name(detector: Any) -> Optional[str]:
    """A turn detection mode ("vad", "stt", ...), else the detector's model name, else its class name."""
    if detector is not None and not isinstance(detector, str):
        detector = _name(_attr(detector, "model")) or type(detector).__name__
    return _name(detector)


def _interruptions(options: Any) -> dict[str, Any]:
    """Interruption settings from turn_handling, or the older option fields."""
    handling = _attr(options, "turn_handling")
    if isinstance(handling, dict):
        interruption = handling.get("interruption") or {}
        enabled, min_s, words = (interruption.get(k) for k in ("enabled", "min_duration", "min_words"))
    else:
        enabled, min_s, words = (_attr(options, k) for k in ("allow_interruptions", "min_interruption_duration",
                                                             "min_interruption_words"))
    return _compact({
        "enabled": enabled if isinstance(enabled, bool) else None,
        "min_duration_ms": _millis(min_s),
        "min_words": words if isinstance(words, int) and not isinstance(words, bool) and words >= 0 else None,
    })


def _live_interruptions(activity: Any) -> Optional[dict[str, Any]]:
    """The interruption settings LiveKit's running AgentActivity uses, the agent's over the session's; None without
    one."""
    enabled = _attr(activity, "allow_interruptions")
    if not isinstance(enabled, bool):
        return None
    handling = _attr(_attr(activity, "_agent"), "_turn_handling")
    own = handling.get("interruption") if isinstance(handling, dict) else None
    session_side = _interruptions(_attr(_attr(activity, "_session"), "options"))
    agent_side = _interruptions(SimpleNamespace(turn_handling={"interruption": own})) if isinstance(own, dict) else {}
    return {**session_side, **agent_side, "enabled": enabled}


def _tool_infos(tools: Any, depth: int = 0) -> Iterator[tuple[str, Optional[str]]]:
    """(name, description) of each FunctionTool and RawFunctionTool, looking inside toolsets (MCP)."""
    for tool in list(tools or ()):
        if isinstance(tool, getattr(llm, "Toolset", ())):
            if depth < 3:
                yield from _tool_infos(_attr(tool, "tools"), depth + 1)
            continue
        info = _attr(tool, "info")
        if isinstance(tool, llm.FunctionTool):
            description = _attr(info, "description")
        elif isinstance(tool, llm.RawFunctionTool):
            schema = _attr(info, "raw_schema")
            description = schema.get("description") if isinstance(schema, dict) else None
        else:
            continue
        name = _attr(info, "name")
        if isinstance(name, str) and name:
            has_text = isinstance(description, str) and description
            yield _cut_utf16(name, _MAX_NAME), _cut_utf16(description, _MAX_DESCRIPTION) if has_text else None


def _turn_latencies(items: list[Any], turn_ids: set[str]) -> list[dict[str, Any]]:
    """One entry per assistant message with a timing; the user delays come from the turn it answered."""
    latencies: list[dict[str, Any]] = []
    user: dict[str, Any] = {}
    for item in items:
        role = getattr(item, "role", None) if getattr(item, "type", None) == "message" else None
        if role == "user":
            user = _metrics(item)
        elif role == "assistant":
            agent = _metrics(item)
            transcribed, ended = user.get("transcription_delay"), user.get("end_of_turn_delay")
            # LiveKit's end-of-turn delay includes the wait for the transcript
            endpointing = max(0, ended - transcribed) if _is_num(ended) and _is_num(transcribed) else None
            entry = _compact({
                "transcriber_ms": _millis(transcribed),
                "endpointing_ms": _millis(endpointing),
                "model_ms": _millis(agent.get("llm_node_ttft")),
                "voice_ms": _millis(agent.get("tts_node_ttfb")),
                "turn_ms": _millis(agent.get("e2e_latency")),
            })
            user = {}
            if entry:
                turn_id = _turn_id(item)
                latencies.append({"turn_id": turn_id, **entry} if turn_id in turn_ids else entry)
    return latencies[:_MAX_TURN_LATENCIES]


def _caller_speech(events: list[Any]) -> list[tuple[float, Optional[float]]]:
    """When the caller talked, as (started, stopped) from the session's user states; stopped is None when they were
    still talking at the end."""
    spans: list[tuple[float, Optional[float]]] = []
    started: Optional[float] = None
    for event in events:
        at, state = _attr(event, "created_at"), _attr(event, "new_state")
        if _attr(event, "type") != "user_state_changed" or not _is_num(at):
            continue
        if state == "speaking" and started is None:
            started = at
        elif state != "speaking" and started is not None:
            spans.append((started, at))
            started = None
    if started is not None:
        spans.append((started, None))
    return spans


def _stop_before_close(start: Any, end: float, states: list[tuple[float, Optional[float]]]) -> Optional[float]:
    """A caller line ending at the session close: LiveKit ends a line the call cut off at the close itself, so it ends
    where the voice detector saw its last speech stop instead (a staging "No." ran 11.5 s, not 0.7 s). Still talking
    at the close, or nothing heard: no end rather than the close. A line whose speech really ran to the close keeps it."""
    spans = [s for s in states if s[0] <= end and (s[1] is None or not _is_num(start) or s[1] > start - _SPEECH_SLACK_S)]
    stop = spans[-1][1] if spans else None
    if stop is None:
        return None
    return stop if end - stop > _CLOSE_SLACK_S else end


def _caller_span(start: Any, end: Any, speech: list[tuple[float, Optional[float]]]) -> tuple[Any, Any]:
    """A realtime caller line's (start, end): the end is when its speech stopped, and a line stamped after the caller
    had already stopped (Gemini stamps its reply's start) moves back to when that speech started."""
    if not _is_num(start):
        return start, end
    span = next((s for s in reversed(speech) if s[0] <= start + _SPEECH_SLACK_S), None)
    if span is None:
        return start, end
    began, stopped = span
    if stopped is not None and stopped <= start:
        start = began
    return start, end if _is_num(end) else stopped


def _realtime_turn_latencies(items: list[Any], turn_ids: set[str], requests: list[tuple[float, Any]],
                             speech: list[tuple[float, Optional[float]]], caller_real: bool = False,
                             ) -> list[dict[str, Any]]:
    """Each realtime reply's wait from the caller stopping: turn_ms to when the agent started speaking, and
    model_ms to the model's first audio. Only when the caller's stops are real (caller_real: a VAD on the
    session, or a model whose speech detection reports them); else none, rather than a wait from a stand-in stop.
    A reply whose response is found by provider ID is paired with the caller's last stop before that response was
    created, and gets both; without the match, with the last stop before it started speaking, turn_ms alone. A
    reply that answers no new caller speech (the greeting, a second reply) gets none."""
    if not caller_real:
        return []
    responses = [(start, m) for start, m in requests if _attr(m, "type") == "realtime_model_metrics"]
    by_id = {_attr(m, "request_id"): (start, m) for start, m in reversed(responses) if _attr(m, "request_id")}
    stops = [stopped for _, stopped in speech if stopped is not None]
    answered: set[float] = set()
    latencies: list[dict[str, Any]] = []
    for item in items:
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            continue
        metrics = _metrics(item)
        heard = metrics.get("started_speaking_at")
        if not _is_num(heard):
            continue
        ids = metrics.get("provider_request_ids") or ()
        response = next((by_id[i] for i in ids if isinstance(i, str) and i in by_id), None)
        stopped = next((t for t in reversed(stops) if t <= (response[0] if response else heard)), None)
        if stopped is None or stopped in answered:
            continue
        answered.add(stopped)
        ttft = _attr(response[1], "ttft") if response else None
        first_audio = response[0] + ttft if response and _is_num(ttft) and ttft > 0 else None  # 0: a placeholder
        entry = _compact({
            "model_ms": _millis(first_audio - stopped) if first_audio is not None else None,
            "turn_ms": _millis(heard - stopped),
        })
        turn_id = _turn_id(item)
        latencies.append({"turn_id": turn_id, **entry} if turn_id in turn_ids else entry)
    return latencies[:_MAX_TURN_LATENCIES]


def _request_log(events: list[Any]) -> list[tuple[float, Any]]:
    """(start time, metrics) of each LLM request and realtime response the session recorded, earliest first. The
    empty realtime record every session opens with (no request_id, tokens or ttft) and xAI's session-close record are
    no responses. Neither are records that share one request ID for the whole session (Nova Sonic's usage pieces, on
    the monotonic clock; GPT Live's): they are stamped as the reply's audio arrives, not when it was asked for, so
    neither a request's start nor what became of it can be told. Their tokens still count in usage."""
    log = []
    seen = Counter(_attr(_attr(e, "metrics"), "request_id") for e in events if _attr(e, "type") == "metrics_collected"
                   and _attr(_attr(e, "metrics"), "type") == "realtime_model_metrics")
    for event in events:
        metrics = _attr(event, "metrics") if _attr(event, "type") == "metrics_collected" else None
        kind, stamp, duration = _attr(metrics, "type"), _attr(metrics, "timestamp"), _attr(metrics, "duration")
        if kind == "realtime_model_metrics":
            request_id = _attr(metrics, "request_id")
            empty = not request_id and not _request_tokens(metrics, True) and not (_is_num(_attr(metrics, "ttft"))
                                                                                  and _attr(metrics, "ttft") >= 0)
            monotonic = _is_num(stamp) and 0 < stamp < _EPOCH_FLOOR
            if empty or request_id == "session_close" or monotonic or (request_id and seen[request_id] > 1):
                continue
        if kind not in ("llm_metrics", "realtime_model_metrics") or not (_is_num(stamp) and stamp > 0):
            continue
        # an LLM request is stamped when it ends; a realtime response when it is created
        ended = kind == "llm_metrics" and _is_num(duration) and duration > 0
        log.append((stamp - duration if ended else stamp, metrics))
    return sorted(log, key=lambda entry: entry[0])


def _llm_requests(log: list[tuple[float, Any]], events: list[Any], items: list[Any],
                  base: float) -> list[dict[str, Any]]:
    """Each request and what became of its answer. An LLM request owns its speech's reply and tool calls from its
    start until that speech's next request; a realtime response names no speech, so it owns the history's."""
    speeches: dict[str, list[Any]] = {}
    for event in events:
        handle = _attr(event, "speech_handle") if _attr(event, "type") == "speech_created" else None
        if isinstance(_attr(handle, "id"), str):
            speeches[handle.id] = list(_attr(handle, "chat_items") or ())

    def pool_key(metrics: Any) -> Any:
        return "realtime" if _attr(metrics, "type") == "realtime_model_metrics" else _attr(metrics, "speech_id")

    ends: list[float] = []
    upcoming: dict[Any, float] = {}
    for start, metrics in reversed(log):
        ends.append(upcoming.get(pool_key(metrics), math.inf))
        upcoming[pool_key(metrics)] = start
    ends.reverse()

    llm_failures = [at for at, step, *_ in _reported_errors(events) if step == "llm"]
    claimed: set[Any] = set()
    requests = []
    for (start, metrics), end in list(zip(log, ends))[:_MAX_LLM_REQUESTS]:
        key = pool_key(metrics)
        pool = items if key == "realtime" else speeches.get(key, []) if isinstance(key, str) else []
        owned = [item for item in pool if _item_key(item) not in claimed
                 and _is_num(_attr(item, "created_at")) and start <= item.created_at < end]
        claimed.update(_item_key(item) for item in owned)
        reply = next((item for item in owned if _attr(item, "type") == "message"
                      and _attr(item, "role") == "assistant"), None)
        ttft = _attr(metrics, "ttft")
        usage = _request_tokens(metrics, key == "realtime")
        if reply is not None:
            outcome = "cut_off" if _attr(reply, "interrupted") else "spoken"
        elif any(_attr(item, "type") == "function_call" for item in owned):
            outcome = "tool_call"
        else:
            # The LLM failed on it (no output, an LLM error while it ran): it never answered, whatever its ttft.
            # its window runs from its start to its end (an LLM request is stamped then; a realtime response lasts
            # its duration from creation), plus the slack
            duration = _attr(metrics, "duration")
            ran = duration if _is_num(duration) and duration > 0 else 0.0
            failed = not usage and any(start <= at <= start + ran + _FAILURE_SLACK_S for at in llm_failures)
            outcome = "not_spoken" if _is_num(ttft) and ttft >= 0 and not failed else "stopped"
        requests.append({"start_ms": max(0, int(round((start - base) * 1000))), "outcome": outcome, **usage})
    return requests


def _reported_errors(events: list[Any]) -> list[tuple[float, str, Any, Any]]:
    """The STT, LLM and TTS failures the session reported, oldest first, each with when it happened and the model
    that failed. A failure stamped on the monotonic clock (Nova Sonic's) is moved onto the wall clock, as its
    responses are."""
    found = []
    clock = time.time() - time.monotonic()
    for event in events:
        error = _attr(event, "error") if _attr(event, "type") == "error" else None
        step = _ERROR_STEPS.get(_attr(error, "type"))
        stamp = _attr(error, "timestamp")
        if _is_num(stamp) and 0 < stamp < _EPOCH_FLOOR:
            stamp += clock
        at = next((t for t in (stamp, _attr(event, "created_at")) if _is_num(t) and t > 0), None)
        if step is not None and at is not None:
            found.append((at, step, error, _attr(event, "source")))
    return sorted(found, key=lambda entry: entry[0])


def _errors(events: list[Any], base: float) -> list[dict[str, Any]]:
    """Each STT, LLM or TTS failure the session reported. LiveKit retries a recoverable one: it recovered unless the
    same step failed for good before it next worked, and end_ms is when it next worked. LiveKit's STT hears the
    caller; the LLM and TTS are the agent's. source_model and source_provider name the model that failed."""
    found = _reported_errors(events)
    clock = time.time() - time.monotonic()
    worked: dict[str, list[float]] = {}  # step -> when it recorded metrics, i.e. worked
    for event in events:
        metrics = _attr(event, "metrics") if _attr(event, "type") == "metrics_collected" else None
        step, stamp = _METRIC_STEPS.get(_attr(metrics, "type")), _attr(metrics, "timestamp")
        if step is not None and _is_num(stamp) and stamp > 0:
            worked.setdefault(step, []).append(stamp + clock if stamp < _EPOCH_FLOOR else stamp)

    def ms(t: float) -> int:
        return max(0, int(round((t - base) * 1000)))

    errors = []
    for at, step, error, source in found[:_MAX_ERRORS]:
        recoverable = _attr(error, "recoverable")
        back = min((t for t in worked.get(step, ()) if t > at), default=None)
        fatal = min((t for t, s, e, _ in found if s == step and t >= at and _attr(e, "recoverable") is False),
                    default=None)
        model = _name(_attr(source, "model"))
        recovered = (recoverable and not (fatal is not None and (back is None or fatal < back))
                     if isinstance(recoverable, bool) else None)
        errors.append(_compact({
            "step": step,
            "side": "caller" if step == "stt" else "agent",
            "start_ms": ms(at),
            "end_ms": ms(back) if recovered and back is not None else None,
            "message": _error_message(error),
            "recovered": recovered,
            "source_model": model,
            "source_provider": _name(_vendor(model, source)),
        }))
    return errors


def _error_message(error: Any) -> Optional[str]:
    """The failure's label and exception as text, with links and long ids (keys, signatures) masked."""
    exc, label = _attr(error, "error"), _name(_attr(error, "label"))
    text = (str(exc).strip() or type(exc).__name__) if exc is not None else ""
    text = f"{label}: {text}" if label and text else text or label or ""
    text = re.sub(r"[a-z][a-z0-9+.-]*://\S+", "<url>", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", re.sub(r"[A-Za-z0-9+/_-]{32,}", "***", text)).strip()
    return _cut_utf16(text, _MAX_ERROR_MESSAGE) or None


def _item_key(item: Any) -> Any:
    item_id = _attr(item, "id")
    return item_id if isinstance(item_id, str) and item_id else id(item)


def _request_tokens(metrics: Any, realtime: bool) -> dict[str, int]:
    """A request's input (cached included), cached and output tokens; none when it reported no usage."""
    if realtime:
        counts = (_attr(metrics, "input_tokens"), _attr(_attr(metrics, "input_token_details"), "cached_tokens"),
                  _attr(metrics, "output_tokens"))
    else:
        counts = (_attr(metrics, "prompt_tokens"), _attr(metrics, "prompt_cached_tokens"),
                  _attr(metrics, "completion_tokens"))
    values = [int(value) if _is_num(value) and value >= 0 else None for value in counts]
    if not any(values):
        return {}
    return _compact(dict(zip(("input_tokens", "cached_input_tokens", "output_tokens"), values)))


def _usage(usage: list[Any]) -> dict[str, int]:
    """LLM token, STT audio and TTS character totals; a field only when entries of its type exist. A realtime model's
    speech and text tokens, which are priced apart, are split out when it reports any speech tokens."""
    llms, stts, ttss = _of_type(usage, "llm_usage"), _of_type(usage, "stt_usage"), _of_type(usage, "tts_usage")
    split = {key: _total(llms, field) for key, field in (
        ("llm_input_audio_tokens", "input_audio_tokens"), ("llm_input_text_tokens", "input_text_tokens"),
        ("llm_output_audio_tokens", "output_audio_tokens"), ("llm_output_text_tokens", "output_text_tokens"))}
    speech = split["llm_input_audio_tokens"] or split["llm_output_audio_tokens"]
    return _compact({
        "llm_input_tokens": _total(llms, "input_tokens") if llms else None,
        "llm_cached_input_tokens": _total(llms, "input_cached_tokens") if llms else None,
        "llm_output_tokens": _total(llms, "output_tokens") if llms else None,
        **(split if speech else {}),
        # OpenAI's realtime transcription bills tokens and reports 0 s of audio: 0 is no measure
        "stt_audio_ms": (_total(stts, "audio_duration", 1000) or None) if stts else None,
        "tts_characters": _total(ttss, "characters_count") if ttss else None,
    })


def _of_type(usage: list[Any], kind: str) -> list[Any]:
    return [e for e in usage if _attr(e, "type") == kind]


def _tokens(entry: Any) -> int:
    """An LLM usage entry's input plus output tokens."""
    return _total([entry], "input_tokens") + _total([entry], "output_tokens")


def _total(entries: list[Any], field: str, scale: int = 1) -> int:
    """A usage field summed over entries (non-numbers and negatives skipped), times scale, as an int."""
    return int(round(sum(v for v in (_attr(e, field) for e in entries) if _is_num(v) and v >= 0) * scale))


def _first_name(*values: Any) -> Optional[str]:
    """The first usable provider, model or voice name among values."""
    return next(filter(None, map(_name, values)), None)


def _name(value: Any) -> Optional[str]:
    """A non-empty string other than "unknown", cut to 200 UTF-16 units; else None."""
    text = value.strip() if isinstance(value, str) else ""
    return _cut_utf16(text, _MAX_NAME) if text and text != "unknown" else None


def _millis(seconds: Any) -> Optional[int]:
    """Seconds as whole milliseconds; None unless a finite number >= 0."""
    return int(round(seconds * 1000)) if _is_num(seconds) and seconds >= 0 else None


def _is_num(value: Any) -> bool:
    """A finite int or float, bools excluded."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _attr(obj: Any, name: str) -> Any:
    """getattr(obj, name, None) that also swallows errors raised by properties (current_agent before start)."""
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001
        return None


def _message_text(item: Any) -> str:
    """ChatMessage.text_content (1.8 strips LiveKit's <expr/> markup from agent text), else the string parts joined."""
    text = _attr(item, "text_content")
    if not isinstance(text, str):
        content = getattr(item, "content", None)
        parts = [content] if isinstance(content, str) else [c for c in content or [] if isinstance(c, str)]
        text = "\n".join(parts)
    return text.strip()


def _as_dict(data: Any, setter: str) -> Optional[dict[str, Any]]:
    """What a custom_data setter got, as a dict (any mapping, as 0.3.x took); anything else is left out with a
    warning, never raised."""
    try:
        return dict(data)
    except Exception:  # noqa: BLE001
        logger.warning("SUPERBRYN_CALL_REPORT_ERROR: %s takes a dict, got %s; left out", setter, type(data).__name__)
        return None


def _is_text(value: Any, setter: str) -> bool:
    """Whether a setter got non-blank text; None and blanks are ignored, anything else is left out with a warning."""
    if isinstance(value, str):
        return bool(value.strip())
    if value is not None:
        logger.warning("SUPERBRYN_CALL_REPORT_ERROR: %s takes text, got %s; left out", setter, type(value).__name__)
    return False


def _cut(value: Any, limit: int) -> Any:
    """A string cut to limit UTF-16 units; anything else as it is."""
    return _cut_utf16(value, limit) if isinstance(value, str) else value


def _cut_utf16(text: str, limit: int) -> str:
    """At most limit UTF-16 code units, as the server's zod .max() counts; a split surrogate pair is dropped."""
    if len(text) * 2 <= limit:  # fits even if every character is a surrogate pair
        return text
    return text.encode("utf-16-le", errors="surrogatepass")[:limit * 2].decode("utf-16-le", errors="ignore")


def _metrics(item: Any) -> dict[str, Any]:
    metrics = getattr(item, "metrics", None)
    return metrics if isinstance(metrics, dict) else {}


def _parse_json(value: Any) -> Any:
    """Parsed JSON when the string is strict JSON (no NaN or Infinity, which JSON.parse refuses), else the raw value."""
    if isinstance(value, str):
        try:
            return json.loads(value, parse_constant=_not_json, parse_float=_finite)
        except (ValueError, RecursionError):
            pass
    return value


def _not_json(token: str) -> Any:
    raise ValueError(f"{token} is not JSON")


def _finite(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"{text} is out of range")
    return number


def _strict_json(name: str, value: Any) -> Any:
    """value when strict JSON can hold it (no NaN, Infinity or objects), else None with one warning line."""
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError, RecursionError) as e:
        logger.warning("SUPERBRYN_CALL_REPORT_ERROR: %s left out of call.ended: %s", name, e)
        return None


def _compact(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop unset fields: the contract's optional fields must be absent, not null."""
    return {k: v for k, v in fields.items() if v is not None}


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _now_iso() -> str:
    return _iso(time.time())
