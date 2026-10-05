"""
Eval mode: send the agent's own record of each SuperBryn test call (transcript, tool calls, latency,
configuration and usage; never audio) to SuperBryn. Everything runs in the background; other calls send nothing.
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
import re
import stat
import tempfile
import time
import uuid
import weakref
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional

import aiohttp
from livekit import rtc
from livekit.agents import llm

if TYPE_CHECKING:
    from livekit.agents import AgentSession, JobContext

logger = logging.getLogger("test_data")

_DEFAULT_BASE_URL = "https://api.superbryn.com"
_AGENT_DATA_PATH = "/public-api/v1/test-calls/agent-data"
_CALLER_NUMBERS_PATH = "/public-api/v1/test-calls/caller-numbers"
_TEST_ID_KEY_SUFFIXES = ("x-superbryn-call-id", "superbryn_call_id")

_SIP_WAIT_S = 15.0
_POLL_S = 0.25
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
# An LLM error this long after a request's metrics were stamped still belongs to it.
_FAILURE_SLACK_S = 1.0
_MAX_ERROR_MESSAGE = 1000
# LiveKit's error types by pipeline step; a realtime model is the LLM step
_ERROR_STEPS = {"stt_error": "stt", "llm_error": "llm", "tts_error": "tts", "realtime_model_error": "llm"}
# Where LiveKit's FallbackAdapter / StreamAdapter / RealtimeModelFallbackAdapter (1.x) and common wrappers keep
# the component they wrap.
_WRAPPED_ATTRS = {
    "stt": ("_stt_instances", "_stt", "_wrapped_stt", "_inner", "_wrapped"),
    "llm": ("_llm_instances", "_llm", "_wrapped_llm", "_inner", "_wrapped", "_models"),
    "tts": ("_tts_instances", "_wrapped_tts", "_tts", "_inner", "_wrapped"),
}

# base URL -> (monotonic fetch time, caller numbers); a temp file shares them with later jobs' processes
_caller_numbers_cache: dict[str, tuple[float, list[str]]] = {}
_attached_jobs: weakref.WeakSet = weakref.WeakSet()  # job contexts eval mode is attached to
_no_key_logged = False
_no_agent_logged = False


def attach_test_data(
    ctx: JobContext,
    session: AgentSession,
    *,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Report this call to SuperBryn if it is a SuperBryn test call; call before session.start().

    Returns at once: recognition and sending run in a background task and one shutdown callback.
    """
    global _no_key_logged, _no_agent_logged
    try:
        key = api_key or os.getenv("SUPERBRYN_TEST_API_KEY") or os.getenv("SUPERBRYN_API_KEY")
        if not key:
            if not _no_key_logged:
                _no_key_logged = True
                logger.warning("SUPERBRYN_TEST_DATA_DISABLED: no API key, set SUPERBRYN_TEST_API_KEY")
            return
        sent_agent = _resolve_agent_id(agent_id)
        agent = _agent_uuid(sent_agent)
        if not agent:
            if not _no_agent_logged:
                _no_agent_logged = True
                logger.warning("SUPERBRYN_TEST_DATA_DISABLED: %s", f"agent_id {sent_agent!r} is not a SuperBryn agent ID (a UUID)"
                               if sent_agent else "no agent ID, set SUPERBRYN_AGENT_ID")
            return
        base = (base_url or os.getenv("SUPERBRYN_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        if ctx in _attached_jobs:  # one record per job: a second one would reuse its event IDs
            logger.warning("SUPERBRYN_TEST_DATA_ERROR: attach_test_data already ran in this job, "
                           "only the first session is reported")
            return
        loop = asyncio.get_running_loop()
        call = _TestCall(ctx, session, key, agent, base)
        for event, callback in (("close", call.on_close), ("conversation_item_added", call.sample),
                                ("agent_state_changed", call.sample)):
            try:
                session.on(event, callback)
            except Exception as e:  # noqa: BLE001 - not an event emitter: those fields are left out
                logger.debug("SUPERBRYN_TEST_DATA: cannot listen to %s: %s", event, e)
        call.task = loop.create_task(call.run(), name="superbryn_test_data")
        ctx.add_shutdown_callback(call.on_shutdown)
        _attached_jobs.add(ctx)
    except Exception as e:  # noqa: BLE001
        logger.warning("SUPERBRYN_TEST_DATA_ERROR: could not attach: %s", e)


class _TestCall:
    """One job's eval-mode state: the recognition task, the start fields and the server's answer."""

    def __init__(self, ctx: Any, session: Any, api_key: str, agent_id: str, base_url: str) -> None:
        self.ctx = ctx
        self.session = session
        self.api_key = api_key
        self.agent_id = agent_id
        self.base_url = base_url
        # the room is usually not connected yet, so its name comes from the job
        job_room = getattr(getattr(ctx, "job", None), "room", None)
        self.room_name: str = getattr(job_room, "name", "") or getattr(ctx.room, "name", "") or ""
        self.task: Optional[asyncio.Task] = None
        self.test_id: Optional[str] = None
        self.is_test: Optional[bool] = None
        self.call_id = ""
        self.start_fields: dict[str, Any] = {}
        self.started_ts: Optional[float] = None
        # the server refused call.started, so it would refuse call.ended too
        self.refused = False
        self.close_reason: Optional[str] = None
        # when the session closed, i.e. the caller hung up; the job shuts down only once the room goes
        self.closed_at: Optional[float] = None
        self.tools_seen: dict[str, Optional[str]] = {}  # tool name -> first description seen
        self.turn_detection: Optional[dict[str, Any]] = None  # the running agent's, sampled during the call
        self.sample_failed = False

    def confirmed(self) -> bool:
        """A recognised call, unless the server said it isn't a test or refused call.started."""
        if self.is_test is not None:
            return self.is_test
        return bool(self.call_id and self.start_fields) and not self.refused

    async def run(self) -> bool:
        """Recognise a test call and send call.started; True when it is a SuperBryn test call."""
        try:
            await self._recognise_and_start()
        except Exception as e:  # noqa: BLE001
            logger.warning("SUPERBRYN_TEST_DATA_ERROR: %s", e)
        return self.confirmed()

    async def _recognise_and_start(self) -> None:
        sip = await _wait_for_sip_participant(self.ctx.room)
        attributes = dict(getattr(sip, "attributes", None) or {})
        self.test_id = _test_id(self.ctx, attributes)
        remote = _e164(attributes.get("sip.phoneNumber") or None)
        if not self.test_id and not await _is_caller_number(remote, self.base_url, self.api_key):
            logger.debug("not a SuperBryn test call, nothing is sent")
            return

        self.call_id = await _room_sid(self.ctx) or self.room_name
        if not self.call_id:
            logger.warning("SUPERBRYN_TEST_DATA_ERROR: no room SID or name, nothing is sent")
            return
        trunk = _e164(attributes.get("sip.trunkPhoneNumber") or None)
        inbound = bool(attributes.get("sip.ruleID"))  # a dispatch rule matched: we called their agent
        self.started_ts = time.time()
        self.start_fields = _compact({
            "superbryn_call_id": self.test_id,
            "agent_id": self.agent_id,
            "from": _cut(remote if inbound else trunk, _MAX_ID),
            "to": _cut(trunk if inbound else remote, _MAX_ID),
            "started_at": _iso(self.started_ts),
        })
        self.sample()
        status, answer = await _send("POST", self.base_url + _AGENT_DATA_PATH, self.api_key,
                                     self._event("started"), timeout=_STARTED_TIMEOUT_S)
        self.refused = status is not None and answer is None
        self._read_answer("call.started", answer)

    async def on_shutdown(self, reason: str = "") -> None:
        """Send call.ended for a confirmed test call, all within 8 s of the job's shutdown; never raises."""
        try:
            await asyncio.wait_for(self._end(reason), _SHUTDOWN_S)
        except asyncio.TimeoutError:
            logger.error("SUPERBRYN_TEST_DATA_FAILED: call.ended for call %s not sent within %g s of shutdown",
                         self.call_id, _SHUTDOWN_S)
        except Exception as e:  # noqa: BLE001
            logger.warning("SUPERBRYN_TEST_DATA_ERROR: %s", e)

    async def _end(self, reason: str) -> None:
        """Stop an unfinished recognition, then send call.ended for a confirmed test call."""
        if self.task is not None and not self.task.done():
            # still waiting for a SIP participant, the caller numbers or call.started's answer: stop it;
            # call.ended repeats the start fields, so once call.started went out it can link alone
            self.task.cancel()
            await asyncio.wait({self.task})
        if not (self.confirmed() and self.call_id):
            return
        self.sample()
        items = list(getattr(getattr(self.session, "history", None), "items", None) or [])
        # every event the session emitted, as its own session report reads them
        events = list(_attr(self.session, "_recorded_events") or ())
        requests = _section("llm_requests", _request_log, events) or []
        base = _time_base(items, [start for start, _ in requests])
        transcript, tool_calls = _history_payload(items, base)
        llm_requests = _section("llm_requests", _llm_requests, requests, events, items, base)
        errors = _section("errors", _errors, events, base)
        ended_reason = self.close_reason or (reason.strip() if isinstance(reason, str) else "")
        body = self._event("ended", **_compact({
            "ended_at": self._ended_at(),
            "ended_reason": _cut_utf16(ended_reason, _MAX_NAME) or None,
            "transcript": _strict_json("transcript", transcript),
            "tool_calls": _strict_json("tool_calls", tool_calls),
            "diagnostics": _section("diagnostics", _diagnostics, self.session, items, transcript, self.tools_seen,
                                    self.turn_detection, llm_requests, errors),
        }))
        _, answer = await _send("POST", self.base_url + _AGENT_DATA_PATH, self.api_key, body,
                                timeout=_ENDED_TIMEOUT_S)
        self._read_answer("call.ended", answer)

    def on_close(self, event: Any) -> None:
        """Keep the session's close reason (a CloseReason value) for ended_reason, and its time; never raises."""
        try:
            reason = getattr(event, "reason", None)
            if reason is not None:
                self.close_reason = str(getattr(reason, "value", reason))
            created = getattr(event, "created_at", None)
            if self.closed_at is None and isinstance(created, (int, float)) and math.isfinite(created):
                self.closed_at = float(created)
        except Exception as e:  # noqa: BLE001
            logger.debug("SUPERBRYN_TEST_DATA: unreadable close event: %s", e)

    def _ended_at(self) -> str:
        """When the session closed, else now; never before the start."""
        if self.closed_at is None:
            return _now_iso()
        return _iso(max(self.closed_at, self.started_ts or self.closed_at))

    def sample(self, _event: Any = None) -> None:
        """Remember the tools the running agent offers (MCP ones only exist while it runs) and its turn detection;
        runs on every conversation item and agent state change, so it stays a cheap walk; never raises."""
        try:
            activity, agent = _attr(self.session, "_activity"), _attr(self.session, "current_agent")
            for name, description in _tool_infos([*(_attr(activity, "tools") or ()),
                                                  *(_attr(self.session, "tools") or ()),
                                                  *(_attr(agent, "tools") or ())]):
                self.tools_seen.setdefault(name, description)
            self.turn_detection = _live_turn_detection(activity) or self.turn_detection
        except Exception as e:  # noqa: BLE001
            if not self.sample_failed:
                self.sample_failed = True
                logger.warning("SUPERBRYN_TEST_DATA_ERROR: could not read tools or turn detection: %s", e)

    def _event(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Contract body for call.<kind>; every event repeats the start fields."""
        return {"event_id": f"lk:{self.call_id}:{kind}", "event": f"call.{kind}", "call_id": self.call_id,
                "source": "livekit", "schema_version": _SCHEMA_VERSION, **self.start_fields, **fields}

    def _read_answer(self, event: str, answer: Optional[dict[str, Any]]) -> None:
        if answer is None:
            return
        if isinstance(answer.get("is_test"), bool):
            self.is_test = answer["is_test"]
        logger.info("SUPERBRYN_TEST_DATA_SENT: %s for call %s: status=%s is_test=%s",
                    event, self.call_id, answer.get("status"), answer.get("is_test"))


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
                timeout: float) -> tuple[Optional[int], Optional[dict[str, Any]]]:
    """Up to 3 attempts with the same body (so the same event_id): (status, the JSON of a 2xx);
    the status is None when no answer came, and there is no JSON when it was refused."""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    for attempt in range(_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_BACKOFF_S[attempt - 1])
        try:
            status, data = await asyncio.wait_for(_http_request(method, url, headers, body), timeout)
        except Exception as e:  # noqa: BLE001 - timeouts and network errors are retried
            logger.warning("SUPERBRYN_TEST_DATA_RETRY: %s %s attempt %d: %r", method, url, attempt + 1, e)
            continue
        if 200 <= status < 300:
            return status, data if isinstance(data, dict) else {}
        if status not in (408, 429) and status < 500:
            hint = " (needs an org API key with test_data:write)" if status in (401, 403) else ""
            logger.error("SUPERBRYN_TEST_DATA_REJECTED: %s %s -> %s %s%s", method, url, status, data, hint)
            return status, None
        logger.warning("SUPERBRYN_TEST_DATA_RETRY: %s %s attempt %d -> %s", method, url, attempt + 1, status)
    logger.error("SUPERBRYN_TEST_DATA_FAILED: %s %s gave up", method, url)
    return None, None


async def _wait_for_sip_participant(room: Any) -> Any:
    """The room's SIP participant, polling up to 15 s for one to join; None if none does."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SIP_WAIT_S
    while True:
        for participant in list(getattr(room, "remote_participants", {}).values()):
            if getattr(participant, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                return participant
        if loop.time() >= deadline:
            return None
        await asyncio.sleep(_POLL_S)


def _test_id(ctx: Any, attributes: dict[str, str]) -> Optional[str]:
    """The first test ID of at most 200 UTF-16 units (the server refuses longer ones); a longer one is ignored."""
    too_long = False
    for value in (*_metadata_test_ids(ctx), *_attribute_test_ids(attributes)):
        if _cut_utf16(value, _MAX_ID) == value:
            return value
        too_long = True
    if too_long:
        logger.warning("SUPERBRYN_TEST_DATA_ERROR: superbryn_call_id over %d characters ignored", _MAX_ID)
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
    """SuperBryn's test numbers, cached for an hour in memory and in a temp file that every job on this host
    reads (LiveKit runs each job in a new process)."""
    hit = _caller_numbers_cache.get(base_url)
    if hit and time.monotonic() - hit[0] < _CALLER_NUMBERS_TTL_S:
        return hit[1]
    shared = _read_numbers_file(base_url)
    if shared is not None:
        age, numbers = shared
        _caller_numbers_cache[base_url] = (time.monotonic() - age, numbers)
        return numbers
    _, answer = await _send("GET", base_url + _CALLER_NUMBERS_PATH, api_key, None, timeout=_STARTED_TIMEOUT_S)
    numbers = answer.get("numbers") if answer else None
    if not isinstance(numbers, list):
        # remember the miss for a minute so a bad key or an outage isn't retried on every call
        _remember_numbers(base_url, [], age=_CALLER_NUMBERS_TTL_S - _CALLER_NUMBERS_RETRY_S)
        return None
    numbers = [n for n in numbers if isinstance(n, str)]
    _remember_numbers(base_url, numbers)
    return numbers


def _numbers_file(base_url: str) -> str:
    """This host's caller-number file for base_url, in the temp dir."""
    digest = hashlib.sha256(base_url.encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"superbryn_caller_numbers_{digest}.json")


def _read_numbers_file(base_url: str) -> Optional[tuple[float, list[str]]]:
    """(age in seconds, numbers) from the file while under an hour old; None when missing, stale or unreadable."""
    try:
        path = _numbers_file(base_url)
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


def _remember_numbers(base_url: str, numbers: list[str], age: float = 0.0) -> None:
    """Cache the numbers (a miss is [], aged to expire in a minute) in memory and in the file; never raises."""
    _caller_numbers_cache[base_url] = (time.monotonic() - age, numbers)
    tmp = None
    try:
        path = _numbers_file(base_url)
        fd, tmp = tempfile.mkstemp(prefix=".superbryn_caller_numbers_", dir=os.path.dirname(path))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time() - age, "numbers": numbers}, f)
        os.replace(tmp, path)  # atomic: a reader sees the old file or the new one, never half of one
    except Exception as e:  # noqa: BLE001 - the in-memory cache still has them
        logger.debug("SUPERBRYN_TEST_DATA: caller numbers not cached on disk: %s", e)
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
    """agent_id= or SUPERBRYN_AGENT_ID; never AGENT_ID, which Observability uses as a free-form name."""
    return explicit or os.getenv("SUPERBRYN_AGENT_ID")


def _agent_uuid(value: Optional[str]) -> Optional[str]:
    """The agent ID when the server's z.uuid() accepts it (RFC 4122 variant, version 1-8), else None."""
    if not value:
        return None
    try:
        parsed = uuid.UUID(value)
        if parsed.variant == uuid.RFC_4122 and 1 <= parsed.version <= 8:
            return str(parsed)
    except ValueError:
        pass
    return None


def _time_base(items: list[Any], starts: Iterable[float] = ()) -> float:
    """The call's time zero: the earliest history moment or LLM request start; 0 when there is none."""
    stamps = [t for item in items
              for t in (getattr(item, "created_at", None), _metrics(item).get("started_speaking_at"))
              if _is_num(t) and t > 0]
    stamps += [t for t in starts if _is_num(t) and t > 0]
    return min(stamps) if stamps else 0.0


def _history_payload(items: list[Any], base: Optional[float] = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Contract turns and tool calls from the history items; ms are from base (by default the history's first moment)."""
    if base is None:
        base = _time_base(items)

    def ms(t: Any) -> Optional[int]:
        return max(0, int(round((t - base) * 1000))) if _is_num(t) and t > 0 else None

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
                transcript.append(_compact({
                    "id": turn_id,
                    "role": "user" if user else "agent",
                    "text": _cut_utf16(text, _MAX_TEXT),
                    "start_ms": ms(metrics.get("started_speaking_at") or getattr(item, "created_at", None)),
                    "end_ms": ms(metrics.get("stopped_speaking_at")),
                    "confidence": confidence if user and _is_num(confidence) and 0 <= confidence <= 1 else None,
                    "interrupted": None if user else bool(getattr(item, "interrupted", False)),
                }))
                if not user:
                    for call in waiting:
                        call.update(_compact({"turn_id": turn_id}))
                    waiting.clear()
                last_agent = None if user else turn_id
        elif kind == "function_call":
            output = outputs.get(item.call_id)
            tool_calls.append(_compact({
                "id": _cut(item.call_id, _MAX_ID),
                "turn_id": last_agent,
                "name": _cut(item.name, _MAX_NAME),
                "arguments": _parse_json(item.arguments),
                "result": getattr(output, "output", None),
                "is_error": bool(output.is_error) if output is not None else None,
                "start_ms": ms(getattr(item, "created_at", None)),
                "end_ms": ms(getattr(output, "created_at", None)),
            }))
            if last_agent is None:
                waiting.append(tool_calls[-1])
    return transcript[:_MAX_TURNS], tool_calls[:_MAX_TOOL_CALLS]


def _turn_id(item: Any) -> Optional[str]:
    """The history item's id as a contract turn id (1-200 UTF-16 units), or None."""
    value = getattr(item, "id", None)
    return _cut_utf16(value, _MAX_ID) if isinstance(value, str) and value else None


def _diagnostics(session: Any, items: list[Any], transcript: list[dict[str, Any]], tools: dict[str, Optional[str]],
                 turn_detection: Optional[dict[str, Any]] = None,
                 llm_requests: Optional[list[dict[str, Any]]] = None,
                 errors: Optional[list[dict[str, Any]]] = None) -> Optional[dict[str, Any]]:
    """call.ended diagnostics, section by section: a section that fails is left out alone. turn_detection is what
    the running agent used, when it was seen during the call; else it comes from the session's options."""
    usage = _section("usage", list, _attr(_attr(session, "usage"), "model_usage") or ()) or []
    options = _attr(session, "options")
    configuration = {kind: _section(kind, _component, session, kind, usage) for kind in ("stt", "llm", "tts")}
    configuration["turn_detection"] = turn_detection or _section("turn_detection", _turn_detection, options)
    configuration["interruptions"] = _section("interruptions", _interruptions, options)
    configuration["tools"] = [_compact({"name": name, "description": description})
                              for name, description in list(tools.items())[:_MAX_TOOLS]]
    turn_ids = {turn["id"] for turn in transcript if "id" in turn}
    diagnostics = {
        "pipeline": _section("pipeline", _pipeline, session),
        "configuration": {k: v for k, v in configuration.items() if v},
        "turn_latencies": _section("turn_latencies", _turn_latencies, items, turn_ids),
        "usage": _section("usage", _usage, usage),
        "llm_requests": llm_requests,
        "errors": errors,
    }
    diagnostics = {k: _strict_json(k, v) for k, v in diagnostics.items() if v}
    return {k: v for k, v in diagnostics.items() if v} or None


def _section(name: str, build: Callable[..., Any], *args: Any) -> Any:
    """build(*args), or None with one warning line when it raises."""
    try:
        return build(*args)
    except Exception as e:  # noqa: BLE001
        logger.warning("SUPERBRYN_TEST_DATA_ERROR: %s left out of call.ended: %s", name, e)
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
    top = max(entries, key=_tokens, default=None) if kind == "llm" else None
    if top is not None and _tokens(top) > 0 and _name(_attr(top, "model")) not in (None, fields["model"]):
        fields = {key: _name(_attr(top, key)) for key in ("provider", "model")}
    if kind == "tts":
        opts = _attr(base, "_opts")
        fields["voice"] = _first_name(_attr(base, "voice_id"), _attr(base, "voice"),
                                      _attr(opts, "voice_id"), _attr(opts, "voice"))
    return _compact(fields)


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
    else:
        detector = None
        min_s, max_s = _attr(options, "min_endpointing_delay"), _attr(options, "max_endpointing_delay")
    return _compact({"type": _detector_name(detector), "min_end_of_turn_ms": _millis(min_s),
                     "max_end_of_turn_ms": _millis(max_s)})


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
    return _compact({"type": _detector_name(_attr(activity, "_turn_detection")),
                     "min_end_of_turn_ms": _millis(_attr(activity, "min_endpointing_delay")),
                     "max_end_of_turn_ms": _millis(_attr(activity, "max_endpointing_delay"))})


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
                "transcriberLatency": _millis(transcribed),
                "endpointingLatency": _millis(endpointing),
                "modelLatency": _millis(agent.get("llm_node_ttft")),
                "voiceLatency": _millis(agent.get("tts_node_ttfb")),
                "turnLatency": _millis(agent.get("e2e_latency")),
            })
            user = {}
            if entry:
                turn_id = _turn_id(item)
                latencies.append({"turn_id": turn_id, **entry} if turn_id in turn_ids else entry)
    return latencies[:_MAX_TURN_LATENCIES]


def _request_log(events: list[Any]) -> list[tuple[float, Any]]:
    """(start time, metrics) of each LLM request and realtime response the session recorded, earliest first."""
    log = []
    for event in events:
        metrics = _attr(event, "metrics") if _attr(event, "type") == "metrics_collected" else None
        kind, stamp, duration = _attr(metrics, "type"), _attr(metrics, "timestamp"), _attr(metrics, "duration")
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

    llm_failures = [at for at, step, _ in _reported_errors(events) if step == "llm"]
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
            failed = not usage and any(start <= at <= _attr(metrics, "timestamp") + _FAILURE_SLACK_S
                                       for at in llm_failures)
            outcome = "not_spoken" if _is_num(ttft) and ttft >= 0 and not failed else "stopped"
        requests.append({"start_ms": max(0, int(round((start - base) * 1000))), "outcome": outcome, **usage})
    return requests


def _reported_errors(events: list[Any]) -> list[tuple[float, str, Any]]:
    """The STT, LLM and TTS failures the session reported, oldest first, each with when it happened."""
    found = []
    for event in events:
        error = _attr(event, "error") if _attr(event, "type") == "error" else None
        step = _ERROR_STEPS.get(_attr(error, "type"))
        at = next((t for t in (_attr(error, "timestamp"), _attr(event, "created_at")) if _is_num(t) and t > 0), None)
        if step is not None and at is not None:
            found.append((at, step, error))
    return sorted(found, key=lambda entry: entry[0])


def _errors(events: list[Any], base: float) -> list[dict[str, Any]]:
    """Each STT, LLM or TTS failure the session reported. LiveKit retries a recoverable one, so it recovered unless
    that step later failed for good; LiveKit's STT hears the caller."""
    found = _reported_errors(events)
    gave_up: dict[str, float] = {}  # step -> its last unrecoverable failure
    for at, step, error in found:
        if _attr(error, "recoverable") is False:
            gave_up[step] = at
    errors = []
    for at, step, error in found[:_MAX_ERRORS]:
        recoverable = _attr(error, "recoverable")
        errors.append(_compact({
            "step": step,
            "side": "caller" if step == "stt" else None,
            "start_ms": max(0, int(round((at - base) * 1000))),
            "message": _error_message(error),
            "recovered": recoverable and gave_up.get(step, -math.inf) <= at if isinstance(recoverable, bool) else None,
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
    """LLM token, STT audio and TTS character totals; a field only when entries of its type exist."""
    llms, stts, ttss = _of_type(usage, "llm_usage"), _of_type(usage, "stt_usage"), _of_type(usage, "tts_usage")
    return _compact({
        "llm_input_tokens": _total(llms, "input_tokens") if llms else None,
        "llm_cached_input_tokens": _total(llms, "input_cached_tokens") if llms else None,
        "llm_output_tokens": _total(llms, "output_tokens") if llms else None,
        "stt_audio_ms": _total(stts, "audio_duration", 1000) if stts else None,
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
        logger.warning("SUPERBRYN_TEST_DATA_ERROR: %s left out of call.ended: %s", name, e)
        return None


def _compact(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop unset fields: the contract's optional fields must be absent, not null."""
    return {k: v for k, v in fields.items() if v is not None}


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _now_iso() -> str:
    return _iso(time.time())
