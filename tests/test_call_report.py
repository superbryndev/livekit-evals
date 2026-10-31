"""Tests for attach() (livekit_evals.call_report) with fake job, room, session, recorder and HTTP; no network."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from aiohttp import web
from livekit import rtc
from livekit.agents import (Agent, AgentSession, AgentStateChangedEvent, CloseEvent, CloseReason,
                            ConversationItemAddedEvent, llm)
from livekit.agents.llm import AgentHandoff, ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.metrics import AgentSessionUsage, LLMModelUsage, STTModelUsage, TTSModelUsage
from livekit.agents.version import __version__ as LIVEKIT_AGENTS_VERSION
from livekit.agents.voice.agent_activity import AgentActivity

from livekit_evals import call_report
from livekit_evals.call_report import attach

# Features only livekit-agents 1.8 has (expressive markup, the realtime fallback adapter).
needs_livekit_1_8 = pytest.mark.skipif(
    tuple(int(part) for part in LIVEKIT_AGENTS_VERSION.split(".")[:2]) < (1, 8), reason="a livekit-agents 1.8 feature")

OUR_NUMBER = "+14155550100"
THEIR_NUMBER = "+16505550199"
AGENT_UUID = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"
BASE = "https://sb.test"
REPORT_URL = BASE + "/public-api/v1/calls/agent-data"
LINKED = (200, {"surface": "testing", "status": "linked", "is_test": True,
                 "link": {"status": "verified", "method": "test_id"}})
STORED = (200, {"surface": "observability", "ok": True, "call_id": "obs_1"})
T0 = 1_760_000_000.0


class FakeHttp:
    """Replaces call_report._http_request: records requests, plays scripted answers per endpoint."""

    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []
        self.script = {
            "caller-numbers": [(200, {"numbers": [OUR_NUMBER, "+919876543210"]})],
            "started": [LINKED],
            "ended": [LINKED],
            "obs_started": [(200, {"surface": "observability", "ok": True, "reason": "call_started_not_stored"})],
            "obs_ended": [STORED],
        }

    async def __call__(self, method, url, headers, body):
        # as strict as the real transport: NaN or Infinity in a body fails the test
        self.calls.append(SimpleNamespace(method=method, url=url, headers=dict(headers),
                                          body=json.loads(json.dumps(body, allow_nan=False)) if body else None))
        prefix = "obs_" if body and body.get("surface") == "observability" else ""
        queue = self.script["caller-numbers" if url.endswith("/caller-numbers") else prefix + body["event"][5:]]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if answer == "hang":
            await asyncio.sleep(3600)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def posts(self, event):
        return [c for c in self.calls if c.method == "POST" and c.body["event"] == event]

    def gets(self):
        return [c for c in self.calls if c.method == "GET"]


class FakeRoom:
    """rtc.Room stand-in; sid is awaitable like livekit-rtc 1.x unless plain_sid is set."""

    def __init__(self, participants=(), name="call-room-1", metadata="", sid="RM_room1", plain_sid=False):
        self.name = name
        self.metadata = metadata
        self.remote_participants = {p.identity: p for p in participants}
        self._sid = sid
        self._plain_sid = plain_sid
        self.handlers = {}

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def off(self, event, callback):
        self.handlers.get(event, []).remove(callback)

    def emit(self, event, *args):
        for callback in list(self.handlers.get(event, [])):
            callback(*args)

    @property
    def sid(self):
        if self._plain_sid:
            return self._sid

        async def _sid():
            return self._sid

        return _sid()


class FakeCtx:
    """JobContext stand-in; shutdown() runs the callbacks concurrently, as the worker does."""

    def __init__(self, room, job_metadata="", job_room_name="call-room-1", job_room_sid="RM_job1"):
        self.room = room
        self.job = SimpleNamespace(metadata=job_metadata,
                                   room=SimpleNamespace(name=job_room_name, sid=job_room_sid, metadata=""))
        self.shutdown_callbacks = []

    def add_shutdown_callback(self, callback):
        self.shutdown_callbacks.append(callback)

    async def shutdown(self, reason="room disconnected"):
        await asyncio.gather(*(cb(reason) if inspect.signature(cb).parameters else cb()
                               for cb in self.shutdown_callbacks))


def sip_participant(phone=OUR_NUMBER, trunk=THEIR_NUMBER, rule="SDR_inbound", joined=T0, **attributes):
    """A SIP participant that joined at `joined`; a non-empty sip.ruleID means the call came in to their agent."""
    attrs = {"sip.phoneNumber": phone, "sip.trunkPhoneNumber": trunk, "sip.ruleID": rule, **attributes}
    return SimpleNamespace(identity=f"sip_{phone}", kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
                           attributes=attrs, joined_at=datetime.fromtimestamp(joined, timezone.utc),
                           disconnect_reason=None)


def web_participant(identity="web-user", joined=T0):
    """A caller in a browser (a standard participant)."""
    return SimpleNamespace(identity=identity, kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, attributes={},
                           joined_at=datetime.fromtimestamp(joined, timezone.utc), disconnect_reason=None)


def session_with(*items):
    return SimpleNamespace(history=ChatContext(list(items)))


class FakeSession:
    """AgentSession stand-in: history, event handlers and whatever a test sets (llm, options, usage, tools)."""

    def __init__(self, *items, agent=None, **attributes):
        self.history = ChatContext(list(items))
        self.agent = agent
        self.handlers = {}
        self.__dict__.update(attributes)

    @property
    def current_agent(self):
        if self.agent is None:
            raise RuntimeError("VoiceAgent isn't running")  # as AgentSession does before start()
        return self.agent

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event, arg):
        for callback in self.handlers.get(event, []):
            callback(arg)


class FakeRealtime(llm.RealtimeModel):
    provider = "openai"
    model = "gpt-realtime"

    def __init__(self):  # no capabilities needed
        pass


class Realtime(llm.RealtimeModel):
    """A realtime model that detects turns itself, as OpenAI's does by default."""

    def __init__(self, model="gpt-realtime", provider="openai"):
        super().__init__(capabilities=llm.RealtimeCapabilities(
            message_truncation=True, turn_detection=True, user_transcription=True,
            auto_tool_reply_generation=False, audio_output=True, manual_function_calls=True))
        self._model, self._provider = model, provider

    model = property(lambda self: self._model)
    provider = property(lambda self: self._provider)

    def session(self, *, turn_detection_disabled=False):
        raise NotImplementedError


class NamedDetector:
    model = "turn-detector-v1-mini"  # as livekit.agents.inference.TurnDetector reports


class EnglishModel:
    """A turn detector without a model name."""


class BrokenAdapter:
    @property
    def _inner(self):
        raise RuntimeError("adapter closed")


def chat(role, text, at, **fields):
    return ChatMessage(role=role, content=[text], created_at=T0 + at, **fields)


def tool(name, description):
    async def run() -> str:
        return "ok"

    return llm.function_tool(run, name=name, description=description)


def raw_tool(name, description=None):
    async def run(raw_arguments: dict) -> str:
        return "ok"

    schema = {"name": name, "parameters": {"type": "object", "properties": {}}}
    return llm.function_tool(run, raw_schema={**schema, "description": description} if description else schema)


def ctx_with_test_id():
    return FakeCtx(FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"})))


async def ended_body(http, session, reason="room disconnected"):
    """Run one test call through shutdown; the last call.ended body."""
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown(reason)
    return http.posts("call.ended")[-1].body


async def recognised(ctx) -> bool:
    """Wait for the background recognition (and call.started) to settle; True for a reported call."""
    report = ctx.shutdown_callbacks[0].__self__
    return await asyncio.wait_for(asyncio.shield(report._task), 2)


async def until(condition, timeout=2.0):
    """Wait until condition() holds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "timed out"
        await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def fast_and_clean(monkeypatch, tmp_path):
    for name in ("SUPERBRYN_TEST_API_KEY", "SUPERBRYN_API_KEY", "SUPERBRYN_AGENT_ID", "AGENT_ID", "SUPERBRYN_BASE_URL",
                 "WEBHOOK_URL"):
        monkeypatch.delenv(name, raising=False)
    # Test calls need the agent; tests about something else get it from the environment.
    monkeypatch.setenv("SUPERBRYN_AGENT_ID", AGENT_UUID)
    monkeypatch.setattr(call_report, "_bad_agent_logged", False)
    monkeypatch.setattr(call_report, "_numbers_refused_logged", False)
    monkeypatch.setattr(call_report, "_SIP_WAIT_S", 0.2)
    monkeypatch.setattr(call_report, "_POLL_S", 0.01)
    monkeypatch.setattr(call_report, "_BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(call_report, "_no_key_logged", False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the caller-number file of this test only
    call_report._caller_numbers_cache.clear()


@pytest.fixture
def http(monkeypatch):
    fake = FakeHttp()
    monkeypatch.setattr(call_report, "_http_request", fake)
    return fake


class FakeRecorder:
    """RecordingManager stand-in: what it was asked, and a URL shaped like the real one."""

    def __init__(self, credentials_url, api_key):
        self.credentials_url, self.api_key = credentials_url, api_key
        self.rooms, self.stopped = [], False

    async def start_recording(self, room_name):
        self.rooms.append(room_name)
        return f"https://recordings.test/call_recordings/{room_name}/20251009-085320/call.mp3"

    async def stop_recording(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def recorders(monkeypatch):
    """Every recording a test started; none touches LiveKit or S3."""
    made = []

    def make(*args, **kwargs):
        made.append(FakeRecorder(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(call_report, "RecordingManager", make)
    return made


# --- recognising a test call -------------------------------------------------


@pytest.mark.parametrize("where", ["job", "room"])
async def test_metadata_test_id_sends_call_started(http, where):
    metadata = json.dumps({"superbryn_call_id": "sbc_meta"})
    # their agent dialled our number: no dispatch rule, so from = their trunk, to = us
    room = FakeRoom([sip_participant(rule="")], metadata=metadata if where == "room" else "")
    ctx = FakeCtx(room, job_metadata=metadata if where == "job" else "")
    attach(ctx, session_with(), api_key="key_1", agent_id=AGENT_UUID, base_url=BASE + "/")

    assert await recognised(ctx) is True
    assert http.gets() == []  # a test ID needs no caller-number prefilter
    [started] = http.posts("call.started")
    assert started.url == REPORT_URL
    assert started.headers == {"X-API-Key": "key_1", "Content-Type": "application/json"}
    # identity only: the start fields and the testing block
    assert started.body == {
        "schema_version": 1, "event_id": "lk:RM_job1:started", "event": "call.started", "surface": "testing",
        "source": "livekit", "call_id": "RM_job1", "agent_id": AGENT_UUID, "from": THEIR_NUMBER, "to": OUR_NUMBER,
        "started_at": "2025-10-09T08:53:20.000Z",  # when the caller joined the room
        "testing": {"superbryn_call_id": "sbc_meta"},
    }


@pytest.mark.parametrize("key", ["sip.h.x-superbryn-call-id", "sip.h.X-SuperBryn-Call-Id",
                                 "superbryn_call_id"])
async def test_sip_header_attribute_test_id(http, key):
    room = FakeRoom([sip_participant(**{key: " sbc_hdr "})])
    ctx = FakeCtx(room)
    attach(ctx, session_with(), api_key="key_1", base_url=BASE)

    assert await recognised(ctx) is True
    assert http.gets() == []
    [started] = http.posts("call.started")
    assert started.body["testing"] == {"superbryn_call_id": "sbc_hdr"}
    # we called their agent: from = us, to = their trunk number
    assert (started.body["from"], started.body["to"]) == (OUR_NUMBER, THEIR_NUMBER)
    assert started.body["agent_id"] == AGENT_UUID


async def test_a_trunk_number_given_as_digits_only_is_sent_with_its_plus(http):
    room = FakeRoom([sip_participant(trunk=THEIR_NUMBER.lstrip("+"), superbryn_call_id="sbc_hdr")])
    ctx = FakeCtx(room)
    attach(ctx, session_with(), api_key="key_1", base_url=BASE)

    assert await recognised(ctx) is True
    [started] = http.posts("call.started")
    assert (started.body["from"], started.body["to"]) == (OUR_NUMBER, THEIR_NUMBER)


async def test_candidate_confirmed_by_server_sends_both_events(http):
    ctx = FakeCtx(FakeRoom([sip_participant(phone="+1 (415) 555-0100")]))  # formatting differs from ours
    attach(ctx, session_with(ChatMessage(role="user", content=["hi"])), api_key="k", base_url=BASE)

    assert await recognised(ctx) is True
    [get] = http.gets()
    assert get.url == BASE + "/public-api/v1/test-calls/caller-numbers"
    assert get.headers["X-API-Key"] == "k"
    [started] = http.posts("call.started")
    assert started.body["surface"] == "testing" and "testing" not in started.body  # no test ID to send

    await ctx.shutdown()
    [ended] = http.posts("call.ended")
    assert ended.body["event_id"] == "lk:RM_job1:ended"
    assert ended.body["transcript"][0]["text"] == "hi"

    # the caller numbers are cached for the process
    ctx2 = FakeCtx(FakeRoom([sip_participant()], name="call-room-2"), job_room_name="call-room-2")
    attach(ctx2, session_with(), api_key="k", base_url=BASE)
    assert await recognised(ctx2) is True
    assert len(http.gets()) == 1


async def test_candidate_denied_by_server_stops(http):
    http.script["started"] = [(200, {"status": "ignored", "is_test": False, "reason": "not_a_test"})]
    ctx = FakeCtx(FakeRoom([sip_participant(phone="4155550100")]))  # last-10-digit match
    attach(ctx, session_with(), api_key="k", base_url=BASE)

    assert await recognised(ctx) is False
    await ctx.shutdown()
    assert len(http.posts("call.started")) == 1
    assert http.posts("call.ended") == []


@pytest.mark.parametrize("participants", [[sip_participant(phone="+12125550123")], [web_participant()], []],
                         ids=["other-caller", "web-caller", "no-caller"])
async def test_production_false_sends_nothing_for_a_call_that_is_not_a_test(http, recorders, participants):
    ctx = FakeCtx(FakeRoom(participants))
    attach(ctx, session_with(ChatMessage(role="user", content=["hi"])), api_key="k", base_url=BASE, production=False)

    assert await recognised(ctx) is False
    await ctx.shutdown()
    assert [c for c in http.calls if c.method == "POST"] == []
    assert len(http.gets()) == (1 if participants and participants[0].kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                                else 0)
    assert recorders == []


async def test_production_false_still_reports_a_test_call(http, recorders):
    ctx = ctx_with_test_id()
    attach(ctx, session_with(), api_key="k", base_url=BASE, production=False)

    assert await recognised(ctx) is True
    await ctx.shutdown()
    assert [c.body["surface"] for c in http.calls] == ["testing", "testing"]
    assert recorders == []  # test calls are never recorded by the package


@pytest.mark.parametrize("plain_sid", [False, True], ids=["awaitable-sid", "plain-sid"])
async def test_room_sid_when_job_has_none(http, plain_sid):
    room = FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"}), plain_sid=plain_sid)
    ctx = FakeCtx(room, job_room_sid="")
    attach(ctx, session_with(), api_key="k", base_url=BASE)

    await recognised(ctx)
    [started] = http.posts("call.started")
    assert (started.body["call_id"], started.body["event_id"]) == ("RM_room1", "lk:RM_room1:started")


def numbers_file(tmp_path):
    return tmp_path / os.path.basename(call_report._numbers_file(BASE, "k"))


def age_numbers_file(tmp_path, seconds):
    data = json.loads(numbers_file(tmp_path).read_text())
    numbers_file(tmp_path).write_text(json.dumps({**data, "fetched_at": data["fetched_at"] - seconds}))


async def test_caller_numbers_are_shared_with_later_jobs_on_the_host(http, tmp_path):
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert json.loads(numbers_file(tmp_path).read_text())["numbers"] == [OUR_NUMBER, "+919876543210"]
    call_report._caller_numbers_cache.clear()  # LiveKit runs the next job in a new process
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert len(http.gets()) == 1

    age_numbers_file(tmp_path, 3600)  # an hour old: fetched again
    call_report._caller_numbers_cache.clear()
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert len(http.gets()) == 2
    assert call_report._numbers_file("https://other.test", "k") != call_report._numbers_file(BASE, "k")
    assert call_report._numbers_file(BASE, "another agent's key") != call_report._numbers_file(BASE, "k")


async def test_a_refused_key_is_asked_again_by_any_job_only_after_an_hour(http, tmp_path):
    http.script["caller-numbers"] = [(403, {"error": "insufficient_scope"}), (200, {"numbers": [OUR_NUMBER]})]
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is False
    age_numbers_file(tmp_path, 60)
    call_report._caller_numbers_cache.clear()
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is False  # a minute on: still refused
    assert len(http.gets()) == 1

    age_numbers_file(tmp_path, 3540)
    call_report._caller_numbers_cache.clear()
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert len(http.gets()) == 2


async def test_an_unanswered_caller_number_lookup_is_retried_by_any_job_after_a_minute(http, tmp_path):
    http.script["caller-numbers"] = [ConnectionError("down")] * 3 + [(200, {"numbers": [OUR_NUMBER]})]
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is False
    asked = len(http.gets())
    call_report._caller_numbers_cache.clear()
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is False
    assert len(http.gets()) == asked

    age_numbers_file(tmp_path, 60)
    call_report._caller_numbers_cache.clear()
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is True


async def test_the_caller_number_file_never_breaks_the_lookup(http, monkeypatch, tmp_path):
    numbers_file(tmp_path).write_text("{not json")  # unreadable: fetched again and replaced
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert json.loads(numbers_file(tmp_path).read_text())["numbers"] == [OUR_NUMBER, "+919876543210"]
    assert [p.name for p in tmp_path.iterdir()] == [numbers_file(tmp_path).name]  # no temp file left behind

    elsewhere = tmp_path / "elsewhere.json"  # a link (or a pipe or device, which could block) is never opened
    numbers_file(tmp_path).rename(elsewhere)
    numbers_file(tmp_path).symlink_to(elsewhere)
    assert call_report._read_numbers_file(BASE, "k") is None

    def no_temp_dir():
        raise FileNotFoundError("No usable temporary directory")

    monkeypatch.setattr(tempfile, "gettempdir", no_temp_dir)  # nowhere to keep it: memory only
    call_report._caller_numbers_cache.clear()
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert await call_report._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert len(http.gets()) == 2


async def test_a_test_id_over_200_characters_is_ignored(http, caplog):
    metadata = json.dumps({"superbryn_call_id": "sbc_" + "y" * 250})
    ctx = FakeCtx(FakeRoom([sip_participant()]), job_metadata=metadata)
    attach(ctx, session_with(), api_key="k", base_url=BASE)

    assert await recognised(ctx) is True  # the caller number still matches
    [started] = http.posts("call.started")
    assert "testing" not in started.body
    assert len(http.gets()) == 1
    assert caplog.text.count("superbryn_call_id over 200 characters ignored") == 1

    header = {"sip.h.x-superbryn-call-id": "sbc_hdr"}  # a valid test ID elsewhere is used instead
    ctx2 = FakeCtx(FakeRoom([sip_participant(**header)], name="call-room-2"), job_metadata=metadata,
                   job_room_name="call-room-2", job_room_sid="RM_job2")
    attach(ctx2, session_with(), api_key="k", base_url=BASE)
    assert await recognised(ctx2) is True
    assert http.posts("call.started")[1].body["testing"] == {"superbryn_call_id": "sbc_hdr"}


async def test_from_and_to_are_cut_to_200_characters(http):
    room = FakeRoom([sip_participant(phone="sip:" + "1" * 300, trunk="2" * 250)],
                    metadata=json.dumps({"superbryn_call_id": "sbc_1"}))
    ctx = FakeCtx(room)
    attach(ctx, session_with(), api_key="k", base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown()

    for event in ("call.started", "call.ended"):
        [post] = http.posts(event)
        assert (post.body["from"], post.body["to"]) == (("sip:" + "1" * 300)[:200], "2" * 200)


# --- call.ended payload -------------------------------------------------------


async def test_transcript_and_tool_call_mapping(http):
    t0 = 1_760_000_000.0
    session = session_with(
        ChatMessage(role="system", content=["You are a support agent."], created_at=t0),
        ChatMessage(id="item_hello", role="assistant", content=["Hello, how can I help?"], created_at=t0 + 1.0,
                    metrics={"started_speaking_at": t0 + 1.0, "stopped_speaking_at": t0 + 2.5}),
        ChatMessage(id="item_ask", role="user", content=["Where is my order?", "It's 4471."], created_at=t0 + 6.0,
                    metrics={"started_speaking_at": t0 + 4.0, "stopped_speaking_at": t0 + 5.5}),
        FunctionCall(call_id="call_1", name="lookup_order", arguments='{"order_id": "4471"}', created_at=t0 + 6.2),
        FunctionCallOutput(call_id="call_1", name="lookup_order", output='{"status": "shipped"}', is_error=False,
                           created_at=t0 + 6.7),
        FunctionCall(call_id="call_2", name="notify", arguments="not json", created_at=t0 + 7.0),
        FunctionCallOutput(call_id="call_2", output="boom", is_error=True, created_at=t0 + 7.1),
        FunctionCall(call_id="call_3", name="pending_tool", arguments="{}", created_at=t0 + 7.5),
        ChatMessage(role="assistant", content=[""], created_at=t0 + 8.0),
        ChatMessage(id="item_shipped", role="assistant", content=["It has shipped."], created_at=t0 + 9.0),
    )
    ctx = FakeCtx(FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"})))
    attach(ctx, session, api_key="k", agent_id=AGENT_UUID, base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown()

    [started] = http.posts("call.started")
    [ended] = http.posts("call.ended")
    body = ended.body
    assert body["event_id"] == "lk:RM_job1:ended"
    assert body["ended_at"].endswith("Z")
    for field in ("call_id", "surface", "source", "schema_version", "testing", "agent_id", "from", "to",
                  "started_at"):
        assert body[field] == started.body[field]
    assert body["transcript"] == [
        {"id": "item_hello", "role": "agent", "text": "Hello, how can I help?", "start_ms": 1000, "end_ms": 2500,
         "interrupted": False},
        {"id": "item_ask", "role": "user", "text": "Where is my order?\nIt's 4471.", "start_ms": 4000,
         "end_ms": 5500},
        {"id": "item_shipped", "role": "agent", "text": "It has shipped.", "start_ms": 9000, "interrupted": False},
    ]
    # a user turn came after "Hello", so each tool call belongs to the next agent turn
    assert body["tool_calls"] == [
        {"id": "call_1", "turn_id": "item_shipped", "name": "lookup_order", "arguments": {"order_id": "4471"},
         "result": '{"status": "shipped"}', "is_error": False, "start_ms": 6200, "end_ms": 6700},
        {"id": "call_2", "turn_id": "item_shipped", "name": "notify", "arguments": "not json", "result": "boom",
         "is_error": True, "start_ms": 7000, "end_ms": 7100},
        {"id": "call_3", "turn_id": "item_shipped", "name": "pending_tool", "arguments": {}, "start_ms": 7500},
    ]
    assert "diagnostics" not in body  # nothing to report: the section is left out, not sent empty


@needs_livekit_1_8
async def test_agent_text_is_sent_without_livekit_expressive_markup(http):
    markup = '<expr type="expression" label="excited"/> Your code is <expr type="spell">A7X9</expr>. Anything else?'
    session = FakeSession(
        chat("user", 'I typed <expr type="x"/> myself', 0, id="item_u1"),
        chat("assistant", markup, 1, id="item_a1"),
        SimpleNamespace(type="message", role="assistant", content="An older SDK's message.", id="item_a2"),
    )
    body = await ended_body(http, session)

    assert [turn["text"] for turn in body["transcript"]] == [
        'I typed <expr type="x"/> myself', "Your code is A7X9. Anything else?", "An older SDK's message."]


async def test_tool_arguments_that_are_not_strict_json_are_sent_as_the_raw_string(http):
    session = FakeSession(
        FunctionCall(call_id="call_1", name="calc", arguments='{"amount": 1e999, "ratio": NaN}', created_at=T0),
        FunctionCall(call_id="call_2", name="calc", arguments='{"amount": -Infinity}', created_at=T0 + 1),
        FunctionCall(call_id="call_3", name="calc", arguments='{"amount": 1.5e3, "items": [1, -0.5]}',
                     created_at=T0 + 2),
    )
    body = await ended_body(http, session)

    assert [call["arguments"] for call in body["tool_calls"]] == [
        '{"amount": 1e999, "ratio": NaN}', '{"amount": -Infinity}', {"amount": 1500.0, "items": [1, -0.5]}]


async def test_a_section_strict_json_cannot_hold_is_left_out_alone(http, caplog):
    session = FakeSession(
        chat("user", "Split it three ways", 0, id="item_u1"),
        SimpleNamespace(type="function_call", call_id="call_1", name="calc", arguments={"ratio": float("nan")},
                        created_at=T0 + 1),
        chat("assistant", "Done.", float("inf"), id="item_a1"),  # no usable time: sent without start_ms
        llm=SimpleNamespace(provider="openai", model="gpt-4.1"),
    )
    body = await ended_body(http, session)

    assert "tool_calls" not in body
    assert body["transcript"] == [{"id": "item_u1", "role": "user", "text": "Split it three ways", "start_ms": 0},
                                  {"id": "item_a1", "role": "agent", "text": "Done.", "interrupted": False}]
    assert body["diagnostics"]["configuration"]["llm"] == {"provider": "openai", "model": "gpt-4.1"}
    assert caplog.text.count("tool_calls left out of call.ended") == 1


@pytest.fixture
async def strict_server():
    """A local SuperBryn stand-in on 127.0.0.1 that parses bodies as strictly as JSON.parse."""
    received = []

    def refuse(token):
        raise ValueError(f"{token} is not JSON")

    async def agent_data(request):
        received.append((request.content_type, json.loads(await request.text(), parse_constant=refuse)))
        return web.json_response(LINKED[1])

    app = web.Application()
    app.router.add_post("/public-api/v1/calls/agent-data", agent_data)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    host, port = runner.addresses[0][:2]
    yield f"http://{host}:{port}", received
    await runner.cleanup()


async def test_the_real_transport_sends_strict_json(strict_server):
    base, received = strict_server
    session = FakeSession(
        chat("user", "Split 10 dollars three ways", 0, id="item_u1"),
        FunctionCall(call_id="call_1", name="calc", arguments='{"amount": 1e999, "ratio": NaN}', created_at=T0 + 1),
    )
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=base)
    assert await recognised(ctx) is True
    await ctx.shutdown()

    assert [(kind, body["event"]) for kind, body in received] == [
        ("application/json", "call.started"), ("application/json", "call.ended")]
    assert received[1][1]["tool_calls"][0]["arguments"] == '{"amount": 1e999, "ratio": NaN}'


async def test_text_is_cut_in_utf16_units_like_the_server(http):
    emoji_text = "a" + "\U0001F600" * 10_000  # 10,001 characters, 20,001 UTF-16 units
    ascii_text = "b" * 20_000
    session = session_with(ChatMessage(role="user", content=[emoji_text]),
                           ChatMessage(role="assistant", content=[ascii_text]))
    ctx = FakeCtx(FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"})))
    attach(ctx, session, api_key="k", base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    user, agent = (turn["text"] for turn in ended.body["transcript"])
    assert len(user.encode("utf-16-le")) // 2 <= 20_000
    assert user == "a" + "\U0001F600" * 9_999  # the split surrogate pair is dropped
    assert agent == ascii_text


async def test_turn_ids_confidence_and_interrupted(http):
    session = FakeSession(
        chat("user", "Hi", 0, id="item_u1", transcript_confidence=0.93),
        chat("assistant", "Hello! How can", 1, id="item_a1", interrupted=True, transcript_confidence=0.5),
        chat("user", "Sorry, go on", 2, id="item_u2", transcript_confidence=1.2),
        chat("user", "Hello?", 3, id="item_u3", transcript_confidence=-0.1),
        chat("assistant", "How can I help?", 4, id="i" * 250),
    )
    body = await ended_body(http, session)

    # confidence is for user turns in [0, 1] only; interrupted is for agent turns
    assert body["transcript"] == [
        {"id": "item_u1", "role": "user", "text": "Hi", "start_ms": 0, "confidence": 0.93},
        {"id": "item_a1", "role": "agent", "text": "Hello! How can", "start_ms": 1000, "interrupted": True},
        {"id": "item_u2", "role": "user", "text": "Sorry, go on", "start_ms": 2000},
        {"id": "item_u3", "role": "user", "text": "Hello?", "start_ms": 3000},
        {"id": "i" * 200, "role": "agent", "text": "How can I help?", "start_ms": 4000, "interrupted": False},
    ]


async def test_turn_latencies_pair_each_reply_with_the_user_turn_before_it(http):
    session = FakeSession(
        chat("assistant", "Hi, how can I help?", 0, id="item_greet", metrics={"tts_node_ttfb": 0.2004}),
        chat("user", "Uh", 2, id="item_u0", metrics={"transcription_delay": 0.5, "end_of_turn_delay": 0.9}),
        chat("user", "Where is my order?", 3, id="item_u1",
             metrics={"transcription_delay": 0.1044, "end_of_turn_delay": 0.2064}),
        chat("assistant", "It has shipped.", 5, id="item_a1",
             metrics={"llm_node_ttft": 0.7284, "tts_node_ttfb": 0.1636, "e2e_latency": 1.1027}),
        chat("user", "Thanks", 6, id="item_u2", metrics={"transcription_delay": 0.3, "end_of_turn_delay": 0.2}),
        chat("assistant", "", 7, id="item_silent", metrics={"llm_node_ttft": 0.5004}),
        chat("assistant", "Bye!", 8, id="item_a2"),  # no timings, and the user turn is already answered
    )
    diagnostics = (await ended_body(http, session))["diagnostics"]

    assert diagnostics["turn_latencies"] == [
        {"turn_id": "item_greet", "voice_ms": 200},  # answers no user message
        # the last user message counts; endpointing leaves out the transcript wait (206.4 - 104.4 ms)
        {"turn_id": "item_a1", "transcriber_ms": 104, "endpointing_ms": 102, "model_ms": 728,
         "voice_ms": 164, "turn_ms": 1103},
        {"transcriber_ms": 300, "endpointing_ms": 0, "model_ms": 500},  # no text, so no turn_id
    ]


async def test_tool_call_turn_id_is_the_agent_turn_before_it_else_the_next_one(http):
    session = FakeSession(
        chat("user", "Where is my order?", 0, id="item_u1"),
        chat("assistant", "Let me check.", 1, id="item_a1"),
        FunctionCall(call_id="call_1", name="lookup_order", arguments="{}", created_at=T0 + 2),
        chat("assistant", "", 3, id="item_empty"),  # not a turn
        FunctionCall(call_id="call_2", name="notify", arguments="{}", created_at=T0 + 4),
        chat("user", "And my refund?", 5, id="item_u2"),
        FunctionCall(call_id="call_3", name="lookup_refund", arguments="{}", created_at=T0 + 6),
        chat("assistant", "Your refund is on its way.", 7, id="item_a2"),
        chat("user", "Bye", 8, id="item_u3"),
        FunctionCall(call_id="call_4", name="log_call", arguments="{}", created_at=T0 + 9),
    )
    body = await ended_body(http, session)

    assert [(call["id"], call.get("turn_id")) for call in body["tool_calls"]] == [
        ("call_1", "item_a1"), ("call_2", "item_a1"), ("call_3", "item_a2"), ("call_4", None)]
    assert "turn_id" not in body["tool_calls"][3]


@pytest.mark.parametrize(("model", "pipeline"), [
    (FakeRealtime(), "realtime"),
    (SimpleNamespace(provider="openai", model="gpt-4.1"), "stt_llm_tts"),
    (None, None),
], ids=["realtime", "stt-llm-tts", "no-llm"])
async def test_pipeline(http, model, pipeline):
    body = await ended_body(http, FakeSession(llm=model))

    assert body.get("diagnostics", {}).get("pipeline") == pipeline
    if model is not None:
        assert body["diagnostics"]["configuration"]["llm"] == {"provider": "openai", "model": model.model}


async def test_configuration_from_turn_handling(http):
    def adapter(**inner):  # a wrapper that names itself, not the provider
        return SimpleNamespace(provider="wrapper", model="wrapper", **inner)

    session = FakeSession(
        stt=adapter(_stt_instances=[SimpleNamespace(provider="deepgram", model="nova-3")]),
        llm=adapter(_llm_instances=[SimpleNamespace(provider="openai", model="gpt-4.1")]),
        tts=adapter(_wrapped_tts=SimpleNamespace(provider="cartesia", model="sonic-2", voice=None,
                                                 _opts=SimpleNamespace(voice="voice-id"))),
        options=SimpleNamespace(turn_handling={
            "turn_detection": NamedDetector(),
            "endpointing": {"mode": "fixed", "min_delay": 0.3, "max_delay": 2.5, "alpha": 0.9},
            "interruption": {"enabled": True, "min_duration": 0.5, "min_words": 0, "mode": "adaptive"},
            "preemptive_generation": {"enabled": False, "preemptive_tts": False, "max_retries": 3},
        }),
    )
    diagnostics = (await ended_body(http, session))["diagnostics"]

    assert diagnostics["pipeline"] == "stt_llm_tts"
    assert diagnostics["configuration"] == {
        "stt": {"provider": "deepgram", "model": "nova-3"},
        "llm": {"provider": "openai", "model": "gpt-4.1"},
        "tts": {"provider": "cartesia", "model": "sonic-2", "voice": "voice-id"},
        "turn_detection": {"type": "turn-detector-v1-mini", "min_end_of_turn_ms": 300, "max_end_of_turn_ms": 2500,
                           "preemptive_generation": False},
        "interruptions": {"enabled": True, "min_duration_ms": 500, "min_words": 0},
    }


@pytest.mark.parametrize(("detector", "expected"), [("vad", "vad"), (EnglishModel(), "EnglishModel")],
                         ids=["mode", "class-name"])
async def test_turn_detection_type(http, detector, expected):
    options = SimpleNamespace(turn_handling={"turn_detection": detector, "endpointing": {"min_delay": 0.5}})
    diagnostics = (await ended_body(http, FakeSession(options=options)))["diagnostics"]

    assert diagnostics["configuration"]["turn_detection"] == {"type": expected, "min_end_of_turn_ms": 500}


async def test_configuration_from_older_options_and_usage(http):
    session = FakeSession(
        llm=SimpleNamespace(provider="unknown", model="unknown"),  # a custom LLM that names neither
        tts=SimpleNamespace(provider="elevenlabs", model="", voice_id="voice-1"),
        options=SimpleNamespace(min_endpointing_delay=0.4, max_endpointing_delay=5.0, allow_interruptions=False,
                                min_interruption_duration=0.7, min_interruption_words=2, preemptive_generation=True),
        usage=AgentSessionUsage(model_usage=[
            LLMModelUsage(provider="", model=""),
            LLMModelUsage(provider="openai", model="gpt-4.1-mini"),
            TTSModelUsage(provider="elevenlabs", model="eleven_flash_v2_5"),
        ]),
    )
    configuration = (await ended_body(http, session))["diagnostics"]["configuration"]

    assert configuration == {
        "llm": {"provider": "openai", "model": "gpt-4.1-mini"},
        "tts": {"provider": "elevenlabs", "model": "eleven_flash_v2_5", "voice": "voice-1"},
        "turn_detection": {"min_end_of_turn_ms": 400, "max_end_of_turn_ms": 5000, "preemptive_generation": True},
        "interruptions": {"enabled": False, "min_duration_ms": 700, "min_words": 2},
    }


@needs_livekit_1_8
async def test_a_realtime_fallback_adapter_reports_its_model(http):
    adapter = llm.RealtimeModelFallbackAdapter([Realtime("gpt-realtime"), Realtime("gpt-realtime-mini")])
    diagnostics = (await ended_body(http, FakeSession(llm=adapter)))["diagnostics"]

    assert diagnostics["pipeline"] == "realtime"
    assert diagnostics["configuration"]["llm"] == {"provider": "openai", "model": "gpt-realtime"}


async def test_the_current_agents_own_models_are_reported(http):
    agent = Agent(instructions="x", llm=SimpleNamespace(provider="openai", model="gpt-4o"),
                  tts=SimpleNamespace(provider="cartesia", model="sonic-2", voice_id="agent-voice"))
    session = FakeSession(
        agent=agent,
        stt=SimpleNamespace(provider="deepgram", model="nova-3"),  # the agent has none: the session's runs
        llm=SimpleNamespace(provider="openai", model="gpt-4.1-mini"),
        tts=SimpleNamespace(provider="cartesia", model="sonic-2", voice_id="session-voice"),
    )
    configuration = (await ended_body(http, session))["diagnostics"]["configuration"]

    assert configuration["stt"] == {"provider": "deepgram", "model": "nova-3"}
    assert configuration["llm"] == {"provider": "openai", "model": "gpt-4o"}
    assert configuration["tts"] == {"provider": "cartesia", "model": "sonic-2", "voice": "agent-voice"}

    only_the_agent = FakeSession(agent=Agent(instructions="x", llm=FakeRealtime()))
    assert (await ended_body(http, only_the_agent))["diagnostics"]["pipeline"] == "realtime"


@pytest.mark.parametrize(("tokens", "expected"), [
    ((40, 300), {"provider": "openai", "model": "gpt-4o"}),
    ((300, 40), {"provider": "openai", "model": "gpt-4.1-mini"}),
], ids=["most-tokens-elsewhere", "most-tokens-configured"])
async def test_the_llm_is_named_by_the_model_most_tokens_went_to(http, tokens, expected):
    usage = [LLMModelUsage(provider="openai", model="gpt-4.1-mini", input_tokens=tokens[0], output_tokens=5),
             LLMModelUsage(provider="openai", model="gpt-4o", input_tokens=tokens[1], output_tokens=5)]
    session = FakeSession(llm=SimpleNamespace(provider="openai", model="gpt-4.1-mini"),
                          usage=AgentSessionUsage(model_usage=usage))
    configuration = (await ended_body(http, session))["diagnostics"]["configuration"]

    assert configuration["llm"] == expected


async def test_live_turn_detection_is_what_livekit_resolved():  # real AgentSession + AgentActivity, not started
    realtime = AgentSession(llm=Realtime(), vad=None)  # no turn detection given, as realtime agents usually do
    assert call_report._live_turn_detection(AgentActivity(Agent(instructions="x"), realtime)) == {
        "type": "realtime_llm"}

    stt = SimpleNamespace(capabilities=SimpleNamespace(streaming=True, aligned_transcript=False))
    session = AgentSession(stt=stt, vad=None,
                           turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0.4}})
    # LiveKit drafts replies early unless told not to
    assert call_report._live_turn_detection(AgentActivity(Agent(instructions="x"), session)) == {
        "type": "stt", "min_end_of_turn_ms": 400, "max_end_of_turn_ms": 3000, "preemptive_generation": True}
    agent = Agent(instructions="x", turn_handling={"turn_detection": "manual", "preemptive_generation": {"enabled": False},
                                                   "endpointing": {"min_delay": 1.2, "max_delay": 4.0}})
    assert call_report._live_turn_detection(AgentActivity(agent, session)) == {
        "type": "manual", "min_end_of_turn_ms": 1200, "max_end_of_turn_ms": 4000, "preemptive_generation": False}

    assert call_report._live_turn_detection(None) is None
    assert call_report._live_turn_detection(SimpleNamespace(_turn_detection="vad")) is None  # another SDK shape


async def test_live_turn_detection_before_1_8():  # an AgentActivity without _rt_turn_detection_enabled
    # A turn detector the SDK dropped (no VAD) is reported as none, with the delays that still apply.
    older = SimpleNamespace(_turn_detection=None, min_endpointing_delay=0.3, max_endpointing_delay=2.5, llm=None)
    assert call_report._live_turn_detection(older) == {"min_end_of_turn_ms": 300, "max_end_of_turn_ms": 2500}
    older._turn_detection = "vad"
    assert call_report._live_turn_detection(older) == {"type": "vad", "min_end_of_turn_ms": 300, "max_end_of_turn_ms": 2500}
    older._session = SimpleNamespace(options=SimpleNamespace(preemptive_generation=False))  # a bool on the session
    assert call_report._live_turn_detection(older)["preemptive_generation"] is False
    older.llm = Realtime()  # server-side turn detection takes the turns
    assert call_report._live_turn_detection(older) == {"type": "realtime_llm"}


async def test_turn_detection_seen_during_the_call_wins_over_the_options(http):
    options = SimpleNamespace(turn_handling={"turn_detection": NamedDetector(),
                                             "endpointing": {"min_delay": 0.3, "max_delay": 2.5}})
    session = FakeSession(options=options)
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    assert await recognised(ctx) is True

    session._activity = SimpleNamespace(_rt_turn_detection_enabled=True)  # the realtime model takes turns
    session.emit("agent_state_changed", AgentStateChangedEvent(old_state="initializing", new_state="listening"))
    session._activity = None  # closed before the job's shutdown
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert ended.body["diagnostics"]["configuration"]["turn_detection"] == {"type": "realtime_llm"}


def llm_event(start, end, speech_id=None, ttft=0.3, prompt=0, cached=0, completion=0):
    """A recorded metrics_collected event for one LLM request, stamped when it ended as LiveKit does."""
    metrics = SimpleNamespace(type="llm_metrics", timestamp=T0 + end, duration=end - start, ttft=ttft,
                              speech_id=speech_id, prompt_tokens=prompt, prompt_cached_tokens=cached,
                              completion_tokens=completion)
    return SimpleNamespace(type="metrics_collected", metrics=metrics)


def speech_event(speech_id, *items):
    """A recorded speech_created event; its handle lists the replies and tool calls the speech added."""
    return SimpleNamespace(type="speech_created", speech_handle=SimpleNamespace(id=speech_id, chat_items=list(items)))


async def test_each_llm_request_gets_what_became_of_its_answer(http):
    user = chat("user", "Where is my order?", 0.5)
    lookup = FunctionCall(call_id="call_1", name="lookup_order", arguments="{}", created_at=T0 + 1.6)
    heard = chat("assistant", "Your order 4471 has shipped.", 2.5)
    cut = chat("assistant", "Sure, your ord", 5.2, interrupted=True)
    session = FakeSession(user, lookup, heard, cut)
    session._recorded_events = [
        speech_event("sp1", lookup, heard),
        speech_event("sp2", cut),
        llm_event(1.0, 1.5, "sp1", prompt=900, cached=600, completion=20),  # only a tool call
        llm_event(2.0, 2.4, "sp1", prompt=950, completion=12),  # the answer after it, heard in full
        llm_event(4.0, 4.3, "sp3", ttft=0.2),  # an answer thrown away before it played
        llm_event(4.6, 5.0, "sp2", prompt=980, completion=30),  # played, then the caller cut in
        llm_event(6.0, 6.1, "sp4", ttft=-1),  # nothing came back
    ]
    body = await ended_body(http, session)

    # one clock for the transcript and the requests: ms from when the caller joined (T0)
    assert body["diagnostics"]["llm_requests"] == [
        {"start_ms": 1000, "outcome": "tool_call", "input_tokens": 900, "cached_input_tokens": 600,
         "output_tokens": 20},
        {"start_ms": 2000, "outcome": "spoken", "input_tokens": 950, "cached_input_tokens": 0, "output_tokens": 12},
        {"start_ms": 4000, "outcome": "not_spoken"},
        {"start_ms": 4600, "outcome": "cut_off", "input_tokens": 980, "cached_input_tokens": 0, "output_tokens": 30},
        {"start_ms": 6000, "outcome": "stopped"},
    ]
    assert [turn["start_ms"] for turn in body["transcript"]] == [500, 2500, 5200]


async def test_a_request_the_llm_failed_on_is_stopped_whatever_its_first_token_time(http):
    hello = chat("assistant", "Hello, how can I help?", 3.6)
    session = FakeSession(chat("user", "Hi", 0.0), hello)
    session._recorded_events = [
        speech_event("sp2", hello),
        llm_event(1.0, 3.0, "sp1", ttft=0.4),  # no output: the gateway refused it
        error_event("llm_error", 3.0, "429 quota exceeded", True, "livekit.LLM"),
        llm_event(3.1, 3.5, "sp2", prompt=900, completion=10),  # the retry, heard
    ]
    body = await ended_body(http, session)

    assert body["diagnostics"]["llm_requests"] == [
        {"start_ms": 1000, "outcome": "stopped"},
        {"start_ms": 3100, "outcome": "spoken", "input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 10},
    ]


async def test_realtime_responses_own_the_replies_that_start_after_them(http):
    session = FakeSession(chat("assistant", "Hello, Acme support.", 0.8), chat("user", "Hi", 2.0),
                          chat("assistant", "Let me check that", 3.4, interrupted=True))
    session._recorded_events = [
        SimpleNamespace(type="metrics_collected", metrics=SimpleNamespace(
            type="realtime_model_metrics", timestamp=T0 + created, duration=0.9, ttft=0.4, input_tokens=given,
            output_tokens=produced, input_token_details=SimpleNamespace(cached_tokens=cached)))
        for created, given, produced, cached in ((0.3, 500, 40, 0), (2.9, 700, 55, 300))
    ]
    body = await ended_body(http, session)

    assert body["diagnostics"]["llm_requests"] == [
        {"start_ms": 300, "outcome": "spoken", "input_tokens": 500, "cached_input_tokens": 0, "output_tokens": 40},
        {"start_ms": 2900, "outcome": "cut_off", "input_tokens": 700, "cached_input_tokens": 300, "output_tokens": 55},
    ]


def rt_event(request_id, created, ttft=0.4, duration=1.0, cancelled=False, given=0, produced=0, cached=0,
             audio_in=0, audio_out=0, session_duration=0.0):
    """A recorded metrics_collected event for one realtime response, stamped when it was created."""
    metrics = SimpleNamespace(
        type="realtime_model_metrics", request_id=request_id, timestamp=created, duration=duration, ttft=ttft,
        cancelled=cancelled, input_tokens=given, output_tokens=produced, session_duration=session_duration,
        input_token_details=SimpleNamespace(cached_tokens=cached, audio_tokens=audio_in, text_tokens=given - audio_in),
        output_token_details=SimpleNamespace(audio_tokens=audio_out, text_tokens=produced - audio_out))
    return SimpleNamespace(type="metrics_collected", metrics=metrics)


def caller(state, at):
    """A recorded user_state_changed event: the caller started ("speaking") or stopped ("listening") talking."""
    return SimpleNamespace(type="user_state_changed", new_state=state, created_at=T0 + at)


def heard(text, at, *request_ids, **fields):
    """An agent reply the caller started hearing at `at`, with the provider response IDs LiveKit keeps on it."""
    metrics = {"started_speaking_at": T0 + at, **({"provider_request_ids": list(request_ids)} if request_ids else {})}
    return chat("assistant", text, at, metrics=metrics, **fields)


async def test_realtime_reply_waits_run_from_the_caller_stopping(http):
    session = FakeSession(
        heard("Hello, Acme support.", 0.4, "resp_0", id="item_greet"),
        chat("user", "Where is my order?", 2.0, id="item_u1", transcript_confidence=0.87,
             metrics={"started_speaking_at": T0 + 2.0}),
        heard("It has shipped.", 3.8, "resp_1", id="item_a1"),
        heard("Anything else?", 5.5, "resp_2", id="item_a2"),
        chat("user", "No thanks", 7.0, id="item_u2", transcript_confidence=1.0,
             metrics={"started_speaking_at": T0 + 7.0}),
        heard("Bye!", 8.1, id="item_a3"),  # no provider ID: its wait counts, the model's part can't be told
        llm=Realtime())
    session._recorded_events = [
        rt_event("resp_0", T0 + 0.0, ttft=0.3),
        caller("speaking", 2.0), caller("listening", 3.0),
        rt_event("resp_1", T0 + 3.2, ttft=0.5),
        rt_event("resp_2", T0 + 5.0, ttft=0.4),
        caller("speaking", 7.0), caller("listening", 7.5),
        rt_event("resp_3", T0 + 7.7, ttft=0.3),
    ]
    body = await ended_body(http, session)

    # the greeting answers no caller speech, and a second reply in the same turn none either
    assert body["diagnostics"]["turn_latencies"] == [
        {"turn_id": "item_a1", "model_ms": 700, "turn_ms": 800},
        {"turn_id": "item_a3", "turn_ms": 600},
    ]
    # a caller line ends when the caller stopped; LiveKit's stand-in confidence of 1.0 is left out
    assert [turn for turn in body["transcript"] if turn["role"] == "user"] == [
        {"id": "item_u1", "role": "user", "text": "Where is my order?", "start_ms": 2000, "end_ms": 3000,
         "confidence": 0.87},
        {"id": "item_u2", "role": "user", "text": "No thanks", "start_ms": 7000, "end_ms": 7500},
    ]


async def test_a_realtime_caller_line_stamped_after_the_caller_stopped_moves_back_to_their_speech(http):
    # Gemini stamps a caller line with the start of the reply it prompted
    session = FakeSession(
        heard("Hello, Acme support.", 0.3, "GR_0", id="item_greet"),
        chat("user", "Where is my order?", 2.6, id="item_u1", transcript_confidence=1.0,
             metrics={"started_speaking_at": T0 + 2.6}),
        heard("Let me check.", 2.8, "GR_1", id="item_a1"),
        llm=Realtime())
    session._recorded_events = [
        rt_event("GR_0", T0 + 0.0, ttft=0.02),
        caller("speaking", 1.0), caller("listening", 2.0),
        rt_event("GR_1", T0 + 2.6, ttft=0.05),
    ]
    body = await ended_body(http, session)

    assert [turn for turn in body["transcript"] if turn["role"] == "user"] == [
        {"id": "item_u1", "role": "user", "text": "Where is my order?", "start_ms": 1000, "end_ms": 2000},
    ]
    assert body["diagnostics"]["turn_latencies"] == [{"turn_id": "item_a1", "model_ms": 650, "turn_ms": 800}]


def nova_session(**attributes):
    """A Nova Sonic call: responses on the monotonic clock (0.1 s is T0 + 0.1, as the clocks read T0 + 100 and 100)
    stamped while the caller still talks, the caller marked stopped only once the reply starts, and a LiveKit ID on
    the reply that no response carries."""
    session = FakeSession(
        heard("Hello, Skyway Airlines.", 0.5, "lk_greet", id="item_greet"),
        chat("user", "My flight got cancelled?", 1.0, id="item_u1", metrics={"started_speaking_at": T0 + 1.0}),
        heard("Yes, I can rebook you.", 3.1, "lk_reply", id="item_a1"),
        llm=Realtime("amazon.nova-2-5-sonic", "Amazon"), **attributes)
    session._recorded_events = [
        rt_event("cmp_0", 0.1, ttft=0.2), caller("speaking", 1.0), rt_event("cmp_0", 1.2, ttft=0.6),
        caller("listening", 2.9)]
    return session


async def test_nova_sonic_replies_get_no_wait_and_its_caller_lines_no_times_without_a_vad(http, monkeypatch):
    # its "caller stopped" is when its own reply started: a wait from it would be the playback delay alone
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: T0 + 100.0, monotonic=lambda: 100.0))
    body = await ended_body(http, nova_session())

    assert "turn_latencies" not in body["diagnostics"]
    assert body["transcript"][1] == {"id": "item_u1", "role": "user", "text": "My flight got cancelled?"}


async def test_a_realtime_reply_with_a_vad_gets_its_wait_without_the_models_part(http, monkeypatch):
    # with a VAD on the session the caller's stops are real: the reply pairs with the last one before it was heard
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: T0 + 100.0, monotonic=lambda: 100.0))
    diagnostics = (await ended_body(http, nova_session(vad=object())))["diagnostics"]

    assert diagnostics["turn_latencies"] == [{"turn_id": "item_a1", "turn_ms": 200}]


async def test_livekits_default_vad_does_not_make_nova_caller_times_real(http, monkeypatch):
    # LiveKit 1.8 gives every session a VAD when none is passed; realtime models never run it, so the stand-in stops
    # stay stand-ins (staging Nova calls sent ~115 ms waits and placeholder caller times on 0.3.1)
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: T0 + 100.0, monotonic=lambda: 100.0))
    body = await ended_body(http, nova_session(vad=object(), _using_default_vad=True))

    assert "turn_latencies" not in body["diagnostics"]
    assert body["transcript"][1] == {"id": "item_u1", "role": "user", "text": "My flight got cancelled?"}


async def test_a_real_sessions_default_vad_is_not_a_given_one():  # real AgentSession, not started
    nova = Realtime("amazon.nova-2-5-sonic", "Amazon")
    assert call_report._caller_times_real(AgentSession(llm=nova)) is False
    assert call_report._caller_times_real(AgentSession(llm=nova, vad=object())) is True
    assert call_report._caller_times_real(AgentSession(llm=Realtime())) is True  # OpenAI's own stops are real


def test_the_agents_own_vad_wins_over_the_sessions():
    session = lambda agent, **kw: FakeSession(agent=agent, **kw)  # noqa: E731
    assert call_report._given_vad(session(SimpleNamespace(vad=object()), vad=object(), _using_default_vad=True))
    assert not call_report._given_vad(session(SimpleNamespace(vad=None), vad=object(), _using_default_vad=False))
    assert call_report._given_vad(session(None, vad=object(), _using_default_vad=False))
    assert not call_report._given_vad(session(None, vad=None))


async def test_gemini_live_replies_found_by_id_still_get_no_wait_its_caller_stops_are_stand_ins(http):
    # Gemini Live's IDs match its responses, but LiveKit marks the caller stopped as its reply starts (1-230 ms
    # before it on recorded calls), so neither a wait nor caller times can be told
    session = FakeSession(
        chat("user", "Where is my order?", 2.0, id="item_u1", metrics={"started_speaking_at": T0 + 2.0}),
        heard("It has shipped.", 3.4, "resp_1", id="item_a1"),
        llm=Realtime("google/gemini-3.8-live", "livekit"))
    session._recorded_events = [caller("speaking", 2.0), rt_event("resp_1", T0 + 2.5, ttft=0.8),
                                caller("listening", 3.3)]
    body = await ended_body(http, session)

    assert "turn_latencies" not in body["diagnostics"]
    assert body["transcript"][0] == {"id": "item_u1", "role": "user", "text": "Where is my order?"}


async def test_the_empty_record_a_realtime_session_opens_with_and_shared_ids_are_no_requests(http):
    # every realtime session opens with a record of no request, tokens or ttft; GPT Live keeps one ID for the session
    opening = rt_event(None, T0 + 0.0, ttft=-1, duration=0.0)
    session = FakeSession(chat("user", "Hi", 0.0), heard("Hello, Acme support.", 0.5, "resp_1"), llm=Realtime())
    session._recorded_events = [opening, rt_event("resp_1", T0 + 0.2, ttft=0.3, given=50, produced=20)]
    assert (await ended_body(http, session))["diagnostics"]["llm_requests"] == [
        {"start_ms": 200, "outcome": "spoken", "input_tokens": 50, "cached_input_tokens": 0, "output_tokens": 20}]

    live = FakeSession(heard("Hi.", 1.5, "item_1"), heard("Sure.", 6.0, "item_2"), llm=Realtime("openai/gpt-live-1"))
    live._recorded_events = [opening, rt_event("live_1", T0 + 1.0, ttft=-1), rt_event("live_1", T0 + 5.5, ttft=-1)]
    assert "llm_requests" not in (await ended_body(http, live))["diagnostics"]


async def test_a_realtime_reply_found_by_id_pairs_with_the_stop_before_its_response(http):
    # the caller speaks again between the response's creation and its playback: that later speech is the next
    # reply's to answer
    session = FakeSession(
        chat("user", "Where is my order?", 2.0, id="item_u1", metrics={"started_speaking_at": T0 + 2.0}),
        heard("It has shipped.", 3.8, "resp_1", id="item_a1"),
        chat("user", "Okay", 3.3, id="item_u2", metrics={"started_speaking_at": T0 + 3.3}),
        heard("Anything else?", 5.0, "resp_2", id="item_a2"),
        llm=Realtime())
    session._recorded_events = [
        caller("speaking", 2.0), caller("listening", 3.0),
        rt_event("resp_1", T0 + 3.2, ttft=0.5),
        caller("speaking", 3.3), caller("listening", 3.5),
        rt_event("resp_2", T0 + 4.6, ttft=0.3),
    ]
    diagnostics = (await ended_body(http, session))["diagnostics"]

    assert diagnostics["turn_latencies"] == [
        {"turn_id": "item_a1", "model_ms": 700, "turn_ms": 800},
        {"turn_id": "item_a2", "model_ms": 1400, "turn_ms": 1500},
    ]


async def test_a_realtime_model_reports_its_voice_and_only_whether_it_can_be_interrupted(http):
    model = Realtime()
    model._opts = SimpleNamespace(voice="marin")
    options = SimpleNamespace(turn_handling={"interruption": {"enabled": True, "min_duration": 0.5, "min_words": 2}})
    session = FakeSession(heard("Hello, Acme support.", 0.5, "resp_1"), llm=model, options=options)
    configuration = (await ended_body(http, session))["diagnostics"]["configuration"]

    # the model's own speech detection interrupts it: LiveKit's duration and word gates don't apply
    assert configuration["interruptions"] == {"enabled": True}
    assert configuration["tts"] == {"provider": "openai", "model": "gpt-realtime", "voice": "marin"}


async def test_zero_seconds_of_stt_audio_is_not_sent(http):
    # OpenAI's realtime transcription bills tokens and reports 0 s of audio
    usage = SimpleNamespace(model_usage=[
        SimpleNamespace(type="stt_usage", provider="openai", model="gpt-4o-transcribe", audio_duration=0.0),
        LLMModelUsage(provider="openai", model="gpt-realtime", input_tokens=10, output_tokens=5)])
    session = FakeSession(heard("Hello, Acme support.", 0.5, "resp_1"), llm=Realtime(), usage=usage)
    diagnostics = (await ended_body(http, session))["diagnostics"]

    assert diagnostics["usage"] == {"llm_input_tokens": 10, "llm_cached_input_tokens": 0, "llm_output_tokens": 5}


async def test_a_realtime_response_that_failed_while_it_ran_is_stopped(http):
    # the failure comes 1.5 s after the response was created, within the 2 s it ran
    session = FakeSession(chat("user", "Hi", 0.0), llm=Realtime())
    error = SimpleNamespace(type="realtime_model_error", timestamp=T0 + 2.5, label="openai.realtime",
                            error=RuntimeError("server error"), recoverable=False)
    session._recorded_events = [rt_event("resp_1", T0 + 1.0, ttft=0.3, duration=2.0),
                                SimpleNamespace(type="error", created_at=T0 + 2.5, error=error)]
    diagnostics = (await ended_body(http, session))["diagnostics"]

    assert diagnostics["llm_requests"] == [{"start_ms": 1000, "outcome": "stopped"}]


async def test_realtime_responses_and_usage_fill_only_the_existing_fields(http):
    usage = AgentSessionUsage(model_usage=[LLMModelUsage(
        provider="openai", model="gpt-realtime", input_tokens=1200, input_cached_tokens=300, input_audio_tokens=1000,
        input_text_tokens=200, output_tokens=400, output_audio_tokens=350, output_text_tokens=50)])
    session = FakeSession(heard("Hello, Acme support.", 0.5, "resp_1"), llm=Realtime(), usage=usage)
    session._recorded_events = [
        rt_event("resp_1", T0 + 0.0, given=1200, produced=400, cached=300, audio_in=1000, audio_out=350),
        rt_event("resp_2", T0 + 3.0, ttft=-1, cancelled=True),  # the caller cut in before any audio
    ]
    diagnostics = (await ended_body(http, session))["diagnostics"]

    # audio tokens count in the totals; a response the caller cut off before any audio was stopped
    assert diagnostics["llm_requests"] == [
        {"start_ms": 0, "outcome": "spoken", "input_tokens": 1200, "cached_input_tokens": 300, "output_tokens": 400},
        {"start_ms": 3000, "outcome": "stopped"},
    ]
    # speech and text tokens are priced apart, so a realtime model's split is sent too
    assert diagnostics["usage"] == {
        "llm_input_tokens": 1200, "llm_cached_input_tokens": 300, "llm_output_tokens": 400,
        "llm_input_audio_tokens": 1000, "llm_input_text_tokens": 200, "llm_output_audio_tokens": 350,
        "llm_output_text_tokens": 50}


# Nova Sonic's 38 usage pieces from a recorded 2.5 Sonic call, one completionId for them all: (monotonic s, input
# text, input audio, output text, output audio, ttft). A reply's output pieces carry one ttft, the caller's input none.
NOVA_PIECES = [
    (4.913, 22, 0, 0, 0, 0.0), (5.023, 0, 157, 0, 0, 0.0), (5.024, 1, 0, 0, 0, 0.0), (5.131, 12, 0, 0, 0, 0.0),
    (5.134, 43, 0, 0, 0, 0.0), (5.172, 4, 0, 0, 0, 0.0), (5.275, 302, 0, 0, 0, 0.0), (5.551, 0, 0, 13, 0, 0.0),
    (5.82, 0, 0, 2, 18, 0.0028848648071289062), (6.093, 0, 0, 0, 20, 0.0028848648071289062),
    (6.678, 0, 0, 0, 20, 0.0028848648071289062), (20.617, 0, 100, 0, 0, 0.0), (20.618, 3, 0, 0, 0, 0.0),
    (20.777, 0, 0, 15, 5, 0.0), (20.834, 0, 7, 0, 0, 0.0), (20.845, 3, 0, 0, 0, 0.0), (21.033, 0, 3, 0, 0, 0.0),
    (21.033, 3, 0, 0, 0, 0.0), (21.055, 1, 0, 0, 0, 0.0), (21.591, 0, 0, 11, 0, 0.0),
    (21.632, 0, 0, 6, 14, 0.0009641647338867188), (23.507, 0, 0, 0, 20, 0.0009641647338867188),
    (24.755, 0, 0, 4, 16, 0.0009641647338867188), (31.756, 0, 81, 0, 0, 0.0), (31.756, 3, 0, 0, 0, 0.0),
    (32.327, 0, 0, 12, 0, 0.0), (32.384, 0, 0, 15, 5, 0.0023860931396484375),
    (32.798, 0, 0, 0, 20, 0.0023860931396484375), (33.228, 0, 0, 0, 20, 0.0023860931396484375),
    (33.857, 0, 0, 0, 20, 0.0023860931396484375), (46.319, 0, 100, 0, 0, 0.0), (46.319, 3, 0, 0, 0, 0.0),
    (46.445, 0, 0, 15, 5, 0.0), (46.956, 0, 17, 0, 0, 0.0), (46.956, 3, 0, 0, 0, 0.0), (46.974, 1, 0, 0, 0, 0.0),
    (47.347, 0, 0, 11, 0, 0.0), (47.496, 0, 0, 7, 13, 0.0012772083282470703),
]


async def test_nova_sonic_usage_pieces_are_no_requests_but_their_tokens_count(http, monkeypatch):
    # its pieces are stamped as each reply's audio arrives (staging calls: within ~40 ms of the reply starting to
    # play), so no request's start or outcome can be told; usage keeps the session's tokens and their split
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: T0 + 100.0, monotonic=lambda: 100.0))
    usage = AgentSessionUsage(model_usage=[LLMModelUsage(
        provider="Amazon", model="amazon.nova-2-5-sonic", input_tokens=869, input_audio_tokens=465,
        input_text_tokens=404, output_tokens=307, output_audio_tokens=196, output_text_tokens=111)])
    session = FakeSession(
        chat("user", "", 0.0),  # the session's start
        heard("Hello, this is the Brief Voice AI model.", 7.29, "lk_greet"),
        chat("user", "hello who are you and what model are you", 21.59),
        heard("I am the Brief Voice AI model.", 21.80, "lk_reply_1"),
        chat("user", "what are some things you can help me with", 32.34),
        heard("I can answer questions and give brief replies.", 32.44, "lk_reply_2"),
        chat("user", "can you tell me one interesting fact about space", 47.37),
        llm=Realtime("amazon.nova-2-5-sonic", "Amazon"), usage=usage)
    session._recorded_events = [
        rt_event("7316680d", at, ttft=ttft, duration=0.0, given=it + ia, produced=ot + oa, audio_in=ia, audio_out=oa)
        for at, it, ia, ot, oa, ttft in NOVA_PIECES]
    diagnostics = (await ended_body(http, session))["diagnostics"]

    assert "llm_requests" not in diagnostics
    assert diagnostics["usage"] == {
        "llm_input_tokens": 869, "llm_cached_input_tokens": 0, "llm_output_tokens": 307,
        "llm_input_audio_tokens": 465, "llm_input_text_tokens": 404, "llm_output_audio_tokens": 196,
        "llm_output_text_tokens": 111}


async def test_a_text_llms_usage_has_no_speech_split(http):
    usage = AgentSessionUsage(model_usage=[LLMModelUsage(
        provider="openai", model="gpt-4.1-mini", input_tokens=500, input_text_tokens=500, output_tokens=50)])
    diagnostics = (await ended_body(http, FakeSession(usage=usage)))["diagnostics"]

    assert diagnostics["usage"] == {"llm_input_tokens": 500, "llm_cached_input_tokens": 0, "llm_output_tokens": 50}


async def test_xais_session_close_record_is_no_request(http):
    # xAI closes with a record of the session's length, which it bills by
    session = FakeSession(chat("user", "Hi", 0.0), heard("Hello, Acme support.", 1.5, "resp_1"), llm=Realtime())
    session._recorded_events = [rt_event("session_close", T0 + 9.0, ttft=-1, session_duration=95.5)]

    assert "llm_requests" not in (await ended_body(http, session))["diagnostics"]


async def test_no_recorded_requests_sends_no_llm_requests(http):
    body = await ended_body(http, FakeSession(chat("user", "Hi", 0.0), chat("assistant", "Hello", 1.0)))

    assert "llm_requests" not in (body.get("diagnostics") or {})


def error_event(kind, at, message, recoverable, label, source=None):
    """A recorded ErrorEvent, as the session emits one for a failing STT, LLM or TTS (source: the model that failed)."""
    error = SimpleNamespace(type=kind, timestamp=T0 + at, label=label, error=RuntimeError(message),
                            recoverable=recoverable)
    return SimpleNamespace(type="error", created_at=T0 + at, error=error, source=source)


def plugin(module, **attributes):
    """A component whose class lives in a provider plugin's module, as livekit.plugins.<vendor> ones do."""
    return type("Plugin", (), {"__module__": module, **attributes})()


async def test_errors_name_the_model_that_failed_and_its_vendor(http):
    session = FakeSession(chat("user", "Hi", 0.0), chat("assistant", "Hello", 1.0))
    session._recorded_events = [
        # LiveKit Inference names the vendor in the model; its own provider is "livekit"
        error_event("llm_error", 2.0, "timed out", True, "inference.LLM",
                    SimpleNamespace(model="openai/gpt-4.1-mini", provider="livekit")),
        error_event("stt_error", 3.0, "closed", True, "deepgram.STT", plugin("livekit.plugins.deepgram.stt",
                                                                              model="nova-3")),
        error_event("llm_error", 4.0, "closed", True, "xai.Realtime", plugin("livekit.plugins.xai.realtime",
                                                                              model="grok-voice-3.0")),
        error_event("llm_error", 5.0, "closed", True, "aws.Realtime", plugin("livekit.plugins.aws.realtime",
                                                                              model="amazon.nova-2-5-sonic")),
        error_event("tts_error", 6.0, "closed", False, "custom.TTS", SimpleNamespace(model="unknown")),
    ]
    errors = (await ended_body(http, session))["diagnostics"]["errors"]

    assert [(e.get("source_model"), e.get("source_provider")) for e in errors] == [
        ("openai/gpt-4.1-mini", "openai"), ("nova-3", "deepgram"), ("grok-voice-3.0", "xai"),
        ("amazon.nova-2-5-sonic", "aws"), (None, None)]


async def test_pipeline_failures_are_sent_as_errors(http):
    session = FakeSession(chat("user", "Hi", 0.0), chat("assistant", "Hello", 1.0))
    leaked = "connection closed: wss://api.deepgram.com/v1/listen?token=abcdefghijklmnopqrstuvwxyz0123456789"
    session._recorded_events = [
        error_event("stt_error", 2.0, leaked, True, "deepgram.STT"),  # retried, and it came back
        error_event("llm_error", 3.0, "rate limited", True, "openai.LLM"),  # retried, then failed for good
        error_event("llm_error", 4.0, "rate limited", False, "openai.LLM"),
        error_event("tts_error", 5.0, "voice not found", False, "cartesia.TTS"),
        error_event("interruption_detection_error", 6.0, "model missing", True, "detector"),  # no such step
    ]
    body = await ended_body(http, session)

    assert body["diagnostics"]["errors"] == [
        {"step": "stt", "side": "caller", "start_ms": 2000, "message": "deepgram.STT: connection closed: <url>",
         "recovered": True},
        {"step": "llm", "side": "agent", "start_ms": 3000, "message": "openai.LLM: rate limited", "recovered": False},
        {"step": "llm", "side": "agent", "start_ms": 4000, "message": "openai.LLM: rate limited", "recovered": False},
        {"step": "tts", "side": "agent", "start_ms": 5000, "message": "cartesia.TTS: voice not found",
         "recovered": False},
    ]


async def test_no_failure_is_sent_as_an_empty_error_list_but_only_from_an_event_record(http):
    session = FakeSession(chat("user", "Hi", 0.0), chat("assistant", "Hello", 1.0))
    session._recorded_events = []  # LiveKit records every session event: none was a failure
    assert (await ended_body(http, session))["diagnostics"]["errors"] == []

    session = FakeSession(chat("user", "Hi", 0.0), chat("assistant", "Hello", 1.0))  # an SDK without an event record
    assert "errors" not in ((await ended_body(http, session)).get("diagnostics") or {})  # can't say there were none


async def test_a_retried_failure_recovered_when_its_step_worked_again_before_failing_for_good(http):
    session = FakeSession(chat("user", "Hi", 0.0), chat("assistant", "Hello", 1.0))
    worked = SimpleNamespace(type="metrics_collected", metrics=SimpleNamespace(
        type="llm_metrics", timestamp=T0 + 3.0, duration=0.4, ttft=0.2, speech_id=None))
    session._recorded_events = [
        error_event("llm_error", 2.0, "rate limited", True, "openai.LLM"),  # retried, and the LLM worked at 3 s
        worked,
        error_event("llm_error", 6.0, "rate limited", False, "openai.LLM"),  # later, a failure for good
    ]
    body = await ended_body(http, session)

    assert body["diagnostics"]["errors"] == [
        {"step": "llm", "side": "agent", "start_ms": 2000, "end_ms": 3000, "message": "openai.LLM: rate limited",
         "recovered": True},
        {"step": "llm", "side": "agent", "start_ms": 6000, "message": "openai.LLM: rate limited", "recovered": False},
    ]


async def test_live_interruptions_are_what_livekit_resolved():  # real AgentSession + AgentActivity, not started
    stt = SimpleNamespace(capabilities=SimpleNamespace(streaming=True, aligned_transcript=False))
    session = AgentSession(stt=stt, vad=None, turn_handling={"turn_detection": "stt"})
    assert call_report._live_interruptions(AgentActivity(Agent(instructions="x"), session)) == {
        "enabled": True, "min_duration_ms": 500, "min_words": 0}
    off = Agent(instructions="x", allow_interruptions=False)  # the agent's own setting wins over the session's
    assert call_report._live_interruptions(AgentActivity(off, session))["enabled"] is False
    own = Agent(instructions="x", turn_handling={"interruption": {"min_duration": 1.2, "min_words": 2}})
    assert call_report._live_interruptions(AgentActivity(own, session)) == {
        "enabled": True, "min_duration_ms": 1200, "min_words": 2}
    assert call_report._live_interruptions(None) is None


async def test_livekit_inference_stt_stand_in_confidence_is_left_out(http):
    # LiveKit Inference STT says 1.0 when its gateway gives no confidence
    line = chat("user", "Hi", 0.0, id="item_u1", transcript_confidence=1.0)
    inference = FakeSession(line, chat("assistant", "Hello", 1.0), stt=SimpleNamespace(provider="livekit",
                                                                                     model="deepgram/nova-3"))
    direct = FakeSession(line, chat("assistant", "Hello", 1.0), stt=SimpleNamespace(provider="deepgram",
                                                                                  model="nova-3"))

    assert "confidence" not in (await ended_body(http, inference))["transcript"][0]
    assert (await ended_body(http, direct))["transcript"][0]["confidence"] == 1.0


async def test_a_nova_sonic_failure_on_the_monotonic_clock_keeps_its_time(http, monkeypatch):
    # Nova Sonic stamps its errors on the monotonic clock: 3 s here is T0 + 3, as the clocks read T0 + 100 and 100
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: T0 + 100.0, monotonic=lambda: 100.0))
    session = FakeSession(chat("user", "Hi", 0.0), chat("assistant", "Hello", 1.0), llm=Realtime())
    error = SimpleNamespace(type="realtime_model_error", timestamp=3.0, label="aws.realtime.RealtimeModel",
                            error=RuntimeError("stream closed"), recoverable=True)
    session._recorded_events = [SimpleNamespace(type="error", created_at=T0 + 3.4, error=error)]
    body = await ended_body(http, session)

    assert body["diagnostics"]["errors"] == [
        {"step": "llm", "side": "agent", "start_ms": 3000, "message": "aws.realtime.RealtimeModel: stream closed",
         "recovered": True},
    ]


async def test_tools_of_every_active_agent_are_sent_once(http):
    first = SimpleNamespace(tools=[
        tool("lookup_order", "Look up an order by its number."),
        llm.Toolset(id="kb", tools=[raw_tool("search_kb", "Search the help centre.")]),
        "not a tool",
    ])
    second = SimpleNamespace(tools=[tool("lookup_order", "A later description."), raw_tool("transfer")])
    third = SimpleNamespace(tools=[tool("hang_up", "Hang up.")])
    session = FakeSession(agent=first, tools=[tool("end_call", "End the call.")])
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    assert await recognised(ctx) is True  # call.started saw the first agent

    # every conversation item and agent state change samples the agent running then
    session.agent = SimpleNamespace(tools=[tool("on_a_message", "Seen when a message was added.")])
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=chat("user", "Hi", 1)))
    session.agent = second
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=AgentHandoff(new_agent_id="x")))
    session.agent = third
    session.emit("agent_state_changed", AgentStateChangedEvent(old_state="listening", new_state="thinking"))
    third.tools.append(tool("goodbye", "Say goodbye."))  # seen at shutdown
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert ended.body["diagnostics"]["configuration"]["tools"] == [
        {"name": "end_call", "description": "End the call."},
        {"name": "lookup_order", "description": "Look up an order by its number."},
        {"name": "search_kb", "description": "Search the help centre."},
        {"name": "on_a_message", "description": "Seen when a message was added."},
        {"name": "transfer"},
        {"name": "hang_up", "description": "Hang up."},
        {"name": "goodbye", "description": "Say goodbye."},
    ]


async def test_mcp_tools_of_the_running_agent_are_sent(http):
    # LiveKit sets an MCP toolset up only once the agent runs, keeps mcp_servers= tools on the running
    # AgentActivity alone, and empties both when the session closes, before the job's shutdown
    session = FakeSession(agent=SimpleNamespace(tools=[llm.Toolset(id="mcp_toolset_1", tools=[])]))
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    assert await recognised(ctx) is True  # nothing set up yet

    mcp_server_tools = llm.Toolset(id="mcp_toolset_2", tools=[raw_tool("search_kb", "Search the help centre.")])
    session._activity = SimpleNamespace(tools=[mcp_server_tools])
    session.emit("agent_state_changed", AgentStateChangedEvent(old_state="initializing", new_state="listening"))
    session._activity = None
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert ended.body["diagnostics"]["configuration"]["tools"] == [
        {"name": "search_kb", "description": "Search the help centre."}]


async def test_a_failing_sample_warns_once_and_never_reaches_the_agent(http, caplog):
    session = FakeSession(agent=SimpleNamespace(tools=[tool("end_call", "End the call.")]))
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    assert await recognised(ctx) is True
    session._activity = SimpleNamespace(tools=17)  # a TypeError here: LiveKit's emitter re-raises those
    for _ in range(3):
        session.emit("agent_state_changed", AgentStateChangedEvent(old_state="listening", new_state="thinking"))
    session._activity = None
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert ended.body["diagnostics"]["configuration"]["tools"] == [{"name": "end_call", "description": "End the call."}]
    assert caplog.text.count("could not read tools or turn detection") == 1


@pytest.mark.parametrize(("entries", "expected"), [
    ([LLMModelUsage(provider="openai", model="gpt-4.1", input_tokens=500, input_cached_tokens=400,
                    output_tokens=60),
      LLMModelUsage(provider="openai", model="gpt-4.1-mini", input_tokens=300, input_cached_tokens=200,
                    output_tokens=40),
      STTModelUsage(provider="deepgram", model="nova-3", audio_duration=30.25),
      STTModelUsage(provider="deepgram", model="nova-2", audio_duration=17.75),
      TTSModelUsage(provider="cartesia", model="sonic-2", characters_count=900)],
     {"llm_input_tokens": 800, "llm_cached_input_tokens": 600, "llm_output_tokens": 100, "stt_audio_ms": 48000,
      "tts_characters": 900}),
    ([LLMModelUsage(provider="openai", model="gpt-realtime", input_tokens=40, output_tokens=5)],
     {"llm_input_tokens": 40, "llm_cached_input_tokens": 0, "llm_output_tokens": 5}),
], ids=["all-types", "llm-only"])
async def test_usage_totals(http, entries, expected):
    session = FakeSession(usage=AgentSessionUsage(model_usage=entries))
    diagnostics = (await ended_body(http, session))["diagnostics"]

    assert diagnostics["usage"] == expected


async def test_ended_reason_from_the_close_event(http):
    session = FakeSession()
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    await recognised(ctx)
    session.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED))
    await ctx.shutdown("room disconnected")

    [started] = http.posts("call.started")
    [ended] = http.posts("call.ended")
    assert ended.body["ended_reason"] == "user_initiated"
    assert "ended_reason" not in started.body


@pytest.mark.parametrize(("reason", "expected"), [("room deleted", "room deleted"), ("", None)],
                         ids=["shutdown-reason", "none"])
async def test_ended_reason_falls_back_to_the_shutdown_reason(http, reason, expected):
    body = await ended_body(http, FakeSession(), reason=reason)

    assert body.get("ended_reason") == expected


async def test_values_are_clamped_to_the_server_limits(http):
    tools = [tool("first", "d" * 3999 + "\U0001F600"), tool("n" * 250, "Long name.")]  # 4001 UTF-16 units
    tools += [tool(f"tool_{i}", f"Tool {i}.") for i in range(210)]
    long_call = FunctionCall(call_id="call_" + "x" * 250, name="m" * 250, arguments="{}", created_at=T0)
    session = FakeSession(long_call, FunctionCallOutput(call_id=long_call.call_id, output="ok", is_error=False),
                          agent=SimpleNamespace(tools=tools), stt=SimpleNamespace(provider="p" * 300, model="nova-3"))
    body = await ended_body(http, session, reason="r" * 300)

    sent = body["diagnostics"]["configuration"]
    assert len(sent["tools"]) == 200
    assert sent["tools"][0] == {"name": "first", "description": "d" * 3999}  # the split emoji is dropped
    assert sent["tools"][1]["name"] == "n" * 200
    assert sent["tools"][-1]["name"] == "tool_197"
    assert sent["stt"] == {"provider": "p" * 200, "model": "nova-3"}
    assert body["ended_reason"] == "r" * 200
    [call] = body["tool_calls"]
    assert (call["id"], call["name"], call["result"]) == (("call_" + "x" * 250)[:200], "m" * 200, "ok")


async def test_a_failing_section_is_left_out_alone(http, caplog):
    session = FakeSession(
        chat("user", "Hi", 0, id="item_u1", metrics={"transcription_delay": 0.1, "end_of_turn_delay": 0.3}),
        chat("assistant", "Hello!", 1, id="item_a1", metrics={"e2e_latency": 0.8}),
        stt=BrokenAdapter(),
        llm=SimpleNamespace(provider="openai", model="gpt-4.1"),
        options=SimpleNamespace(turn_handling={"endpointing": ["not", "a", "dict"], "interruption": {"enabled": True}}),
    )
    body = await ended_body(http, session)

    assert [turn["text"] for turn in body["transcript"]] == ["Hi", "Hello!"]
    assert body["diagnostics"] == {
        "pipeline": "stt_llm_tts",
        "configuration": {"llm": {"provider": "openai", "model": "gpt-4.1"}, "interruptions": {"enabled": True}},
        "turn_latencies": [{"turn_id": "item_a1", "transcriber_ms": 100, "endpointing_ms": 200,
                            "turn_ms": 800}],
    }
    assert caplog.text.count("left out of call.ended") == 2  # one line each: stt and turn_detection


# --- delivery -----------------------------------------------------------------


async def test_retries_reuse_the_event_id(http, monkeypatch):
    monkeypatch.setattr(call_report, "_STARTED_TIMEOUT_S", 0.05)
    http.script["started"] = [ConnectionError("down"), (503, {"error": "unavailable"}), "hang"]
    http.script["ended"] = [(500, {}), LINKED]
    ctx = FakeCtx(FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"})))
    attach(ctx, session_with(), api_key="k", base_url=BASE)

    # the server never answered call.started, but a test ID is enough to still send call.ended
    assert await recognised(ctx) is True
    await ctx.shutdown()
    started = http.posts("call.started")
    ended = http.posts("call.ended")
    assert len(started) == 3 and len(ended) == 2
    assert all(c.body == started[0].body for c in started)
    assert all(c.body == ended[0].body for c in ended)
    assert started[0].body["event_id"] == "lk:RM_job1:started"
    assert ended[0].body["event_id"] == "lk:RM_job1:ended"


async def test_client_errors_are_not_retried(http):
    http.script["started"] = [(401, {"error": "invalid_api_key"})]
    ctx = FakeCtx(FakeRoom([sip_participant(**{"sip.h.x-superbryn-call-id": "sbc_1"})]))
    attach(ctx, session_with(), api_key="bad", base_url=BASE)

    await recognised(ctx)
    assert len(http.posts("call.started")) == 1


async def test_phone_test_whose_call_started_got_no_answer_still_sends_call_ended(http):
    # a caller-number match has no test ID; call.ended repeats the start fields, so it can link on its own
    http.script["started"] = [(503, {"error": "unavailable"})]
    ctx = FakeCtx(FakeRoom([sip_participant()]))
    attach(ctx, session_with(), api_key="k", base_url=BASE)

    assert await recognised(ctx) is True
    await ctx.shutdown()
    assert len(http.posts("call.started")) == 3
    [ended] = http.posts("call.ended")
    assert "superbryn_call_id" not in ended.body
    assert {"from", "to", "started_at", "agent_id"} <= ended.body.keys()


@pytest.mark.parametrize("answer", [(403, {"code": "insufficient_scope"}), (409, {"code": "agent_mismatch"})],
                         ids=["403", "409"])
async def test_refused_call_started_sends_no_call_ended(http, answer):
    http.script["started"] = [answer]
    ctx = FakeCtx(FakeRoom([sip_participant()]))
    attach(ctx, session_with(), api_key="k", base_url=BASE)

    assert await recognised(ctx) is False
    await ctx.shutdown()
    assert len(http.posts("call.started")) == 1
    assert http.posts("call.ended") == []


async def test_ended_at_is_when_the_session_closed_not_the_later_shutdown(http):
    session = FakeSession()
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    assert await recognised(ctx) is True
    report = ctx.shutdown_callbacks[0].__self__
    hung_up = report._started_ts + 60.0  # the job shuts down only once the room goes, later than this
    session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED, created_at=hung_up))
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert ended.body["ended_at"] == call_report._iso(hung_up)


async def closed_body(http, session, closed_at):
    """One test call whose session closed at closed_at; its call.ended body."""
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    await recognised(ctx)
    session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED, created_at=closed_at))
    await ctx.shutdown()
    return http.posts("call.ended")[-1].body


def last_caller_line(states, stopped_at, closed_at):
    """A call whose last caller line ("No.", from 10 s) LiveKit stopped at stopped_at, with these caller states."""
    session = FakeSession(
        heard("Anything else I can help with?", 1.0),
        chat("user", "No.", 10.0, metrics={"started_speaking_at": T0 + 10.0, "stopped_speaking_at": T0 + stopped_at}))
    session._recorded_events = states
    return session, T0 + closed_at


async def test_a_caller_line_cut_by_the_hang_up_ends_where_the_caller_stopped(http):
    # LiveKit ends a line the call cut off at the close: on staging a "No." ran 11.5 s for 0.7 s of speech
    session, closed = last_caller_line([caller("speaking", 10.0), caller("listening", 10.7)], 21.5, 21.6)
    line = (await closed_body(http, session, closed))["transcript"][1]

    assert line["end_ms"] - line["start_ms"] == 700


async def test_a_caller_still_talking_at_the_close_gets_no_end(http):
    session, closed = last_caller_line([caller("speaking", 10.0)], 21.5, 21.6)
    line = (await closed_body(http, session, closed))["transcript"][1]

    assert "start_ms" in line and "end_ms" not in line


async def test_a_last_caller_line_that_really_ran_to_the_close_keeps_its_end(http):
    # "No." then the caller hung up: the voice detector's stop is within 2 s of LiveKit's end, which stands
    session, closed = last_caller_line([caller("speaking", 10.0), caller("listening", 11.0)], 10.6, 11.5)
    line = (await closed_body(http, session, closed))["transcript"][1]

    assert line["end_ms"] - line["start_ms"] == 600


async def test_ended_at_is_never_before_the_start(http):
    session = FakeSession()
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED, created_at=1.0))
    await recognised(ctx)
    await ctx.shutdown()

    [started] = http.posts("call.started")
    [ended] = http.posts("call.ended")
    assert ended.body["ended_at"] == started.body["started_at"]


# --- the job's shutdown -------------------------------------------------------


async def test_shutdown_does_not_wait_for_a_caller(http, monkeypatch):
    monkeypatch.setattr(call_report, "_SIP_WAIT_S", 30.0)  # e.g. a console session, ended at once
    ctx = FakeCtx(FakeRoom([]))
    attach(ctx, session_with(ChatMessage(role="user", content=["hi"])), api_key="k", base_url=BASE)
    await asyncio.sleep(0.05)

    await asyncio.wait_for(ctx.shutdown(), 1)
    assert ctx.shutdown_callbacks[0].__self__._task.cancelled()
    assert http.calls == []


async def test_shutdown_drops_an_unfinished_caller_number_lookup(http, monkeypatch):
    monkeypatch.setattr(call_report, "_STARTED_TIMEOUT_S", 30.0)
    http.script["caller-numbers"] = ["hang"]
    ctx = FakeCtx(FakeRoom([sip_participant()]))  # no test ID yet
    attach(ctx, session_with(), api_key="k", base_url=BASE)
    await until(lambda: http.gets())

    await asyncio.wait_for(ctx.shutdown(), 1)
    assert [c for c in http.calls if c.method == "POST"] == []


@pytest.mark.parametrize("attributes", [{"sip.h.x-superbryn-call-id": "sbc_1"}, {}], ids=["test-id", "phone-test"])
async def test_call_started_in_flight_at_shutdown_sends_call_ended_alone(http, monkeypatch, attributes):
    monkeypatch.setattr(call_report, "_STARTED_TIMEOUT_S", 30.0)
    http.script["started"] = ["hang"]
    ctx = FakeCtx(FakeRoom([sip_participant(**attributes)]))
    attach(ctx, session_with(ChatMessage(role="user", content=["hi"])), api_key="k", base_url=BASE)
    await until(lambda: http.posts("call.started"))

    await asyncio.wait_for(ctx.shutdown(), 1)
    [started] = http.posts("call.started")
    [ended] = http.posts("call.ended")
    for field in ("call_id", "agent_id", "from", "to", "started_at"):
        assert ended.body[field] == started.body[field]
    assert ended.body.get("testing", {}).get("superbryn_call_id") == attributes.get("sip.h.x-superbryn-call-id")
    assert ended.body["transcript"][0]["text"] == "hi"


async def test_the_shutdown_work_has_one_deadline(http, monkeypatch, caplog):
    monkeypatch.setattr(call_report, "_SHUTDOWN_S", 0.2)
    http.script["ended"] = ["hang"]
    ctx = ctx_with_test_id()
    attach(ctx, session_with(), api_key="k", base_url=BASE)
    assert await recognised(ctx) is True

    await asyncio.wait_for(ctx.shutdown(), 1)
    assert len(http.posts("call.ended")) == 1
    assert "SUPERBRYN_CALL_REPORT_FAILED: call.ended for call RM_job1 not sent within 0.2 s of shutdown" in caplog.text


async def test_a_second_attach_in_the_same_job_is_ignored(http, caplog):
    ctx = ctx_with_test_id()
    attach(ctx, FakeSession(chat("user", "Hi", 0)), api_key="k", base_url=BASE)
    attach(ctx, FakeSession(chat("user", "A second session", 0)), api_key="k", base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown()

    assert len(ctx.shutdown_callbacks) == 1
    assert len(http.posts("call.started")) == 1
    [ended] = http.posts("call.ended")
    assert [turn["text"] for turn in ended.body["transcript"]] == ["Hi"]
    assert caplog.text.count("attach already ran in this job") == 1


# --- settings and the call path -----------------------------------------------


async def test_no_key_does_nothing_but_the_handle_still_works(http, caplog):
    caplog.set_level(logging.INFO)
    for name in ("call-room-1", "call-room-2"):
        ctx = FakeCtx(FakeRoom([sip_participant()], name=name), job_room_name=name)
        report = attach(ctx, session_with(), custom_data={"team_id": "t-1"})
        report.update_custom_data({"resolved": True})
        report.set_call_end_reason("done")
        assert ctx.shutdown_callbacks == []

    assert not [t for t in asyncio.all_tasks() if t.get_name() == "superbryn_call_report"]
    await asyncio.sleep(0.05)
    assert http.calls == []
    assert caplog.text.count("SUPERBRYN_CALL_REPORT_DISABLED") == 1


async def test_settings_come_from_env_and_webhook_url_is_not_read(http, monkeypatch):
    monkeypatch.setenv("SUPERBRYN_API_KEY", "env_key")
    monkeypatch.setenv("SUPERBRYN_TEST_API_KEY", "old_test_key")  # 0.3.x's eval-mode key: not read
    monkeypatch.setenv("WEBHOOK_URL", "https://old.test/webhooks/obs/livekit")  # 0.3.x's route: never posted to
    metadata = json.dumps({"superbryn_call_id": "sbc_1"})
    ctx = FakeCtx(FakeRoom([sip_participant()]), job_metadata=metadata)
    attach(ctx, session_with())
    await recognised(ctx)
    [started] = http.posts("call.started")
    assert started.url == "https://api.superbryn.com/public-api/v1/calls/agent-data"
    assert started.headers["X-API-Key"] == "env_key"
    assert started.body["agent_id"] == AGENT_UUID

    other_uuid = "11111111-2222-4333-8444-555555555555"
    monkeypatch.setenv("SUPERBRYN_BASE_URL", "https://env.test/")
    monkeypatch.setenv("SUPERBRYN_AGENT_ID", other_uuid)
    ctx2 = FakeCtx(FakeRoom([sip_participant()], name="call-room-2"), job_metadata=metadata,
                   job_room_name="call-room-2")
    attach(ctx2, session_with())
    await recognised(ctx2)
    started2 = http.posts("call.started")[1]
    assert started2.url == "https://env.test/public-api/v1/calls/agent-data"
    assert started2.body["agent_id"] == other_uuid
    assert all("old.test" not in c.url for c in http.calls)


@pytest.mark.parametrize("agent", [None, "my-agent", "4347a69f-0c1e-0b6a-9f1e-2b3c4d5e6f70"],
                         ids=["missing", "name", "version-0"])
async def test_an_agent_id_that_is_not_a_uuid_is_left_out(http, monkeypatch, caplog, agent):
    # the server refuses a whole report whose agent_id is not a UUID; an agent key names the agent itself
    caplog.set_level(logging.INFO)
    monkeypatch.delenv("SUPERBRYN_AGENT_ID")
    monkeypatch.setenv("AGENT_ID", "support-agent")  # 0.3.x's free-form name: never read
    metadata = json.dumps({"superbryn_call_id": "sbc_1"})
    for name in ("call-room-1", "call-room-2"):
        ctx = FakeCtx(FakeRoom([sip_participant()], name=name), job_metadata=metadata, job_room_name=name)
        attach(ctx, session_with(), api_key="k", agent_id=agent, base_url=BASE)
        await recognised(ctx)

    assert len(http.posts("call.started")) == 2
    assert all("agent_id" not in c.body for c in http.calls)
    assert caplog.text.count("is not a SuperBryn agent ID") == (1 if agent else 0)


async def test_attach_returns_its_handle_without_awaiting(http):
    assert not inspect.iscoroutinefunction(attach)
    ctx = FakeCtx(FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"})))

    report = attach(ctx, session_with(), api_key="k", base_url=BASE)
    assert isinstance(report, call_report.CallReport)
    assert http.calls == []  # nothing ran on the call path
    assert len(ctx.shutdown_callbacks) == 1

    await recognised(ctx)
    await ctx.shutdown()
    assert [c.body["event"] for c in http.calls] == ["call.started", "call.ended"]


# --- production calls (surface "observability") -------------------------------


async def production_call(http, participants, session=None, **options):
    """Run one call that is not a SuperBryn test through shutdown: (ctx, call.started, call.ended)."""
    ctx = FakeCtx(FakeRoom(participants))
    attach(ctx, session or FakeSession(chat("user", "Hi", 1.0)), api_key="k", base_url=BASE, **options)
    assert await recognised(ctx) is True
    await ctx.shutdown()
    [started], [ended] = http.posts("call.started"), http.posts("call.ended")
    return ctx, started.body, ended.body


async def test_a_call_that_is_not_a_test_is_a_production_call_with_a_stereo_recording(http, recorders):
    ctx, started, ended = await production_call(http, [sip_participant(phone="+12125550123")])

    assert started == {
        "schema_version": 1, "event_id": "lk:RM_job1:started", "event": "call.started", "surface": "observability",
        "source": "livekit", "call_id": "RM_job1", "agent_id": AGENT_UUID, "from": "+12125550123",
        "to": THEIR_NUMBER, "started_at": "2025-10-09T08:53:20.000Z", "observability": {"direction": "inbound"}}
    url = "https://recordings.test/call_recordings/call-room-1/20251009-085320/call.mp3"
    assert ended["observability"] == {"direction": "inbound", "recording_url": url, "stereo_recording_url": url,
                                      "recording_url_source": "superbryn_s3"}
    assert ended["surface"] == "observability" and "testing" not in ended
    [recorder] = recorders
    # with SuperBryn's credentials for this key, of the room by name; stopped when the call ends
    assert (recorder.credentials_url, recorder.api_key) == (BASE + "/api/recording-credentials", "k")
    assert recorder.rooms == ["call-room-1"] and recorder.stopped


async def test_record_false_reports_a_production_call_without_a_recording(http, recorders):
    ctx, started, ended = await production_call(http, [sip_participant(phone="+12125550123")], record=False)

    assert recorders == []
    assert ended["surface"] == "observability" and ended["observability"] == {"direction": "inbound"}
    assert started["observability"] == {"direction": "inbound"}


async def test_record_false_leaves_a_recording_set_by_the_agent(http, recorders):
    ctx = FakeCtx(FakeRoom([sip_participant(phone="+12125550123")]))
    report = attach(ctx, FakeSession(chat("user", "Hi", 1.0)), api_key="k", base_url=BASE, record=False)
    report.set_recording_url("https://example.test/their.mp3")
    assert await recognised(ctx) is True
    await ctx.shutdown()
    [ended] = http.posts("call.ended")
    assert recorders == []
    assert ended.body["observability"] == {"direction": "inbound", "recording_url": "https://example.test/their.mp3",
                                           "recording_url_source": "external"}


async def test_an_outbound_call_and_a_web_call(http, recorders):
    _, started, _ = await production_call(http, [sip_participant(phone="+12125550123", rule="")])
    assert (started["from"], started["to"], started["observability"]) == (
        THEIR_NUMBER, "+12125550123", {"direction": "outbound"})

    http.calls.clear()
    # a caller in a browser: no numbers and no direction, and no wait for a SIP participant
    ctx, started, ended = await production_call(http, [web_participant(joined=T0 + 2)])
    assert {"from", "to", "observability"}.isdisjoint(started)
    assert started["started_at"] == "2025-10-09T08:53:22.000Z"
    assert ended["context"]["participant_identity"] == "web-user"
    assert ended["observability"]["recording_url_source"] == "superbryn_s3"


async def test_a_test_call_is_never_recorded_and_a_denied_one_is_not_reported_again(http, recorders):
    http.script["started"] = [(200, {"surface": "testing", "status": "ignored", "is_test": False,
                                     "reason": "not_a_test"})]
    ctx = FakeCtx(FakeRoom([sip_participant()]))  # our caller number: a test call
    attach(ctx, FakeSession(chat("user", "Hi", 1.0)), api_key="k", base_url=BASE)

    assert await recognised(ctx) is False
    await ctx.shutdown()
    assert [c.body["surface"] for c in http.calls if c.method == "POST"] == ["testing"]  # reported once only
    assert recorders == []


async def test_your_own_recording_is_sent_instead_and_ours_is_not_started(http, recorders):
    own = "https://my-bucket.example/calls/room-1.mp3?X-Amz-Signature=abc"
    ctx = FakeCtx(FakeRoom([sip_participant(phone="+12125550123")]))
    report = attach(ctx, FakeSession(), api_key="k", base_url=BASE)
    report.set_recording_url(own)
    await recognised(ctx)
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert ended.body["observability"] == {"direction": "inbound", "recording_url": own,
                                           "recording_url_source": "external"}
    assert recorders == []


async def test_a_recording_that_fails_to_start_leaves_the_call_reported(http, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise ValueError("url is required")  # e.g. LIVEKIT_URL is not set

    monkeypatch.setattr(call_report, "RecordingManager", broken)
    _, _, ended = await production_call(http, [sip_participant(phone="+12125550123")])

    assert ended["observability"] == {"direction": "inbound"}
    assert "recording not started: url is required" in caplog.text


async def test_a_key_without_test_data_write_still_reports_production_calls(http, caplog):
    http.script["caller-numbers"] = [(403, {"code": "insufficient_scope"})]
    _, started, _ = await production_call(http, [sip_participant()])  # our number, but it can't be checked

    assert started["surface"] == "observability"
    assert caplog.text.count("caller numbers were refused (HTTP 403)") == 1
    assert "SUPERBRYN_CALL_REPORT_REJECTED" not in [r.message[:30] for r in caplog.records if r.levelname == "ERROR"]


# --- what every report carries -------------------------------------------------


async def test_custom_data_is_sent_as_given_and_changes_until_the_call_ends(http):
    ctx = FakeCtx(FakeRoom([sip_participant(phone="+12125550123")]))
    report = attach(ctx, FakeSession(), api_key="k", base_url=BASE,
                    custom_data={"enterprise_id": "ent_42", "team_id": 7, "agent_type_id": "sales_inbound"})
    await recognised(ctx)
    report.update_custom_data({"lead_score": 0.82, "tags": ["vip"], "booked": True, "note": None})
    await ctx.shutdown()

    [started] = http.posts("call.started")
    [ended] = http.posts("call.ended")
    assert "custom_data" not in started.body  # call.started carries identity only
    assert ended.body["custom_data"] == {"enterprise_id": "ent_42", "team_id": 7, "agent_type_id": "sales_inbound",
                                         "lead_score": 0.82, "tags": ["vip"], "booked": True, "note": None}

    http.calls.clear()
    ctx = FakeCtx(FakeRoom([sip_participant(phone="+12125550123")]))
    report = attach(ctx, FakeSession(), api_key="k", base_url=BASE, custom_data={"enterprise_id": "ent_42"})
    report.set_custom_data({"team_id": "t-1", "at": datetime(2026, 10, 28, tzinfo=timezone.utc)})  # replaces it
    await recognised(ctx)
    await ctx.shutdown()
    assert http.posts("call.ended")[0].body["custom_data"] == {"team_id": "t-1", "at": "2026-10-28 00:00:00+00:00"}


async def test_custom_data_strict_json_cannot_hold_is_left_out_alone(http, caplog):
    ctx = FakeCtx(FakeRoom([sip_participant(phone="+12125550123")]))
    attach(ctx, FakeSession(chat("user", "Hi", 1.0)), api_key="k", base_url=BASE, custom_data={"ratio": float("nan")})
    await recognised(ctx)
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert "custom_data" not in ended.body and ended.body["transcript"][0]["text"] == "Hi"
    assert "custom_data left out of call.ended" in caplog.text


@pytest.mark.parametrize(("close", "left", "expected"), [
    (CloseReason.PARTICIPANT_DISCONNECTED, rtc.DisconnectReason.CLIENT_INITIATED, "caller"),
    (CloseReason.PARTICIPANT_DISCONNECTED, rtc.DisconnectReason.ROOM_DELETED, "agent"),  # the agent hung up
    (CloseReason.USER_INITIATED, None, "agent"),  # session.shutdown(), e.g. LiveKit's EndCallTool
    (CloseReason.TASK_COMPLETED, None, "agent"),
    (CloseReason.ERROR, None, "system"),
    (CloseReason.JOB_SHUTDOWN, None, "system"),
    (CloseReason.JOB_SHUTDOWN, rtc.DisconnectReason.CLIENT_INITIATED, "caller"),  # the caller left first
    (None, None, None),  # the session never closed before the job's shutdown
    (CloseReason.JOB_SHUTDOWN, rtc.DisconnectReason.USER_UNAVAILABLE, "caller"),  # no answer: the agent's code shut down
    (CloseReason.PARTICIPANT_DISCONNECTED, rtc.DisconnectReason.USER_REJECTED, "caller"),
    (CloseReason.JOB_SHUTDOWN, rtc.DisconnectReason.SIP_TRUNK_FAILURE, "system"),
], ids=["caller-hung-up", "room-deleted", "user-initiated", "task-completed", "error", "job-shutdown",
        "caller-left-then-shutdown", "unknown", "no-answer", "rejected", "trunk-failure"])
async def test_ended_by(http, close, left, expected):
    participant = sip_participant(phone="+12125550123")
    session = FakeSession()
    ctx = FakeCtx(FakeRoom([participant]))
    attach(ctx, session, api_key="k", base_url=BASE)
    await recognised(ctx)
    participant.disconnect_reason = left
    if close is not None:
        session.emit("close", CloseEvent(reason=close))
    await ctx.shutdown()

    assert http.posts("call.ended")[0].body.get("ended_by") == expected


def outbound_call(monkeypatch, status="dialing"):
    """An outbound phone call LiveKit is still dialling, on a clock tests move: (now, caller, room, ctx)."""
    now = [T0 + 5.0]
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: now[0], monotonic=lambda: 100.0))
    caller = sip_participant(phone="+12125550123", rule="", **{"sip.callStatus": status})  # joined, i.e. dialled, at T0
    room = FakeRoom([caller])
    return now, caller, room, FakeCtx(room)


def iso_at(seconds):
    return datetime.fromtimestamp(T0 + seconds, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def test_an_outbound_call_counts_from_its_answer_and_rings_from_its_dialling(http, monkeypatch, recorders):
    now, caller, room, ctx = outbound_call(monkeypatch)
    attach(ctx, FakeSession(chat("user", "Hi", 23.0)), api_key="k", base_url=BASE)
    await until(lambda: room.handlers.get("participant_attributes_changed"))
    assert http.posts("call.started") == [] and recorders == []  # still ringing: no call yet, nothing recorded

    now[0] = T0 + 20.0
    caller.attributes["sip.callStatus"] = "active"
    room.emit("participant_attributes_changed", {"sip.callStatus": "active"}, caller)
    assert await recognised(ctx) is True
    await ctx.shutdown()

    [started], [ended] = http.posts("call.started"), http.posts("call.ended")
    assert started.body["started_at"] == iso_at(20.0)
    assert ended.body["context"]["ring_started_at"] == iso_at(0.0)
    assert ended.body["transcript"][0]["start_ms"] == 3000  # from the answer, not the dialling
    assert len(recorders) == 1 and ended.body["ended_reason"] == "room disconnected"  # an answered call's own reason
    assert not room.handlers["participant_attributes_changed"]  # stopped listening once answered


@pytest.mark.parametrize(("left", "reason", "by"), [
    (rtc.DisconnectReason.USER_UNAVAILABLE, "user_unavailable", "caller"),
    (rtc.DisconnectReason.USER_REJECTED, "user_rejected", "caller"),
    (rtc.DisconnectReason.SIP_TRUNK_FAILURE, "sip_trunk_failure", "system"),
], ids=["no-answer", "rejected", "trunk-failure"])
async def test_a_call_that_never_connected_says_why_and_is_not_recorded(http, monkeypatch, recorders, left, reason, by):
    now, caller, room, ctx = outbound_call(monkeypatch)
    attach(ctx, FakeSession(), api_key="k", base_url=BASE)
    await until(lambda: room.handlers.get("participant_attributes_changed"))
    now[0] = T0 + 30.0
    caller.disconnect_reason = left
    del room.remote_participants[caller.identity]
    assert await recognised(ctx) is True
    await ctx.shutdown()

    [started], [ended] = http.posts("call.started"), http.posts("call.ended")
    assert started.body["started_at"] == iso_at(0.0)  # it never connected: it counts from the dialling
    assert (ended.body["ended_reason"], ended.body["ended_by"]) == (reason, by)
    assert recorders == []


async def test_a_call_still_ringing_at_shutdown_is_reported_as_not_answered(http, monkeypatch, recorders):
    _, _, room, ctx = outbound_call(monkeypatch, status="ringing")
    attach(ctx, FakeSession(), api_key="k", base_url=BASE)
    await until(lambda: room.handlers.get("participant_attributes_changed"))
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert ended.body["ended_reason"] == "not_answered" and ended.body["started_at"] == iso_at(0.0)
    assert recorders == []


async def test_an_inbound_call_counts_from_when_it_joined_and_never_waits_for_the_answer(http, monkeypatch, recorders):
    # LiveKit shows an inbound call "ringing" until the agent's audio reaches it; that is not waited for
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: T0 + 5.0, monotonic=lambda: 100.0))
    caller = sip_participant(phone="+12125550123", **{"sip.callStatus": "ringing"})  # sip.ruleID set: inbound
    room = FakeRoom([caller])
    ctx = FakeCtx(room)
    attach(ctx, FakeSession(chat("user", "Hi", 1.0)), api_key="k", base_url=BASE)
    assert await recognised(ctx) is True
    await ctx.shutdown()

    [started], [ended] = http.posts("call.started"), http.posts("call.ended")
    assert started.body["started_at"] == iso_at(0.0) and len(recorders) == 1
    assert "participant_attributes_changed" not in room.handlers
    assert ended.body["ended_reason"] == "room disconnected"  # still "ringing" at the end: not "not_answered"


async def test_a_call_already_answered_when_attached_counts_from_then(http, monkeypatch):
    now, _, _, ctx = outbound_call(monkeypatch, status="active")  # e.g. dialled with wait_until_answered=True
    attach(ctx, FakeSession(), api_key="k", base_url=BASE)
    assert await recognised(ctx) is True
    assert http.posts("call.started")[0].body["started_at"] == iso_at(5.0)


async def test_your_end_reason_wins_and_a_failed_session_sends_its_error(http):
    session = FakeSession()
    ctx = FakeCtx(FakeRoom([sip_participant(phone="+12125550123")]))
    report = attach(ctx, session, api_key="k", base_url=BASE)
    await recognised(ctx)
    report.set_call_end_reason("  transfer_to_human ")
    error = llm.LLMError(timestamp=T0 + 5, label="openai.LLM",
                         error=RuntimeError("quota exceeded for key sk_" + "a" * 40), recoverable=False)
    session.emit("close", CloseEvent(reason=CloseReason.ERROR, error=error))
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert (ended.body["ended_reason"], ended.body["ended_by"]) == ("transfer_to_human", "system")
    assert ended.body["context"]["close"] == {"error_type": "llm_error",
                                              "error_message": "openai.LLM: quota exceeded for key ***"}


async def test_context_names_the_first_agent_s_prompt_the_room_and_the_caller(http, monkeypatch):
    pushed = []

    async def push_prompt(**kwargs):
        pushed.append(kwargs)
        return True

    monkeypatch.setattr(call_report, "push_prompt", push_prompt)
    session = FakeSession()  # attached before session.start(): no agent is running yet
    room = FakeRoom([sip_participant(phone="+12125550123")])
    room.creation_time = datetime.fromtimestamp(T0 - 2.4, timezone.utc)
    ctx = FakeCtx(room)
    attach(ctx, session, api_key="k", base_url=BASE, prompt="You are {{name}}.", prompt_version="v3")
    session.agent = Agent(instructions="You are Ava. Caller: Riley.")  # the first agent becomes active
    session.emit("agent_state_changed", AgentStateChangedEvent(old_state="initializing", new_state="listening"))
    await recognised(ctx)
    session.agent = Agent(instructions="You are Ben, in billing.")  # a handoff: its prompt is not sent
    session.emit("agent_state_changed", AgentStateChangedEvent(old_state="listening", new_state="speaking"))
    await ctx.shutdown()

    context = http.posts("call.ended")[0].body["context"]
    environment = context.pop("environment")
    assert context == {
        "system_prompt": "You are Ava. Caller: Riley.",
        "prompt_ref": {"hash": call_report.prompt_ref("You are {{name}}.")["hash"], "version": "v3"},
        "room_name": "call-room-1",
        "participant_identity": "sip_+12125550123",
        "ring_started_at": "2025-10-09T08:53:17.600Z",
    }
    assert environment["livekit_evals_version"] == "0.4.0"
    assert set(environment) == {"livekit_evals_version", "livekit_agents_version", "livekit_rtc_version",
                                "python_version"}
    assert pushed == [{"api_key": "k", "api_base_url": BASE, "template": "You are {{name}}.", "version": "v3",
                       "agent_id": AGENT_UUID}]  # an org key's push names its agent


async def test_the_system_prompt_is_the_first_agent_s_instructions_as_rendered(http):
    chat_context = pytest.importorskip("livekit.agents.llm.chat_context")
    if not hasattr(chat_context, "Instructions"):
        pytest.skip("a livekit-agents 1.8 feature")
    instructions = chat_context.Instructions("You are Ava.", audio="Keep it short.", text="Use markdown.")
    body = await ended_body(http, FakeSession(agent=Agent(instructions=instructions)))

    assert body["context"]["system_prompt"] == instructions.render(modality="audio")


def eou_event(at, waited, transcribed):
    """A recorded metrics_collected event for one caller turn the turn detector closed."""
    metrics = SimpleNamespace(type="eou_metrics", timestamp=T0 + at, end_of_utterance_delay=waited,
                              transcription_delay=transcribed, on_user_turn_completed_delay=0.002)
    return SimpleNamespace(type="metrics_collected", metrics=metrics)


async def test_end_of_turn_events_and_keypresses_are_sent_raw(http, monkeypatch):
    clock = SimpleNamespace(now=T0)
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now))
    session = FakeSession(chat("user", "Press one", 1.0))
    session._recorded_events = [eou_event(6.78, 0.58, 0.21), eou_event(9.0, 0.0, 0.0)]  # 0: no end of speech seen
    ctx = ctx_with_test_id()
    attach(ctx, session, api_key="k", base_url=BASE)
    await recognised(ctx)
    clock.now = T0 + 12.1
    ctx.room.emit("sip_dtmf_received", SimpleNamespace(code=1, digit="1"))
    clock.now = T0 + 13.0
    ctx.room.emit("sip_dtmf_received", SimpleNamespace(code=11, digit="#"))
    await ctx.shutdown()

    diagnostics = http.posts("call.ended")[0].body["diagnostics"]
    assert diagnostics["eou_events"] == [{"at_ms": 6780, "eou_delay_ms": 580, "transcription_delay_ms": 210},
                                         {"at_ms": 9000}]
    assert diagnostics["dtmf"] == [{"at_ms": 12100, "digit": "1"}, {"at_ms": 13000, "digit": "#"}]
    assert ctx.room.handlers["sip_dtmf_received"] == []  # let go of the room when the call ends


async def test_each_agent_line_names_its_provider_request(http):
    session = FakeSession(
        chat("assistant", "Hello!", 0.5, id="item_a1", metrics={"provider_request_ids": ["chatcmpl-1", "chatcmpl-2"]}),
        chat("user", "Hi", 1.0, id="item_u1", metrics={"provider_request_ids": ["not-for-callers"]}),
        chat("assistant", "Bye.", 2.0, id="item_a2", metrics={"provider_request_ids": []}),
    )
    transcript = (await ended_body(http, session))["transcript"]

    assert [turn.get("provider_request_id") for turn in transcript] == ["chatcmpl-1", None, None]


async def test_stt_language_is_sent_and_caller_line_language_and_handoffs_are_not(http):
    session = FakeSession(
        AgentHandoff(old_agent_id=None, new_agent_id="default_agent", created_at=T0),
        chat("user", "नमस्ते", 1.0, id="u1"),
        AgentHandoff(old_agent_id="default_agent", new_agent_id="billing_agent", created_at=T0 + 2),
        chat("assistant", "Billing here.", 3.0, id="a1"),
        stt=SimpleNamespace(model="deepgram/nova-3", provider="livekit", _opts=SimpleNamespace(language="multi")))
    session._recorded_events = [SimpleNamespace(type="user_input_transcribed", is_final=True, item_id="u1",
                                                language="hi")]
    body = await ended_body(http, session)

    assert body["transcript"][0] == {"id": "u1", "role": "user", "text": "नमस्ते", "start_ms": 1000}
    assert all("agent_id" not in turn for turn in body["transcript"])
    assert "agent_handoffs" not in body["diagnostics"]
    assert body["diagnostics"]["configuration"]["stt"]["language"] == "multi"


def test_a_realtime_model_is_named_not_its_helper():
    model = Realtime(model="openai/gpt-live-1", provider="livekit")
    usage = [SimpleNamespace(type="llm_usage", provider="livekit", model="gpt-5.6-luna", input_tokens=900,
                             output_tokens=100)]
    assert call_report._component(SimpleNamespace(llm=model), "llm", usage)["model"] == "openai/gpt-live-1"


async def test_the_answer_is_watched_from_when_the_caller_joined_even_while_the_lookup_is_slow(http, monkeypatch,
                                                                                            recorders):
    now, caller, room, ctx = outbound_call(monkeypatch)
    lookup_done = asyncio.Event()

    async def slow_lookup(*args):
        await lookup_done.wait()
        return False

    monkeypatch.setattr(call_report, "_is_caller_number", slow_lookup)
    attach(ctx, FakeSession(chat("user", "Hi", 23.0)), api_key="k", base_url=BASE)
    await until(lambda: room.handlers.get("participant_attributes_changed"))  # watching while the lookup still runs
    now[0] = T0 + 20.0
    caller.attributes["sip.callStatus"] = "active"
    room.emit("participant_attributes_changed", {"sip.callStatus": "active"}, caller)
    now[0] = T0 + 40.0
    lookup_done.set()
    assert await recognised(ctx) is True
    await ctx.shutdown()

    [started] = http.posts("call.started")
    assert started.body["started_at"] == iso_at(20.0)  # the answer, not when the lookup came back


def test_the_handle_never_raises_on_values_it_cannot_take(caplog):
    report = call_report.CallReport()
    report.set_custom_data({"team_id": "t-42"})
    report.set_custom_data(None)
    report.update_custom_data(["not", "a", "dict"])
    report.set_call_end_reason(5)
    report.set_recording_url(5)
    assert report._custom_data == {"team_id": "t-42"} and report._external_recording is None
    assert "update_custom_data takes a dict, got list" in caplog.text
    report.set_call_end_reason("x" * 300)
    assert len(report._end_reason) == 200


async def test_a_tool_call_without_a_name_is_left_out(http):
    session = session_with(
        FunctionCall(call_id="call_1", name="", arguments="{}", created_at=T0 + 1.0),
        FunctionCall(call_id="call_2", name="lookup_order", arguments="{}", created_at=T0 + 2.0),
    )
    ctx = FakeCtx(FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"})))
    attach(ctx, session, api_key="k", agent_id=AGENT_UUID, base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert [t["name"] for t in ended.body["tool_calls"]] == ["lookup_order"]


async def test_a_long_room_name_used_as_the_call_id_keeps_the_event_id_within_200(http):
    room = FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"}), sid="", plain_sid=True)
    ctx = FakeCtx(room, job_room_sid="", job_room_name="r" * 250)
    attach(ctx, session_with(), api_key="k", base_url=BASE)
    await recognised(ctx)

    [started] = http.posts("call.started")
    assert started.body["call_id"] == "r" * 189 and len(started.body["event_id"]) == 200


def test_the_refusal_hint_reads_the_servers_error_code():
    assert "SUPERBRYN_AGENT_ID" in call_report._refusal_hint(400, {"error": {"code": "agent_id_required"}})
    assert "not this key's agent" in call_report._refusal_hint(403, {"error": {"code": "agent_key_mismatch"}})
    assert "test_data:write" in call_report._refusal_hint(403, {"error": {"code": "insufficient_scope"}})


async def test_a_console_run_sends_nothing(http):
    ctx = FakeCtx(FakeRoom([]))
    ctx.is_fake_job = lambda: True
    attach(ctx, session_with(), api_key="k", base_url=BASE)
    assert ctx.shutdown_callbacks == [] and http.posts("call.started") == []


async def test_a_recording_that_gives_no_url_is_logged_under_the_package(http, monkeypatch, caplog):
    class NoUrl:
        def __init__(self, **kwargs):
            pass

        async def start_recording(self, room_name):
            return None

        async def stop_recording(self):
            pass

    monkeypatch.setattr(call_report, "RecordingManager", NoUrl)
    _, _, ended = await production_call(http, [sip_participant(phone="+12125550123")])

    assert "SUPERBRYN_CALL_REPORT_ERROR: recording not started for call RM_job1" in caplog.text
    assert "recording_url" not in ended["observability"]


def test_custom_data_takes_any_mapping_and_attach_never_raises_on_what_it_is_given(http, caplog):
    from collections import UserDict
    from types import MappingProxyType

    report = call_report.CallReport()
    report.set_custom_data(MappingProxyType({"team_id": "t-42"}))
    report.update_custom_data(UserDict({"enterprise_id": "e-1"}))
    assert report._custom_data == {"team_id": "t-42", "enterprise_id": "e-1"}

    class Ambiguous:  # like a numpy array: asking whether it is empty raises
        def __bool__(self):
            raise ValueError("truth value is ambiguous")

    handle = attach(FakeCtx(FakeRoom([])), session_with(), api_key="k", base_url=BASE, custom_data=Ambiguous())
    assert handle._custom_data == {} and "set_custom_data takes a dict, got Ambiguous" in caplog.text
    handle.set_call_end_reason(5)
    assert "set_call_end_reason takes text, got int" in caplog.text


async def test_an_answer_watch_that_fails_counts_the_call_from_when_the_caller_joined(http, monkeypatch, recorders, caplog):
    now, caller, room, ctx = outbound_call(monkeypatch)

    async def broken(*args):
        raise RuntimeError("no attributes")

    monkeypatch.setattr(call_report, "_wait_for_answer", broken)
    attach(ctx, FakeSession(chat("user", "Hi", 3.0)), api_key="k", base_url=BASE)
    assert await recognised(ctx) is True
    await ctx.shutdown()

    [started], [ended] = http.posts("call.started"), http.posts("call.ended")
    assert started.body["started_at"] == iso_at(0.0) and len(recorders) == 1
    assert ended.body["ended_reason"] == "room disconnected"
    assert "could not watch for the answer: no attributes" in caplog.text
