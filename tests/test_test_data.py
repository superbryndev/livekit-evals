"""Tests for eval mode (livekit_evals.test_data) with fake job, room, session and HTTP; no network."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import tempfile
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

from livekit_evals import test_data
from livekit_evals.test_data import attach_test_data

# Features only livekit-agents 1.8 has (expressive markup, the realtime fallback adapter).
needs_livekit_1_8 = pytest.mark.skipif(
    tuple(int(part) for part in LIVEKIT_AGENTS_VERSION.split(".")[:2]) < (1, 8), reason="a livekit-agents 1.8 feature")

OUR_NUMBER = "+14155550100"
THEIR_NUMBER = "+16505550199"
AGENT_UUID = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"
BASE = "https://sb.test"
AGENT_DATA_URL = BASE + "/public-api/v1/test-calls/agent-data"
LINKED = (200, {"status": "linked", "is_test": True, "link": {"status": "verified", "method": "test_id"}})
T0 = 1_760_000_000.0


class FakeHttp:
    """Replaces test_data._http_request: records requests, plays scripted answers per endpoint."""

    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []
        self.script = {
            "caller-numbers": [(200, {"numbers": [OUR_NUMBER, "+919876543210"]})],
            "started": [LINKED],
            "ended": [LINKED],
        }

    async def __call__(self, method, url, headers, body):
        # as strict as the real transport: NaN or Infinity in a body fails the test
        self.calls.append(SimpleNamespace(method=method, url=url, headers=dict(headers),
                                          body=json.loads(json.dumps(body, allow_nan=False)) if body else None))
        queue = self.script["caller-numbers" if url.endswith("/caller-numbers") else body["event"][5:]]
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


def sip_participant(phone=OUR_NUMBER, trunk=THEIR_NUMBER, rule="SDR_inbound", **attributes):
    """A SIP participant; a non-empty sip.ruleID means the call came in to their agent."""
    attrs = {"sip.phoneNumber": phone, "sip.trunkPhoneNumber": trunk, "sip.ruleID": rule, **attributes}
    return SimpleNamespace(identity=f"sip_{phone}", kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
                           attributes=attrs)


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

    def __init__(self, model="gpt-realtime"):
        super().__init__(capabilities=llm.RealtimeCapabilities(
            message_truncation=True, turn_detection=True, user_transcription=True,
            auto_tool_reply_generation=False, audio_output=True, manual_function_calls=True))
        self._model = model

    model = property(lambda self: self._model)
    provider = property(lambda self: "openai")

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
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown(reason)
    return http.posts("call.ended")[-1].body


async def recognised(ctx) -> bool:
    """Wait for the background recognition (and call.started) to settle; True for a test call."""
    call = ctx.shutdown_callbacks[0].__self__
    return await asyncio.wait_for(asyncio.shield(call.task), 2)


async def until(condition, timeout=2.0):
    """Wait until condition() holds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "timed out"
        await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def fast_and_clean(monkeypatch, tmp_path):
    for name in ("SUPERBRYN_TEST_API_KEY", "SUPERBRYN_API_KEY", "SUPERBRYN_AGENT_ID", "AGENT_ID", "SUPERBRYN_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    # The agent ID is required; tests about something else get it from the environment.
    monkeypatch.setenv("SUPERBRYN_AGENT_ID", AGENT_UUID)
    monkeypatch.setattr(test_data, "_no_agent_logged", False)
    monkeypatch.setattr(test_data, "_SIP_WAIT_S", 0.2)
    monkeypatch.setattr(test_data, "_POLL_S", 0.01)
    monkeypatch.setattr(test_data, "_BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(test_data, "_no_key_logged", False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the caller-number file of this test only
    test_data._caller_numbers_cache.clear()


@pytest.fixture
def http(monkeypatch):
    fake = FakeHttp()
    monkeypatch.setattr(test_data, "_http_request", fake)
    return fake


# --- recognising a test call -------------------------------------------------


@pytest.mark.parametrize("where", ["job", "room"])
async def test_metadata_test_id_sends_call_started(http, where):
    metadata = json.dumps({"superbryn_call_id": "sbc_meta"})
    # their agent dialled our number: no dispatch rule, so from = their trunk, to = us
    room = FakeRoom([sip_participant(rule="")], metadata=metadata if where == "room" else "")
    ctx = FakeCtx(room, job_metadata=metadata if where == "job" else "")
    attach_test_data(ctx, session_with(), api_key="key_1", agent_id=AGENT_UUID, base_url=BASE + "/")

    assert await recognised(ctx) is True
    assert http.gets() == []  # a test ID needs no caller-number prefilter
    [started] = http.posts("call.started")
    assert started.url == AGENT_DATA_URL
    assert started.headers == {"X-API-Key": "key_1", "Content-Type": "application/json"}
    body = started.body
    assert body["event_id"] == "lk:RM_job1:started"
    assert body["call_id"] == "RM_job1"
    assert body["source"] == "livekit"
    assert body["schema_version"] == 1
    assert body["superbryn_call_id"] == "sbc_meta"
    assert body["agent_id"] == AGENT_UUID
    assert (body["from"], body["to"]) == (THEIR_NUMBER, OUR_NUMBER)
    assert body["started_at"].endswith("Z")


@pytest.mark.parametrize("key", ["sip.h.x-superbryn-call-id", "sip.h.X-SuperBryn-Call-Id",
                                 "superbryn_call_id"])
async def test_sip_header_attribute_test_id(http, key):
    room = FakeRoom([sip_participant(**{key: " sbc_hdr "})])
    ctx = FakeCtx(room)
    attach_test_data(ctx, session_with(), api_key="key_1", base_url=BASE)

    assert await recognised(ctx) is True
    assert http.gets() == []
    [started] = http.posts("call.started")
    assert started.body["superbryn_call_id"] == "sbc_hdr"
    # we called their agent: from = us, to = their trunk number
    assert (started.body["from"], started.body["to"]) == (OUR_NUMBER, THEIR_NUMBER)
    assert started.body["agent_id"] == AGENT_UUID


async def test_candidate_confirmed_by_server_sends_both_events(http):
    ctx = FakeCtx(FakeRoom([sip_participant(phone="+1 (415) 555-0100")]))  # formatting differs from ours
    attach_test_data(ctx, session_with(ChatMessage(role="user", content=["hi"])), api_key="k", base_url=BASE)

    assert await recognised(ctx) is True
    [get] = http.gets()
    assert get.url == BASE + "/public-api/v1/test-calls/caller-numbers"
    assert get.headers["X-API-Key"] == "k"
    [started] = http.posts("call.started")
    assert "superbryn_call_id" not in started.body

    await ctx.shutdown()
    [ended] = http.posts("call.ended")
    assert ended.body["event_id"] == "lk:RM_job1:ended"
    assert ended.body["transcript"][0]["text"] == "hi"

    # the caller numbers are cached for the process
    ctx2 = FakeCtx(FakeRoom([sip_participant()], name="call-room-2"), job_room_name="call-room-2")
    attach_test_data(ctx2, session_with(), api_key="k", base_url=BASE)
    assert await recognised(ctx2) is True
    assert len(http.gets()) == 1


async def test_candidate_denied_by_server_stops(http):
    http.script["started"] = [(200, {"status": "ignored", "is_test": False, "reason": "not_a_test"})]
    ctx = FakeCtx(FakeRoom([sip_participant(phone="4155550100")]))  # last-10-digit match
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)

    assert await recognised(ctx) is False
    await ctx.shutdown()
    assert len(http.posts("call.started")) == 1
    assert http.posts("call.ended") == []


@pytest.mark.parametrize("participants", [[sip_participant(phone="+12125550123")], []],
                         ids=["other-caller", "no-sip-participant"])
async def test_non_candidate_sends_nothing(http, participants):
    ctx = FakeCtx(FakeRoom(participants))
    attach_test_data(ctx, session_with(ChatMessage(role="user", content=["hi"])), api_key="k", base_url=BASE)

    assert await recognised(ctx) is False
    await ctx.shutdown()
    assert [c for c in http.calls if c.method == "POST"] == []
    assert len(http.gets()) == (1 if participants else 0)


@pytest.mark.parametrize("plain_sid", [False, True], ids=["awaitable-sid", "plain-sid"])
async def test_room_sid_when_job_has_none(http, plain_sid):
    room = FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"}), plain_sid=plain_sid)
    ctx = FakeCtx(room, job_room_sid="")
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)

    await recognised(ctx)
    [started] = http.posts("call.started")
    assert (started.body["call_id"], started.body["event_id"]) == ("RM_room1", "lk:RM_room1:started")


def numbers_file(tmp_path):
    return tmp_path / os.path.basename(test_data._numbers_file(BASE))


def age_numbers_file(tmp_path, seconds):
    data = json.loads(numbers_file(tmp_path).read_text())
    numbers_file(tmp_path).write_text(json.dumps({**data, "fetched_at": data["fetched_at"] - seconds}))


async def test_caller_numbers_are_shared_with_later_jobs_on_the_host(http, tmp_path):
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert json.loads(numbers_file(tmp_path).read_text())["numbers"] == [OUR_NUMBER, "+919876543210"]
    test_data._caller_numbers_cache.clear()  # LiveKit runs the next job in a new process
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert len(http.gets()) == 1

    age_numbers_file(tmp_path, 3600)  # an hour old: fetched again
    test_data._caller_numbers_cache.clear()
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert len(http.gets()) == 2
    assert test_data._numbers_file("https://other.test") != test_data._numbers_file(BASE)


async def test_a_failed_caller_number_lookup_is_retried_after_a_minute_by_any_job(http, tmp_path):
    http.script["caller-numbers"] = [(401, {"error": "invalid_api_key"}), (200, {"numbers": [OUR_NUMBER]})]
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is False
    test_data._caller_numbers_cache.clear()
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is False
    assert len(http.gets()) == 1

    age_numbers_file(tmp_path, 60)
    test_data._caller_numbers_cache.clear()
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert len(http.gets()) == 2


async def test_the_caller_number_file_never_breaks_the_lookup(http, monkeypatch, tmp_path):
    numbers_file(tmp_path).write_text("{not json")  # unreadable: fetched again and replaced
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert json.loads(numbers_file(tmp_path).read_text())["numbers"] == [OUR_NUMBER, "+919876543210"]
    assert [p.name for p in tmp_path.iterdir()] == [numbers_file(tmp_path).name]  # no temp file left behind

    elsewhere = tmp_path / "elsewhere.json"  # a link (or a pipe or device, which could block) is never opened
    numbers_file(tmp_path).rename(elsewhere)
    numbers_file(tmp_path).symlink_to(elsewhere)
    assert test_data._read_numbers_file(BASE) is None

    def no_temp_dir():
        raise FileNotFoundError("No usable temporary directory")

    monkeypatch.setattr(tempfile, "gettempdir", no_temp_dir)  # nowhere to keep it: memory only
    test_data._caller_numbers_cache.clear()
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert await test_data._is_caller_number(OUR_NUMBER, BASE, "k") is True
    assert len(http.gets()) == 2


async def test_a_test_id_over_200_characters_is_ignored(http, caplog):
    metadata = json.dumps({"superbryn_call_id": "sbc_" + "y" * 250})
    ctx = FakeCtx(FakeRoom([sip_participant()]), job_metadata=metadata)
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)

    assert await recognised(ctx) is True  # the caller number still matches
    [started] = http.posts("call.started")
    assert "superbryn_call_id" not in started.body
    assert len(http.gets()) == 1
    assert caplog.text.count("superbryn_call_id over 200 characters ignored") == 1

    header = {"sip.h.x-superbryn-call-id": "sbc_hdr"}  # a valid test ID elsewhere is used instead
    ctx2 = FakeCtx(FakeRoom([sip_participant(**header)], name="call-room-2"), job_metadata=metadata,
                   job_room_name="call-room-2", job_room_sid="RM_job2")
    attach_test_data(ctx2, session_with(), api_key="k", base_url=BASE)
    assert await recognised(ctx2) is True
    assert http.posts("call.started")[1].body["superbryn_call_id"] == "sbc_hdr"


async def test_from_and_to_are_cut_to_200_characters(http):
    room = FakeRoom([sip_participant(phone="sip:" + "1" * 300, trunk="2" * 250)],
                    metadata=json.dumps({"superbryn_call_id": "sbc_1"}))
    ctx = FakeCtx(room)
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)
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
    attach_test_data(ctx, session, api_key="k", agent_id=AGENT_UUID, base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown()

    [started] = http.posts("call.started")
    [ended] = http.posts("call.ended")
    body = ended.body
    assert body["event_id"] == "lk:RM_job1:ended"
    assert body["ended_at"].endswith("Z")
    for field in ("call_id", "source", "schema_version", "superbryn_call_id", "agent_id", "from", "to",
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
    app.router.add_post("/public-api/v1/test-calls/agent-data", agent_data)
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
    attach_test_data(ctx, session, api_key="k", base_url=base)
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
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
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
        {"turn_id": "item_greet", "voiceLatency": 200},  # answers no user message
        # the last user message counts; endpointing leaves out the transcript wait (206.4 - 104.4 ms)
        {"turn_id": "item_a1", "transcriberLatency": 104, "endpointingLatency": 102, "modelLatency": 728,
         "voiceLatency": 164, "turnLatency": 1103},
        {"transcriberLatency": 300, "endpointingLatency": 0, "modelLatency": 500},  # no text, so no turn_id
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
        }),
    )
    diagnostics = (await ended_body(http, session))["diagnostics"]

    assert diagnostics["pipeline"] == "stt_llm_tts"
    assert diagnostics["configuration"] == {
        "stt": {"provider": "deepgram", "model": "nova-3"},
        "llm": {"provider": "openai", "model": "gpt-4.1"},
        "tts": {"provider": "cartesia", "model": "sonic-2", "voice": "voice-id"},
        "turn_detection": {"type": "turn-detector-v1-mini", "min_end_of_turn_ms": 300, "max_end_of_turn_ms": 2500},
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
                                min_interruption_duration=0.7, min_interruption_words=2),
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
        "turn_detection": {"min_end_of_turn_ms": 400, "max_end_of_turn_ms": 5000},
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
    assert test_data._live_turn_detection(AgentActivity(Agent(instructions="x"), realtime)) == {
        "type": "realtime_llm"}

    stt = SimpleNamespace(capabilities=SimpleNamespace(streaming=True, aligned_transcript=False))
    session = AgentSession(stt=stt, vad=None,
                           turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0.4}})
    assert test_data._live_turn_detection(AgentActivity(Agent(instructions="x"), session)) == {
        "type": "stt", "min_end_of_turn_ms": 400, "max_end_of_turn_ms": 3000}
    agent = Agent(instructions="x", turn_handling={"turn_detection": "manual",
                                                   "endpointing": {"min_delay": 1.2, "max_delay": 4.0}})
    assert test_data._live_turn_detection(AgentActivity(agent, session)) == {
        "type": "manual", "min_end_of_turn_ms": 1200, "max_end_of_turn_ms": 4000}

    assert test_data._live_turn_detection(None) is None
    assert test_data._live_turn_detection(SimpleNamespace(_turn_detection="vad")) is None  # another SDK shape


async def test_live_turn_detection_before_1_8():  # an AgentActivity without _rt_turn_detection_enabled
    # A turn detector the SDK dropped (no VAD) is reported as none, with the delays that still apply.
    older = SimpleNamespace(_turn_detection=None, min_endpointing_delay=0.3, max_endpointing_delay=2.5, llm=None)
    assert test_data._live_turn_detection(older) == {"min_end_of_turn_ms": 300, "max_end_of_turn_ms": 2500}
    older._turn_detection = "vad"
    assert test_data._live_turn_detection(older) == {"type": "vad", "min_end_of_turn_ms": 300, "max_end_of_turn_ms": 2500}
    older.llm = Realtime()  # server-side turn detection takes the turns
    assert test_data._live_turn_detection(older) == {"type": "realtime_llm"}


async def test_turn_detection_seen_during_the_call_wins_over_the_options(http):
    options = SimpleNamespace(turn_handling={"turn_detection": NamedDetector(),
                                             "endpointing": {"min_delay": 0.3, "max_delay": 2.5}})
    session = FakeSession(options=options)
    ctx = ctx_with_test_id()
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
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

    assert body["diagnostics"]["llm_requests"] == [
        {"start_ms": 500, "outcome": "tool_call", "input_tokens": 900, "cached_input_tokens": 600, "output_tokens": 20},
        {"start_ms": 1500, "outcome": "spoken", "input_tokens": 950, "cached_input_tokens": 0, "output_tokens": 12},
        {"start_ms": 3500, "outcome": "not_spoken"},
        {"start_ms": 4100, "outcome": "cut_off", "input_tokens": 980, "cached_input_tokens": 0, "output_tokens": 30},
        {"start_ms": 5500, "outcome": "stopped"},
    ]
    # one clock for the transcript and the requests
    assert [turn["start_ms"] for turn in body["transcript"]] == [0, 2000, 4700]


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
        {"start_ms": 0, "outcome": "spoken", "input_tokens": 500, "cached_input_tokens": 0, "output_tokens": 40},
        {"start_ms": 2600, "outcome": "cut_off", "input_tokens": 700, "cached_input_tokens": 300, "output_tokens": 55},
    ]


async def test_no_recorded_requests_sends_no_llm_requests(http):
    body = await ended_body(http, FakeSession(chat("user", "Hi", 0.0), chat("assistant", "Hello", 1.0)))

    assert "llm_requests" not in (body.get("diagnostics") or {})


def error_event(kind, at, message, recoverable, label):
    """A recorded ErrorEvent, as the session emits one for a failing STT, LLM or TTS."""
    error = SimpleNamespace(type=kind, timestamp=T0 + at, label=label, error=RuntimeError(message),
                            recoverable=recoverable)
    return SimpleNamespace(type="error", created_at=T0 + at, error=error)


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
        {"step": "llm", "start_ms": 3000, "message": "openai.LLM: rate limited", "recovered": False},
        {"step": "llm", "start_ms": 4000, "message": "openai.LLM: rate limited", "recovered": False},
        {"step": "tts", "start_ms": 5000, "message": "cartesia.TTS: voice not found", "recovered": False},
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
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
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
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
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
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
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
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
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
        "turn_latencies": [{"turn_id": "item_a1", "transcriberLatency": 100, "endpointingLatency": 200,
                            "turnLatency": 800}],
    }
    assert caplog.text.count("left out of call.ended") == 2  # one line each: stt and turn_detection


# --- delivery -----------------------------------------------------------------


async def test_retries_reuse_the_event_id(http, monkeypatch):
    monkeypatch.setattr(test_data, "_STARTED_TIMEOUT_S", 0.05)
    http.script["started"] = [ConnectionError("down"), (503, {"error": "unavailable"}), "hang"]
    http.script["ended"] = [(500, {}), LINKED]
    ctx = FakeCtx(FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"})))
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)

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
    attach_test_data(ctx, session_with(), api_key="bad", base_url=BASE)

    await recognised(ctx)
    assert len(http.posts("call.started")) == 1


async def test_phone_test_whose_call_started_got_no_answer_still_sends_call_ended(http):
    # a caller-number match has no test ID; call.ended repeats the start fields, so it can link on its own
    http.script["started"] = [(503, {"error": "unavailable"})]
    ctx = FakeCtx(FakeRoom([sip_participant()]))
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)

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
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)

    assert await recognised(ctx) is False
    await ctx.shutdown()
    assert len(http.posts("call.started")) == 1
    assert http.posts("call.ended") == []


async def test_ended_at_is_when_the_session_closed_not_the_later_shutdown(http):
    session = FakeSession()
    ctx = ctx_with_test_id()
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
    assert await recognised(ctx) is True
    call = ctx.shutdown_callbacks[0].__self__
    hung_up = call.started_ts + 60.0  # the job shuts down only once the room goes, later than this
    session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED, created_at=hung_up))
    await ctx.shutdown()

    [ended] = http.posts("call.ended")
    assert ended.body["ended_at"] == test_data._iso(hung_up)


async def test_ended_at_is_never_before_the_start(http):
    session = FakeSession()
    ctx = ctx_with_test_id()
    attach_test_data(ctx, session, api_key="k", base_url=BASE)
    session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED, created_at=1.0))
    await recognised(ctx)
    await ctx.shutdown()

    [started] = http.posts("call.started")
    [ended] = http.posts("call.ended")
    assert ended.body["ended_at"] == started.body["started_at"]


# --- the job's shutdown -------------------------------------------------------


async def test_shutdown_does_not_wait_for_a_sip_participant(http, monkeypatch):
    monkeypatch.setattr(test_data, "_SIP_WAIT_S", 30.0)  # e.g. a web or console session, ended at once
    ctx = FakeCtx(FakeRoom([]))
    attach_test_data(ctx, session_with(ChatMessage(role="user", content=["hi"])), api_key="k", base_url=BASE)
    await asyncio.sleep(0.05)

    await asyncio.wait_for(ctx.shutdown(), 1)
    assert ctx.shutdown_callbacks[0].__self__.task.cancelled()
    assert http.calls == []


async def test_shutdown_drops_an_unfinished_caller_number_lookup(http, monkeypatch):
    monkeypatch.setattr(test_data, "_STARTED_TIMEOUT_S", 30.0)
    http.script["caller-numbers"] = ["hang"]
    ctx = FakeCtx(FakeRoom([sip_participant()]))  # no test ID yet
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)
    await until(lambda: http.gets())

    await asyncio.wait_for(ctx.shutdown(), 1)
    assert [c for c in http.calls if c.method == "POST"] == []


@pytest.mark.parametrize("attributes", [{"sip.h.x-superbryn-call-id": "sbc_1"}, {}], ids=["test-id", "phone-test"])
async def test_call_started_in_flight_at_shutdown_sends_call_ended_alone(http, monkeypatch, attributes):
    monkeypatch.setattr(test_data, "_STARTED_TIMEOUT_S", 30.0)
    http.script["started"] = ["hang"]
    ctx = FakeCtx(FakeRoom([sip_participant(**attributes)]))
    attach_test_data(ctx, session_with(ChatMessage(role="user", content=["hi"])), api_key="k", base_url=BASE)
    await until(lambda: http.posts("call.started"))

    await asyncio.wait_for(ctx.shutdown(), 1)
    [started] = http.posts("call.started")
    [ended] = http.posts("call.ended")
    for field in ("call_id", "agent_id", "from", "to", "started_at"):
        assert ended.body[field] == started.body[field]
    assert ended.body.get("superbryn_call_id") == attributes.get("sip.h.x-superbryn-call-id")
    assert ended.body["transcript"][0]["text"] == "hi"


async def test_the_shutdown_work_has_one_deadline(http, monkeypatch, caplog):
    monkeypatch.setattr(test_data, "_SHUTDOWN_S", 0.2)
    http.script["ended"] = ["hang"]
    ctx = ctx_with_test_id()
    attach_test_data(ctx, session_with(), api_key="k", base_url=BASE)
    assert await recognised(ctx) is True

    await asyncio.wait_for(ctx.shutdown(), 1)
    assert len(http.posts("call.ended")) == 1
    assert "SUPERBRYN_TEST_DATA_FAILED: call.ended for call RM_job1 not sent within 0.2 s of shutdown" in caplog.text


async def test_a_second_attach_in_the_same_job_is_ignored(http, caplog):
    ctx = ctx_with_test_id()
    attach_test_data(ctx, FakeSession(chat("user", "Hi", 0)), api_key="k", base_url=BASE)
    attach_test_data(ctx, FakeSession(chat("user", "A second session", 0)), api_key="k", base_url=BASE)
    await recognised(ctx)
    await ctx.shutdown()

    assert len(ctx.shutdown_callbacks) == 1
    assert len(http.posts("call.started")) == 1
    [ended] = http.posts("call.ended")
    assert [turn["text"] for turn in ended.body["transcript"]] == ["Hi"]
    assert caplog.text.count("attach_test_data already ran in this job") == 1


# --- settings and the call path -----------------------------------------------


async def test_no_key_does_nothing(http, caplog):
    caplog.set_level(logging.INFO)
    for name in ("call-room-1", "call-room-2"):
        ctx = FakeCtx(FakeRoom([sip_participant()], name=name), job_room_name=name)
        assert attach_test_data(ctx, session_with()) is None
        assert ctx.shutdown_callbacks == []

    assert not [t for t in asyncio.all_tasks() if t.get_name() == "superbryn_test_data"]
    await asyncio.sleep(0.05)
    assert http.calls == []
    assert caplog.text.count("SUPERBRYN_TEST_DATA_DISABLED") == 1


async def test_settings_come_from_env(http, monkeypatch):
    monkeypatch.setenv("SUPERBRYN_API_KEY", "obs_key")
    metadata = json.dumps({"superbryn_call_id": "sbc_1"})
    ctx = FakeCtx(FakeRoom([sip_participant()]), job_metadata=metadata)
    attach_test_data(ctx, session_with())
    await recognised(ctx)
    [started] = http.posts("call.started")
    assert started.url == "https://api.superbryn.com/public-api/v1/test-calls/agent-data"
    assert started.headers["X-API-Key"] == "obs_key"
    assert started.body["agent_id"] == AGENT_UUID

    other_uuid = "11111111-2222-4333-8444-555555555555"
    monkeypatch.setenv("SUPERBRYN_TEST_API_KEY", "test_key")
    monkeypatch.setenv("SUPERBRYN_BASE_URL", "https://env.test/")
    monkeypatch.setenv("SUPERBRYN_AGENT_ID", other_uuid)
    ctx2 = FakeCtx(FakeRoom([sip_participant()], name="call-room-2"), job_metadata=metadata,
                   job_room_name="call-room-2")
    attach_test_data(ctx2, session_with())
    await recognised(ctx2)
    started2 = http.posts("call.started")[1]
    assert started2.url == "https://env.test/public-api/v1/test-calls/agent-data"
    assert started2.headers["X-API-Key"] == "test_key"
    assert started2.body["agent_id"] == other_uuid


@pytest.mark.parametrize("agent", [None, "my-agent", "4347a69f-0c1e-0b6a-9f1e-2b3c4d5e6f70"],
                         ids=["missing", "name", "version-0"])
async def test_no_valid_agent_id_sends_nothing(http, monkeypatch, caplog, agent):
    caplog.set_level(logging.INFO)
    monkeypatch.delenv("SUPERBRYN_AGENT_ID")
    monkeypatch.setenv("AGENT_ID", AGENT_UUID)  # Observability's variable: never read here
    metadata = json.dumps({"superbryn_call_id": "sbc_1"})
    for name in ("call-room-1", "call-room-2"):
        ctx = FakeCtx(FakeRoom([sip_participant()], name=name), job_metadata=metadata, job_room_name=name)
        attach_test_data(ctx, session_with(), api_key="k", agent_id=agent)
        assert ctx.shutdown_callbacks == []

    await asyncio.sleep(0.05)
    assert http.calls == []
    assert caplog.text.count("SUPERBRYN_TEST_DATA_DISABLED") == 1
    assert (f"{agent!r} is not a SuperBryn agent ID" if agent else "no agent ID, set SUPERBRYN_AGENT_ID") in caplog.text


async def test_attach_returns_without_awaiting(http):
    assert not inspect.iscoroutinefunction(attach_test_data)
    ctx = FakeCtx(FakeRoom([sip_participant()], metadata=json.dumps({"superbryn_call_id": "sbc_1"})))

    assert attach_test_data(ctx, session_with(), api_key="k", base_url=BASE) is None
    assert http.calls == []  # nothing ran on the call path
    assert len(ctx.shutdown_callbacks) == 1

    await recognised(ctx)
    await ctx.shutdown()
    assert [c.body["event"] for c in http.calls] == ["call.started", "call.ended"]
