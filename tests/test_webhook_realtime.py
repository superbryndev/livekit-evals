"""Tests for the webhook's realtime-only fields (per-reply ttft, provider request ID, speech counts) and that a
cascaded session's payload is unchanged; fake room and session, replayed on one clock; no network."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from livekit.agents import llm
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import LLMMetrics, RealtimeModelMetrics, TTSMetrics
from livekit.agents.metrics.base import Metadata

from livekit_evals import webhook_handler as wh

T0 = 1_760_000_000.0
NEW_TURN_KEYS = {"provider_request_id"}
NEW_STATS_KEYS = {"speech_created_count", "cancelled_responses"}


class Clock:
    now_s = T0


class ClockDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(Clock.now_s, tz or timezone.utc)


@pytest.fixture(autouse=True)
def replay_clock(monkeypatch):
    Clock.now_s = T0
    monkeypatch.setattr(wh, "datetime", ClockDateTime)


class FakeRoom:
    name = "call-room-1"
    remote_participants = {}
    creation_time = datetime.fromtimestamp(T0, timezone.utc)

    def on(self, event, handler=None):
        return handler


class Realtime(llm.RealtimeModel):
    def __init__(self):
        super().__init__(capabilities=llm.RealtimeCapabilities(
            message_truncation=True, turn_detection=True, user_transcription=True,
            auto_tool_reply_generation=False, audio_output=True, manual_function_calls=True))

    model = property(lambda self: "gpt-realtime")
    provider = property(lambda self: "openai")

    def session(self, *, turn_detection_disabled=False):
        raise NotImplementedError


class FakeSession:
    """AgentSession stand-in: the components a test sets and session.on(event)(handler) registration."""

    def __init__(self, **components):
        self.handlers = {}
        self.current_agent = SimpleNamespace(instructions="You are a test agent.")
        self.__dict__.update(components)

    def on(self, event):
        def register(handler):
            self.handlers.setdefault(event, []).append(handler)
            return handler
        return register

    def emit(self, event, arg, at):
        Clock.now_s = at
        for handler in self.handlers.get(event, []):
            handler(arg)


async def attach(session, **options):
    handler = wh.WebhookHandler(webhook_url="https://sb.test/webhook", api_key="key", room=FakeRoom(),
                                is_deployed_on_lk_cloud=False, disable_recording=True, sync_prompt=False, **options)
    await handler.attach_to_session(session)
    return handler


def realtime_response(request_id, ttft, cancelled=False, model=None):
    return RealtimeModelMetrics(label="openai.RealtimeModel", request_id=request_id, timestamp=T0, ttft=ttft,
                                cancelled=cancelled, input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                                metadata=Metadata(model_name=model) if model else None)


def reply(session, at, text, request_ids, metrics=(), interrupted=False):
    """The agent speaks from `at` for 2 s; each of `metrics` arrives just before it starts."""
    for m in metrics:
        session.emit("metrics_collected", SimpleNamespace(metrics=m), at - 0.1)
    session.emit("speech_created", SimpleNamespace(source="generate_reply", user_initiated=False), at - 0.5)
    session.emit("agent_state_changed", SimpleNamespace(old_state="listening", new_state="speaking"), at)
    session.emit("agent_state_changed", SimpleNamespace(old_state="speaking", new_state="listening"), at + 2)
    item = ChatMessage(role="assistant", content=[text], interrupted=interrupted,
                       metrics={"started_speaking_at": at, "stopped_speaking_at": at + 2,
                                "provider_request_ids": list(request_ids)})
    session.emit("conversation_item_added", SimpleNamespace(item=item), at + 2)


def caller(session, at, text):
    session.emit("user_state_changed", SimpleNamespace(old_state="listening", new_state="speaking"), at)
    session.emit("user_state_changed", SimpleNamespace(old_state="speaking", new_state="listening"), at + 1)
    session.emit("conversation_item_added", SimpleNamespace(item=ChatMessage(role="user", content=[text])), at + 1)


def agent_turns(handler):
    return [t for t in handler._build_webhook_payload()["call"]["transcript"]["turns"] if t["speaker"] == "assistant"]


async def test_realtime_reply_gets_its_own_ttft_not_the_running_average():
    session = FakeSession(llm=Realtime())
    handler = await attach(session)
    reply(session, T0 + 1, "Hello!", ["resp_1"], [realtime_response("resp_1", 1.8)])
    caller(session, T0 + 4, "Who is this?")
    reply(session, T0 + 6, "Ava from Acme.", ["resp_2"], [realtime_response("resp_2", 0.9)])

    turns = agent_turns(handler)
    assert [t["provider_request_id"] for t in turns] == ["resp_1", "resp_2"]
    assert [t["turn_latency"]["llm_ms"] for t in turns] == [pytest.approx(1800), pytest.approx(900)]
    assert turns[1]["turn_latency"]["total_ms"] == pytest.approx(900)


async def test_an_id_two_responses_share_gives_no_ttft():
    session = FakeSession(llm=Realtime())  # Nova Sonic and GPT Live keep one ID for the session
    handler = await attach(session)
    reply(session, T0 + 1, "Hello!", ["sess_1"], [realtime_response("sess_1", 0.002)])
    caller(session, T0 + 4, "Who is this?")
    reply(session, T0 + 6, "Ava from Acme.", ["sess_1"], [realtime_response("sess_1", 0.003)])

    assert [t["turn_latency"]["llm_ms"] for t in agent_turns(handler)] == [0, 0]


async def test_one_response_delivered_twice_keeps_its_ttft():
    session = FakeSession(llm=Realtime())
    handler = await attach(session)
    response = realtime_response("resp_1", 1.2)
    reply(session, T0 + 1, "Hello!", ["resp_1"], [response, response])

    assert agent_turns(handler)[0]["turn_latency"]["llm_ms"] == pytest.approx(1200)


async def test_opening_record_and_unmeasured_ttft_send_zero():
    session = FakeSession(llm=Realtime())
    handler = await attach(session)
    session.emit("metrics_collected", SimpleNamespace(metrics=realtime_response("", -1)), T0)
    reply(session, T0 + 1, "Hello!", ["resp_1"], [realtime_response("resp_1", -1)])
    caller(session, T0 + 4, "Who is this?")
    reply(session, T0 + 6, "Ava from Acme.", [], [realtime_response("resp_2", 0.8)])

    turns = agent_turns(handler)
    assert [t["turn_latency"]["llm_ms"] for t in turns] == [0, 0]
    assert "provider_request_id" not in turns[1]


async def test_speech_stats_count_replies_started_and_responses_cancelled():
    session = FakeSession(llm=Realtime())
    handler = await attach(session)
    session.emit("speech_created", SimpleNamespace(source="say", user_initiated=True), T0)
    reply(session, T0 + 1, "Hello!", ["resp_1"], [realtime_response("resp_1", 1.0)])
    caller(session, T0 + 4, "Wait, stop")
    reply(session, T0 + 6, "Sure, let me", ["resp_2"], [realtime_response("resp_2", 0.7, cancelled=True)],
          interrupted=True)

    stats = handler._build_webhook_payload()["call"]["speech_stats"]
    assert stats["speech_created_count"] == 3
    assert stats["cancelled_responses"] == 1


async def test_realtime_found_from_its_metrics_when_attach_cannot_see_the_model():
    session = FakeSession()  # the model sits on an Agent that hasn't started yet
    handler = await attach(session)
    reply(session, T0 + 1, "Hello!", ["resp_1"], [realtime_response("resp_1", 1.1, model="gpt-realtime")])

    turn = agent_turns(handler)[0]
    assert turn["provider_request_id"] == "resp_1"
    assert turn["turn_latency"]["llm_ms"] == pytest.approx(1100)  # the metrics name a model with real timings


@pytest.mark.parametrize("extended", [True, False])
async def test_cascaded_session_gets_no_realtime_fields(extended):
    session = FakeSession(llm=SimpleNamespace(model="gpt-4.1-mini"), stt=SimpleNamespace(model="nova-3"),
                          tts=SimpleNamespace(model="sonic-2"))
    handler = await attach(session, extended_capture=extended)

    def llm_metrics(request_id, ttft):
        return LLMMetrics(label="openai.LLM", request_id=request_id, timestamp=T0, duration=ttft + 0.3, ttft=ttft,
                          cancelled=False, completion_tokens=10, prompt_tokens=300, prompt_cached_tokens=0,
                          total_tokens=310, tokens_per_second=30.0)

    def tts_metrics(request_id):
        return TTSMetrics(label="cartesia.TTS", request_id=request_id, timestamp=T0, ttfb=0.2, duration=1.0,
                          audio_duration=2.0, cancelled=False, characters_count=20, streamed=True)

    reply(session, T0 + 1, "Hello!", ["chatcmpl-1"], [llm_metrics("req_1", 0.4), tts_metrics("tts_1")])
    caller(session, T0 + 4, "Who is this?")
    reply(session, T0 + 6, "Ava from Acme.", ["chatcmpl-2"], [llm_metrics("req_2", 0.6), tts_metrics("tts_2")])

    call = handler._build_webhook_payload()["call"]
    turns = [t for t in call["transcript"]["turns"] if t["speaker"] == "assistant"]
    assert not NEW_TURN_KEYS & {key for t in turns for key in t}
    # replies without per-reply message metrics (older LiveKit) keep the running average (400, then (400 + 600) / 2);
    # without extended capture these session-level metrics aren't subscribed, so 0
    expected = [pytest.approx(400), pytest.approx(500)] if extended else [0, 0]
    assert [t["turn_latency"]["llm_ms"] for t in turns] == expected
    if extended:
        assert not NEW_STATS_KEYS & set(call["speech_stats"])
    else:
        assert "speech_stats" not in call
