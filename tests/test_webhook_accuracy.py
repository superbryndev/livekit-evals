"""Tests for the webhook's turn and usage accuracy found by end-to-end calls: usage reports aren't deduplicated away,
one caller turn stays one turn with its own times and confidence, reply waits come from the caller's real stop,
stand-in caller times are dropped, providers are named right, and handoffs say which agent spoke."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from livekit.agents import llm
from livekit.agents.llm import AgentHandoff, ChatMessage
from livekit.agents.metrics import STTMetrics

from livekit_evals import test_data
from test_webhook_realtime import T0, FakeSession, attach, realtime_response, reply
from test_webhook_realtime import replay_clock  # noqa: F401  (autouse: every handler here runs on the replay clock)


class Realtime(llm.RealtimeModel):
    """A realtime model by name, detecting turns itself."""

    def __init__(self, model, provider):
        super().__init__(capabilities=llm.RealtimeCapabilities(
            message_truncation=True, turn_detection=True, user_transcription=True,
            auto_tool_reply_generation=False, audio_output=True, manual_function_calls=True))
        self._model, self._provider = model, provider

    model = property(lambda self: self._model)
    provider = property(lambda self: self._provider)

    def session(self, *, turn_detection_disabled=False):
        raise NotImplementedError


def caller_line(session, at, text, *, finals=None, stop_after=1.0, confidence=None, language=None, stopped=None):
    """The caller speaks from `at`: user states, STT finals (default: the whole text), then LiveKit's message."""
    session.emit("user_state_changed", SimpleNamespace(old_state="listening", new_state="speaking"), at)
    for i, final in enumerate(finals or [text]):
        session.emit("user_input_transcribed", SimpleNamespace(transcript=final, is_final=True, speaker_id=None,
                                                               language=language), at + 0.3 + i * 0.2)
    session.emit("user_state_changed", SimpleNamespace(old_state="speaking", new_state="listening"), at + stop_after)
    metrics = {"started_speaking_at": at, "stopped_speaking_at": stopped if stopped is not None else at + stop_after - 0.5,
               "transcription_delay": 0.2}
    item = ChatMessage(role="user", content=[text], metrics=metrics, transcript_confidence=confidence)
    session.emit("conversation_item_added", SimpleNamespace(item=item), at + stop_after + 0.05)


def turns(handler, speaker=None):
    out = handler._build_webhook_payload()["call"]["transcript"]["turns"]
    return [t for t in out if speaker is None or t["speaker"] == speaker]


def stt_metrics(at, seconds):
    return STTMetrics(label="deepgram.STT", request_id="stream-1", timestamp=at, duration=0.0, audio_duration=seconds,
                      streamed=True)


async def test_every_streamed_stt_report_counts_once():
    session = FakeSession(llm=SimpleNamespace(model="gpt-4.1-mini"))
    handler = await attach(session)
    first = stt_metrics(T0 + 5, 5.0)
    for metrics, at in ((first, T0 + 5), (first, T0 + 5), (stt_metrics(T0 + 10, 4.5), T0 + 10)):
        session.emit("metrics_collected", SimpleNamespace(metrics=metrics), at)

    assert handler._build_webhook_payload()["call"]["usage"]["stt_duration_seconds"] == pytest.approx(9.5)


async def test_one_caller_turn_stays_one_turn_with_its_own_times_and_confidence():
    session = FakeSession(llm=SimpleNamespace(model="gpt-4.1-mini"))
    handler = await attach(session)
    caller_line(session, T0 + 2, "Can you check my order? The number is 4712.",
                finals=["Can you check my order?", "The number is 4712."], stop_after=3.0, confidence=0.97,
                language="en")
    reply(session, T0 + 6, "It shipped yesterday.", [])

    (turn,) = turns(handler, "user")
    assert turn["text"] == "Can you check my order? The number is 4712."
    assert turn["start_time_ms"] == 2000
    assert turn["end_time_ms"] == 4500  # the voice detector's stop, not user_state's 5000
    assert turn["confidence_score"] == pytest.approx(0.97)
    assert turn["language"] == "en"
    assert turn["turn_latency"]["stt_ms"] == pytest.approx(200)


async def test_reply_wait_comes_from_the_caller_turn_it_answers():
    session = FakeSession(llm=SimpleNamespace(model="gpt-4.1-mini"))
    handler = await attach(session)
    caller_line(session, T0 + 1, "Tell me a story.", stop_after=1.5)                 # stops speaking at 2.0 s
    reply(session, T0 + 3, "Once upon a time", [], interrupted=True)                 # cut off by the barge-in
    caller_line(session, T0 + 4, "Stop, just the hours.", stop_after=2.0)            # stops at 5.5 s
    reply(session, T0 + 7, "Nine to five.", [])
    reply(session, T0 + 10, "Anything else?", [])                                    # a second reply, no new caller

    assert [t["response_delay_ms"] for t in turns(handler, "assistant")] == [1000, 1500, None]


async def test_a_model_that_only_stands_in_for_caller_times_sends_none():
    session = FakeSession(llm=Realtime(model="google/gemini-3.8-live", provider="livekit"))
    handler = await attach(session)
    caller_line(session, T0 + 1, "Hello?", confidence=1.0)
    reply(session, T0 + 3, "Hi there.", ["resp_1"], [realtime_response("resp_1", 0.4)])

    (user,) = turns(handler, "user")
    assert (user["start_time_ms"], user["end_time_ms"]) == (None, None)
    assert user["confidence_score"] is None  # a realtime model's 1.0 is a stand-in
    (agent,) = turns(handler, "assistant")
    assert (agent["response_delay_ms"], agent["turn_latency"]) == (None, None)
    call = handler._build_webhook_payload()["call"]
    assert (call["latency"]["llm_ms"], call["latency"]["total_ms"]) == (None, None)
    stats = call["speech_stats"]
    assert [stats[k] for k in ("user_talk_seconds", "agent_talk_ratio", "silence_seconds", "longest_silence_ms",
                               "avg_response_delay_ms", "max_response_delay_ms")] == [None] * 6
    assert stats["agent_turn_count"] == 1


async def test_a_model_with_real_timings_keeps_its_reply_latency():
    session = FakeSession(llm=Realtime(model="gpt-realtime", provider="openai"))
    handler = await attach(session)
    caller_line(session, T0 + 1, "Hello?")
    reply(session, T0 + 3, "Hi there.", ["resp_1"], [realtime_response("resp_1", 0.4)])

    (agent,) = turns(handler, "assistant")
    assert agent["turn_latency"]["llm_ms"] == pytest.approx(400)
    assert agent["response_delay_ms"] == 1500
    call = handler._build_webhook_payload()["call"]
    assert call["latency"]["llm_ms"] == pytest.approx(400)
    assert call["speech_stats"]["avg_response_delay_ms"] == 1500


async def test_a_cascaded_reply_carries_its_own_llm_and_tts_times():
    session = FakeSession(llm=SimpleNamespace(model="gpt-4.1-mini"))
    handler = await attach(session)
    for at, ttft, ttfb in ((T0 + 1, 0.9, 0.2), (T0 + 5, 0.4, 0.1)):
        session.emit("agent_state_changed", SimpleNamespace(old_state="listening", new_state="speaking"), at)
        session.emit("agent_state_changed", SimpleNamespace(old_state="speaking", new_state="listening"), at + 2)
        item = ChatMessage(role="assistant", content=["ok"], metrics={"llm_node_ttft": ttft, "tts_node_ttfb": ttfb})
        session.emit("conversation_item_added", SimpleNamespace(item=item), at + 2)

    assert [(round(t["turn_latency"]["llm_ms"]), round(t["turn_latency"]["tts_ms"]))
            for t in turns(handler, "assistant")] == [(900, 200), (400, 100)]


@pytest.mark.parametrize("model, provider", [
    ("amazon.nova-2-5-sonic", "aws"), ("xai/grok-voice-3.0", "xai"), ("openai/gpt-4.1-mini", "openai"),
    ("deepgram/nova-3", "deepgram"), ("nova-3", "deepgram"), ("sonic-3", "cartesia"), ("models/gemini-3.8-live", "google"),
])
async def test_providers_are_named_from_the_model(model, provider):
    handler = await attach(FakeSession(llm=SimpleNamespace(model="gpt-4.1-mini")))
    assert handler._detect_provider_from_model_name(model) == provider


async def test_a_realtime_call_names_its_voice_and_an_error_its_vendor():
    model = Realtime(model="openai/gpt-realtime-2.1", provider="livekit")
    model._opts = SimpleNamespace(voice="marin")
    session = FakeSession(llm=model)
    handler = await attach(session)
    error = SimpleNamespace(type="llm_error", recoverable=True, label="livekit.agents.inference.llm.LLM",
                            error=TimeoutError("Request timed out."))
    session.emit("error", SimpleNamespace(error=error, source=SimpleNamespace(model="openai/gpt-4.1-mini")), T0 + 1)

    call = handler._build_webhook_payload()["call"]
    assert call["metadata"]["tts_voice_id"] == "marin"
    assert call["errors"][0]["source_provider"] == "openai"


async def test_a_handoff_lists_both_agents_and_tags_each_turn():
    first = SimpleNamespace(id="default_agent", instructions="You are the receptionist.")
    second = SimpleNamespace(id="billing_agent", instructions="You are billing.")
    session = FakeSession(llm=SimpleNamespace(model="gpt-4.1-mini"), current_agent=first)
    handler = await attach(session)
    reply(session, T0 + 1, "Transferring you.", [])
    session.current_agent = second
    handoff = AgentHandoff(old_agent_id="default_agent", new_agent_id="billing_agent", created_at=T0 + 3)
    session.emit("conversation_item_added", SimpleNamespace(item=handoff), T0 + 3)
    reply(session, T0 + 4, "Hi, this is billing.", [])

    call = handler._build_webhook_payload()["call"]
    assert call["metadata"]["system_prompt"] == "You are the receptionist."
    assert call["metadata"]["agents"] == [{"agent_id": "default_agent", "instructions": "You are the receptionist."},
                                          {"agent_id": "billing_agent", "instructions": "You are billing."}]
    assert [t["agent_id"] for t in call["transcript"]["turns"]] == ["default_agent", "billing_agent"]


def test_eval_mode_sends_caller_language_handoffs_and_stt_language():
    events = [SimpleNamespace(type="user_input_transcribed", is_final=True, item_id="u1", language="hi")]
    items = [AgentHandoff(old_agent_id=None, new_agent_id="default_agent", created_at=T0),
             ChatMessage(id="u1", role="user", content=["नमस्ते"], metrics={"started_speaking_at": T0 + 1}),
             AgentHandoff(old_agent_id="default_agent", new_agent_id="billing_agent", created_at=T0 + 2)]

    transcript, _ = test_data._history_payload(items, T0, languages=test_data._caller_languages(events))
    assert transcript[0]["language"] == "hi" and transcript[0]["text"] == "नमस्ते"
    assert test_data._agent_handoffs(items, T0) == [
        {"old_agent_id": "default_agent", "new_agent_id": "billing_agent", "at_ms": 2000}]

    stt = SimpleNamespace(model="deepgram/nova-3", provider="livekit", _opts=SimpleNamespace(language="multi"))
    assert test_data._component(SimpleNamespace(stt=stt), "stt", [])["language"] == "multi"


def test_eval_mode_names_a_realtime_model_not_its_helper():
    model = Realtime(model="openai/gpt-live-1", provider="livekit")
    usage = [SimpleNamespace(type="llm_usage", provider="livekit", model="gpt-5.6-luna", input_tokens=900,
                             output_tokens=100)]
    assert test_data._component(SimpleNamespace(llm=model), "llm", usage)["model"] == "openai/gpt-live-1"



async def test_late_transcripts_livekit_splits_off_stay_in_the_caller_turn():
    session = FakeSession(llm=SimpleNamespace(model="gemini-3.5-flash"))
    handler = await attach(session)
    session.emit("user_state_changed", SimpleNamespace(old_state="listening", new_state="speaking"), T0 + 1)
    session.emit("user_state_changed", SimpleNamespace(old_state="speaking", new_state="listening"), T0 + 4)
    # the STT's finals land after the caller stopped; LiveKit makes each its own message, only the first timed
    for i, (text, timed) in enumerate((("Thanks.", True), ("That's all.", False), ("Goodbye.", False))):
        at = T0 + 6 + i
        session.emit("user_input_transcribed", SimpleNamespace(transcript=text, is_final=True, speaker_id=None,
                                                               language=None), at)
        metrics = {"started_speaking_at": T0 + 1, "stopped_speaking_at": T0 + 3.4} if timed else {}
        session.emit("conversation_item_added",
                     SimpleNamespace(item=ChatMessage(role="user", content=[text], metrics=metrics)), at + 0.05)
    reply(session, T0 + 10, "You're welcome!", [])

    (turn,) = turns(handler, "user")
    assert (turn["text"], turn["start_time_ms"], turn["end_time_ms"]) == ("Thanks. That's all. Goodbye.", 1000, 3400)
    assert turns(handler, "assistant")[0]["response_delay_ms"] == 6600


async def test_a_transcript_that_lands_after_the_reply_started_stays_in_its_caller_turn():
    session = FakeSession(llm=Realtime(model="openai/gpt-realtime-2.1", provider="livekit"))
    handler = await attach(session)
    session.emit("user_state_changed", SimpleNamespace(old_state="listening", new_state="speaking"), T0 + 1)
    session.emit("user_state_changed", SimpleNamespace(old_state="speaking", new_state="listening"), T0 + 3)
    session.emit("agent_state_changed", SimpleNamespace(old_state="listening", new_state="speaking"), T0 + 4)
    # the model's transcript of the caller comes after its reply began
    session.emit("user_input_transcribed", SimpleNamespace(transcript="Who is this?", is_final=True, speaker_id=None,
                                                           language=None), T0 + 4.3)
    session.emit("conversation_item_added", SimpleNamespace(item=ChatMessage(
        role="user", content=["Who is this?"], metrics={"started_speaking_at": T0 + 1})), T0 + 4.3)
    session.emit("agent_state_changed", SimpleNamespace(old_state="speaking", new_state="listening"), T0 + 6)
    session.emit("conversation_item_added", SimpleNamespace(item=ChatMessage(role="assistant", content=["Ava."])),
                 T0 + 6)

    (user,) = turns(handler, "user")
    assert (user["start_time_ms"], user["end_time_ms"]) == (1000, 3000)
    assert turns(handler, "assistant")[0]["response_delay_ms"] == 1000
