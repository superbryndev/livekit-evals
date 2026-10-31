"""Whole calls through attach(): a cascaded and a realtime call, each as a SuperBryn test call and as a production
call, scripted the way LiveKit emits them, on one fake clock, with no network. Set LIVEKIT_EVALS_DUMP_DIR to also
write each body to <dir>/<surface>_<pipeline>_<event>.json."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from livekit import rtc
from livekit.agents import (AgentStateChangedEvent, CloseEvent, CloseReason, ConversationItemAddedEvent,
                            FunctionToolsExecutedEvent, MetricsCollectedEvent, UserInputTranscribedEvent,
                            UserStateChangedEvent, llm)
from livekit.agents.llm import AgentHandoff, ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.metrics import (AgentSessionUsage, EOUMetrics, LLMMetrics, LLMModelUsage, RealtimeModelMetrics,
                                    STTMetrics, STTModelUsage, TTSMetrics, TTSModelUsage)
from livekit.agents.metrics.base import Metadata
from livekit.protocol import egress as egress_proto

from livekit_evals import call_report
from livekit_evals.call_report import attach
from livekit_evals.recording_manager import RecordingManager

T0 = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc).timestamp()
AGENT_UUID = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"
TEST_CALL_ID = "sbc_9b1d7c520f3e4a8b6d25e7f8a9c0d1e"
CALLER = "+14155550123"
TRUNK = "+16505550199"
SUPERBRYN_NUMBER = "+18005550100"
PROMPT = "You are {{agent_name}}, a support agent for Acme."
# Every top-level key the server's call report reads; anything else would be dropped.
TOP_LEVEL = {"schema_version", "event_id", "event", "surface", "source", "call_id", "agent_id", "from", "to",
             "started_at", "ended_at", "ended_reason", "ended_by", "transcript", "tool_calls", "diagnostics",
             "context", "custom_data", "testing", "observability"}
# 0.3.x's package-side numbers and dropped fields: none may come back anywhere in a body.
REMOVED = {"speech_stats", "latency", "turn_detection_stats", "vad", "network", "rtc", "sip", "metadata",
           "agent_handoffs", "agents", "speaker_id", "response_delay_ms", "tokens_per_second", "total_tokens",
           "call_rate_usd", "livekit_project_id", "recording_url_reachable", "sip_trunking_enabled",
           "egress_enabled", "lk_agent_enabled", "transcriberLatency", "endpointingLatency", "modelLatency",
           "voiceLatency", "turnLatency", "turn_latency", "duration_seconds", "call_end_reason"}


class Clock:
    now = T0


class Room:
    """rtc.Room stand-in: the caller, room events, its creation time."""

    def __init__(self, name, sid, phone):
        self.name, self.metadata, self._sid = name, "", sid
        self.creation_time = datetime.fromtimestamp(T0 - 2.4, timezone.utc)  # ringing began
        self.caller = SimpleNamespace(
            identity=f"sip_{phone}", kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP, disconnect_reason=None,
            joined_at=datetime.fromtimestamp(T0, timezone.utc),
            attributes={"sip.callID": "SCL_dummy7Hk2", "sip.phoneNumber": phone, "sip.trunkPhoneNumber": TRUNK,
                        "sip.ruleID": "SDR_dummyRule"})
        self.remote_participants = {self.caller.identity: self.caller}
        self.handlers: dict[str, list] = {}

    @property
    def sid(self):
        async def sid():
            return self._sid
        return sid()

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def off(self, event, callback):
        self.handlers[event].remove(callback)

    def fire(self, event, at, *args):
        Clock.now = at
        for callback in list(self.handlers.get(event, [])):
            callback(*args)


class Ctx:
    """JobContext stand-in; shutdown() runs the shutdown callbacks as the worker does."""

    def __init__(self, room, test_call):
        self.room = room
        self.job = SimpleNamespace(metadata=json.dumps({"superbryn_call_id": TEST_CALL_ID}) if test_call else "",
                                   room=SimpleNamespace(name=room.name, sid=room._sid, metadata=""))
        self.callbacks = []

    def add_shutdown_callback(self, callback):
        self.callbacks.append(callback)

    async def shutdown(self, reason):
        await asyncio.gather(*(callback(reason) for callback in self.callbacks))


class Session:
    """AgentSession stand-in: records every emitted event (as AgentSession._recorded_events does) and adds
    conversation items to the history; no agent runs until start()."""

    def __init__(self, **components):
        self.history = ChatContext([])
        self._recorded_events = []
        self._agent = None
        self._using_default_vad = False
        self.handlers: dict[str, list] = {}
        self.__dict__.update(components)

    @property
    def current_agent(self):
        if self._agent is None:
            raise RuntimeError("VoiceAgent isn't running")
        return self._agent

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event, arg, at):
        Clock.now = at
        self._recorded_events.append(arg)
        if event == "conversation_item_added":
            self.history.items.append(arg.item)
        for callback in list(self.handlers.get(event, [])):
            callback(arg)

    def start(self, agent, at):
        self._agent = agent
        self.emit("agent_state_changed", AgentStateChangedEvent(old_state="initializing", new_state="listening",
                                                                created_at=at), at)


def plugin(module, **attributes):
    """A component whose class lives in a provider plugin's module, as real plugins do."""
    component = type(module.split(".")[-1].upper(), (), {"__module__": module})()
    component.__dict__.update(attributes)
    return component


def tool(name, description):
    async def run() -> str:
        return "ok"
    return llm.function_tool(run, name=name, description=description)


class OpenAIRealtime(llm.RealtimeModel):
    """Stands in for livekit.plugins.openai's RealtimeModel: server VAD, real timings."""

    def __init__(self):
        super().__init__(capabilities=llm.RealtimeCapabilities(
            message_truncation=True, turn_detection=True, user_transcription=True,
            auto_tool_reply_generation=False, audio_output=True, manual_function_calls=True))
        self._opts = SimpleNamespace(model="gpt-realtime", voice="marin")

    model = property(lambda self: "gpt-realtime")
    provider = property(lambda self: "openai")

    def session(self, *, turn_detection_disabled=False):
        raise NotImplementedError


OpenAIRealtime.__module__ = "livekit.plugins.openai.realtime.realtime_model"


def speech(session, at, speech_id, items=()):
    event = SimpleNamespace(type="speech_created", source="generate_reply", user_initiated=False, created_at=at,
                            speech_handle=SimpleNamespace(id=speech_id, chat_items=list(items)))
    session.emit("speech_created", event, at)
    return event


def state(session, who, old, new, at):
    kind = AgentStateChangedEvent if who == "agent" else UserStateChangedEvent
    session.emit(f"{who}_state_changed", kind(old_state=old, new_state=new, created_at=at), at)


def metric(session, metrics, at):
    session.emit("metrics_collected", MetricsCollectedEvent(metrics=metrics, created_at=at), at)


def say(session, at, end, item_id, text, *, interrupted=False, **metrics):
    """The agent speaks [at, end]; LiveKit adds the message when playback ends."""
    state(session, "agent", "thinking", "speaking", at)
    state(session, "agent", "speaking", "listening", end)
    item = ChatMessage(id=item_id, role="assistant", content=[text], interrupted=interrupted, created_at=at - 0.3,
                       metrics={"started_speaking_at": at, "stopped_speaking_at": end, **metrics})
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=item, created_at=end), end)
    return item


def hear(session, start, stop, listening_at, item_id, text, *, confidence=None, language="en", **metrics):
    """The caller speaks [start, stop]: the STT final, then LiveKit's message for the whole turn."""
    state(session, "user", "listening", "speaking", start)
    session.emit("user_input_transcribed", UserInputTranscribedEvent(
        transcript=text, is_final=True, item_id=item_id, language=language, created_at=start + 0.6), start + 0.6)
    state(session, "user", "speaking", "listening", listening_at)
    item = ChatMessage(id=item_id, role="user", content=[text], transcript_confidence=confidence,
                       created_at=listening_at, metrics={"started_speaking_at": start, "stopped_speaking_at": stop,
                                                         **metrics})
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=item, created_at=listening_at + 0.02),
                 listening_at + 0.02)


def run_tool(session, call_at, out_at, call_id, name, args, output):
    call = FunctionCall(call_id=call_id, name=name, arguments=json.dumps(args), created_at=call_at)
    result = FunctionCallOutput(call_id=call_id, name=name, output=output, is_error=False, created_at=out_at)
    session.history.items.extend([call, result])
    session.emit("function_tools_executed", FunctionToolsExecutedEvent(
        function_calls=[call], function_call_outputs=[result], created_at=out_at), out_at)
    return call


def llm_metrics(request_id, at, ttft, duration, prompt, cached, completion, speech_id):
    return LLMMetrics(label="livekit.plugins.openai.llm.LLM", request_id=request_id, timestamp=at, duration=duration,
                      ttft=ttft, cancelled=False, completion_tokens=completion, prompt_tokens=prompt,
                      prompt_cached_tokens=cached, total_tokens=prompt + completion, tokens_per_second=30.0,
                      speech_id=speech_id)


def tts_metrics(request_id, at, ttfb, characters, speech_id, cancelled=False):
    return TTSMetrics(label="livekit.plugins.cartesia.tts.TTS", request_id=request_id, timestamp=at, ttfb=ttfb,
                      duration=1.4, audio_duration=2.6, cancelled=cancelled, characters_count=characters,
                      streamed=True, segment_id=f"seg_{request_id}", speech_id=speech_id)


def eou(session, at, waited, transcribed, speech_id):
    metric(session, EOUMetrics(timestamp=at, end_of_utterance_delay=waited, transcription_delay=transcribed,
                               on_user_turn_completed_delay=0.002, speech_id=speech_id), at)


async def cascade_call(session, room):
    """Greeting; a lookup tool and its answer; an LLM timeout that recovered; a reply the caller cut off; a handoff to
    billing by tool; a keypress; the caller hangs up."""
    receptionist = SimpleNamespace(id="receptionist", instructions="You are Ava, a support agent for Acme.",
                                   tools=[tool("lookup_order", "Look up an order by its number."),
                                          tool("transfer_to_billing", "Hand the caller to the billing agent.")])
    billing = SimpleNamespace(id="billing", instructions="You are Ben, Acme's billing agent.",
                              tools=[tool("lookup_invoice", "Look up an invoice.")])
    session.start(receptionist, T0 + 0.2)

    speech(session, T0 + 0.5, "speech_greet")
    metric(session, tts_metrics("tts_greet", T0 + 1.2, 0.21, 52, "speech_greet"), T0 + 1.2)
    say(session, T0 + 1.0, T0 + 3.6, "item_greet", "Hi, this is Ava from Acme. How can I help you today?",
        tts_node_ttfb=0.21)

    hear(session, T0 + 4.2, T0 + 6.9, T0 + 7.4, "item_u1", "Hi, I'm checking on my order. It's 4471.",
         confidence=0.96, transcription_delay=0.21, end_of_turn_delay=0.58)
    metric(session, STTMetrics(label="livekit.plugins.deepgram.stt.STT", request_id="stt_stream_1",
                               timestamp=T0 + 7.0, duration=0.0, audio_duration=7.0, streamed=True), T0 + 7.0)
    eou(session, T0 + 7.48, 0.58, 0.21, "speech_r1")
    sp1 = speech(session, T0 + 7.5, "speech_r1")
    metric(session, llm_metrics("chatcmpl-dummy1", T0 + 8.15, 0.41, 0.62, 1480, 1152, 19, "speech_r1"), T0 + 8.15)
    lookup = run_tool(session, T0 + 8.16, T0 + 8.74, "call_lookup_1", "lookup_order", {"order_id": "4471"},
                      '{"status": "shipped", "carrier": "UPS", "eta": "2026-10-29"}')
    metric(session, llm_metrics("chatcmpl-dummy2", T0 + 9.3, 0.38, 0.52, 1560, 1152, 24, "speech_r1"), T0 + 9.3)
    metric(session, tts_metrics("tts_r1", T0 + 9.6, 0.19, 82, "speech_r1"), T0 + 9.6)
    r1 = say(session, T0 + 9.7, T0 + 13.8, "item_a1",
             "Your order 4471 shipped with UPS and should arrive on October 29th.",
             llm_node_ttft=0.38, tts_node_ttfb=0.19, e2e_latency=2.3, provider_request_ids=["chatcmpl-dummy2"])
    sp1.speech_handle.chat_items = [lookup, r1]
    room.fire("sip_dtmf_received", T0 + 12.1, SimpleNamespace(code=1, digit="1", participant=room.caller))

    hear(session, T0 + 14.6, T0 + 16.2, T0 + 16.7, "item_u2", "Can you also update my delivery address?",
         confidence=0.91, transcription_delay=0.18, end_of_turn_delay=0.5)
    eou(session, T0 + 16.72, 0.5, 0.18, "speech_r2")
    sp2 = speech(session, T0 + 16.75, "speech_r2")
    error = llm.LLMError(timestamp=T0 + 17.4, label="livekit.plugins.openai.llm.LLM",
                         error=TimeoutError("Request timed out."), recoverable=True)
    session.emit("error", SimpleNamespace(type="error", error=error, source=session.llm, created_at=T0 + 17.4),
                 T0 + 17.4)
    metric(session, llm_metrics("chatcmpl-dummy3", T0 + 18.6, 0.47, 0.71, 1590, 0, 31, "speech_r2"), T0 + 18.6)
    metric(session, tts_metrics("tts_r2", T0 + 18.9, 0.22, 96, "speech_r2", cancelled=True), T0 + 18.9)
    state(session, "agent", "thinking", "speaking", T0 + 19.0)
    state(session, "user", "listening", "speaking", T0 + 20.1)  # the caller cuts in
    state(session, "agent", "speaking", "listening", T0 + 20.4)
    r2 = ChatMessage(id="item_a2", role="assistant", content=["Sure, I can help with that. What's the new"],
                     interrupted=True, created_at=T0 + 18.7,
                     metrics={"started_speaking_at": T0 + 19.0, "stopped_speaking_at": T0 + 20.4,
                              "llm_node_ttft": 0.47, "tts_node_ttfb": 0.22, "e2e_latency": 2.3,
                              "provider_request_ids": ["chatcmpl-dummy3"]})
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=r2, created_at=T0 + 20.4), T0 + 20.4)
    sp2.speech_handle.chat_items = [r2]
    state(session, "user", "speaking", "listening", T0 + 22.0)
    u3 = ChatMessage(id="item_u3", role="user", content=["Actually, just transfer me to billing."],
                     transcript_confidence=0.94, created_at=T0 + 22.0,
                     metrics={"started_speaking_at": T0 + 20.1, "stopped_speaking_at": T0 + 21.5,
                              "transcription_delay": 0.2, "end_of_turn_delay": 0.52})
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=u3, created_at=T0 + 22.02), T0 + 22.02)
    eou(session, T0 + 22.03, 0.52, 0.2, "speech_r3")

    sp3 = speech(session, T0 + 22.05, "speech_r3")
    metric(session, llm_metrics("chatcmpl-dummy4", T0 + 22.6, 0.36, 0.5, 1660, 0, 12, "speech_r3"), T0 + 22.6)
    transfer = run_tool(session, T0 + 22.61, T0 + 22.65, "call_transfer_1", "transfer_to_billing", {}, "transferred")
    sp3.speech_handle.chat_items = [transfer]
    session._agent = billing
    handoff = AgentHandoff(id="item_handoff", old_agent_id="receptionist", new_agent_id="billing",
                           created_at=T0 + 22.7)
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=handoff, created_at=T0 + 22.7),
                 T0 + 22.7)
    sp4 = speech(session, T0 + 22.75, "speech_r4")
    metric(session, llm_metrics("chatcmpl-dummy5", T0 + 23.3, 0.33, 0.48, 420, 0, 32, "speech_r4"), T0 + 23.3)
    metric(session, tts_metrics("tts_r4", T0 + 23.6, 0.2, 82, "speech_r4"), T0 + 23.6)
    r4 = say(session, T0 + 23.7, T0 + 29.3, "item_a3",
             "Hi, this is Ben in billing. I can update your delivery address. What's the new one?",
             llm_node_ttft=0.33, tts_node_ttfb=0.2, e2e_latency=1.7, provider_request_ids=["chatcmpl-dummy5"])
    sp4.speech_handle.chat_items = [r4]

    room.caller.disconnect_reason = rtc.DisconnectReason.CLIENT_INITIATED
    session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED, created_at=T0 + 31.1), T0 + 31.1)


def cascade_session():
    return Session(
        stt=plugin("livekit.plugins.deepgram.stt", model="nova-3", provider="deepgram",
                   _opts=SimpleNamespace(model="nova-3", language="en-US")),
        llm=plugin("livekit.plugins.openai.llm", model="gpt-4.1-mini", provider="openai"),
        tts=plugin("livekit.plugins.cartesia.tts", model="sonic-2", provider="cartesia",
                   _opts=SimpleNamespace(model="sonic-2", voice="a0e99841-438c-4a64-b679-ae501e7d6091")),
        vad=plugin("livekit.plugins.silero.vad", model="silero"),
        tools=[tool("end_call", "End the call.")],
        options=SimpleNamespace(turn_handling={
            "turn_detection": SimpleNamespace(model="turn-detector-v1-mini"),
            "endpointing": {"mode": "fixed", "min_delay": 0.5, "max_delay": 3.0},
            "interruption": {"enabled": True, "min_duration": 0.5, "min_words": 0},
            "preemptive_generation": {"enabled": True},
        }),
        usage=AgentSessionUsage(model_usage=[
            LLMModelUsage(provider="openai", model="gpt-4.1-mini", input_tokens=6710, input_cached_tokens=2304,
                          output_tokens=118),
            STTModelUsage(provider="deepgram", model="nova-3", audio_duration=22.1),
            TTSModelUsage(provider="cartesia", model="sonic-2", characters_count=312, audio_duration=13.9)]),
    )


def realtime_metrics(request_id, at, ttft, duration, given, produced, audio_in, audio_out, cached=0,
                     cancelled=False):
    return RealtimeModelMetrics(
        label="livekit.plugins.openai.realtime.RealtimeModel", request_id=request_id, timestamp=at,
        duration=duration, session_duration=0.0, ttft=ttft, cancelled=cancelled, input_tokens=given,
        output_tokens=produced, total_tokens=given + produced, tokens_per_second=0.0,
        input_token_details=RealtimeModelMetrics.InputTokenDetails(audio_tokens=audio_in, text_tokens=given - audio_in,
                                                                   cached_tokens=cached),
        output_token_details=RealtimeModelMetrics.OutputTokenDetails(audio_tokens=audio_out,
                                                                     text_tokens=produced - audio_out),
        metadata=Metadata(model_name="gpt-realtime", model_provider="openai"))


async def realtime_call(session, room):
    """OpenAI gpt-realtime: greeting; a lookup tool then its answer; a reply the caller cut off, with a failed
    response; the caller hangs up."""
    agent = SimpleNamespace(id="receptionist", instructions="You are Ava, a support agent for Acme.",
                            tools=[tool("lookup_order", "Look up an order by its number.")])
    # what LiveKit's running AgentActivity resolved: the realtime model takes the turns
    session._activity = SimpleNamespace(_rt_turn_detection_enabled=True, allow_interruptions=True, tools=[],
                                        _agent=None, _session=None)
    session.start(agent, T0 + 0.2)

    speech(session, T0 + 0.4, "speech_greet")
    metric(session, realtime_metrics("resp_greet", T0 + 0.45, 0.52, 2.9, 1210, 96, 0, 82), T0 + 0.45)
    say(session, T0 + 1.0, T0 + 3.4, "item_greet", "Hi, this is Ava from Acme. How can I help you today?",
        provider_request_ids=["resp_greet"])

    hear(session, T0 + 4.0, T0 + 6.6, T0 + 6.6, "item_u1", "Hi, I'm checking on my order, it's 4471.", language=None)
    metric(session, realtime_metrics("resp_tool", T0 + 6.7, 0.0, 0.6, 1290, 21, 160, 0, cached=1024), T0 + 6.7)
    lookup = run_tool(session, T0 + 7.25, T0 + 7.8, "call_lookup_1", "lookup_order", {"order_id": "4471"},
                      '{"status": "shipped", "carrier": "UPS", "eta": "2026-10-29"}')
    speech(session, T0 + 7.85, "speech_r1", items=[lookup])
    metric(session, realtime_metrics("resp_r1", T0 + 7.85, 0.61, 3.2, 1330, 141, 0, 120), T0 + 7.85)
    say(session, T0 + 8.5, T0 + 12.6, "item_a1", "Your order 4471 shipped with UPS and should arrive on October 29th.",
        provider_request_ids=["resp_r1"])

    hear(session, T0 + 13.4, T0 + 15.0, T0 + 15.0, "item_u2", "Can you also update my delivery address?",
         language=None)
    speech(session, T0 + 15.1, "speech_r2")
    metric(session, realtime_metrics("resp_r2", T0 + 15.1, 0.55, 1.3, 90, 74, 80, 69, cancelled=True), T0 + 15.1)
    error = llm.RealtimeModelError(timestamp=T0 + 15.4, label="livekit.plugins.openai.realtime.RealtimeModel",
                                   error=RuntimeError("Conversation already has an active response"),
                                   recoverable=True)
    session.emit("error", SimpleNamespace(type="error", error=error, source=session.llm, created_at=T0 + 15.4),
                 T0 + 15.4)
    state(session, "agent", "thinking", "speaking", T0 + 15.7)
    state(session, "user", "listening", "speaking", T0 + 16.6)
    state(session, "agent", "speaking", "listening", T0 + 16.8)
    r2 = ChatMessage(id="item_a2", role="assistant", content=["Sure, I can help with that. What's the new"],
                     interrupted=True, created_at=T0 + 15.4,
                     metrics={"started_speaking_at": T0 + 15.7, "stopped_speaking_at": T0 + 16.8,
                              "provider_request_ids": ["resp_r2"]})
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=r2, created_at=T0 + 16.8), T0 + 16.8)
    state(session, "user", "speaking", "listening", T0 + 18.4)
    u3 = ChatMessage(id="item_u3", role="user", content=["It's 12 Elm Street, Springfield."], created_at=T0 + 18.9,
                     metrics={"started_speaking_at": T0 + 16.6, "stopped_speaking_at": T0 + 18.4})
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=u3, created_at=T0 + 18.92), T0 + 18.92)
    speech(session, T0 + 18.5, "speech_r3")
    metric(session, realtime_metrics("resp_r3", T0 + 18.5, 0.58, 2.6, 410, 70, 370, 70), T0 + 18.5)
    say(session, T0 + 19.2, T0 + 22.3, "item_a3", "Got it, 12 Elm Street, Springfield. I've updated your address.",
        provider_request_ids=["resp_r3"])

    room.caller.disconnect_reason = rtc.DisconnectReason.CLIENT_INITIATED
    session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED, created_at=T0 + 24.1), T0 + 24.1)


def realtime_session():
    return Session(llm=OpenAIRealtime(), vad=None, usage=AgentSessionUsage(model_usage=[LLMModelUsage(
        provider="openai", model="gpt-realtime", input_tokens=4330, input_cached_tokens=1024, input_audio_tokens=610,
        input_text_tokens=3720, output_tokens=402, output_audio_tokens=341, output_text_tokens=61)]))


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """One clock, LiveKit's own RecordingManager with its two network calls stubbed, and recorded HTTP."""
    for name in ("SUPERBRYN_API_KEY", "SUPERBRYN_AGENT_ID", "SUPERBRYN_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://demo.livekit.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "APIdummy")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret_dummy")
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    call_report._caller_numbers_cache.clear()
    Clock.now = T0
    monkeypatch.setattr(call_report, "time", SimpleNamespace(time=lambda: Clock.now,
                                                             monotonic=lambda: Clock.now - T0 + 5000.0))
    monkeypatch.setattr(call_report, "_SIP_WAIT_S", 0.2)
    monkeypatch.setattr(call_report, "_POLL_S", 0.01)

    sent, egress = [], []

    async def http(method, url, headers, body):
        sent.append(SimpleNamespace(method=method, url=url, headers=headers,
                                    body=json.loads(json.dumps(body, allow_nan=False)) if body else None))
        if url.endswith("/caller-numbers"):
            return 200, {"numbers": [SUPERBRYN_NUMBER]}
        if body["surface"] == "testing":
            return 200, {"surface": "testing", "status": "linked", "is_test": True}
        return 200, {"surface": "observability", "ok": True}

    async def push_prompt(**kwargs):
        sent.append(SimpleNamespace(method="POST", url=kwargs["api_base_url"] + "/public-api/v1/prompts", body=kwargs))
        return True

    def recorder(**kwargs):
        manager = RecordingManager(**kwargs)

        async def credentials(room_name):
            return {"access_key": "AKIA_DUMMY", "secret_key": "dummy", "session_token": "dummy",
                    "bucket": "example-recordings", "region": "ap-south-1",
                    "base_url": "https://example-recordings.s3.ap-south-1.amazonaws.com"}

        async def start(request):
            egress.append(request)
            return SimpleNamespace(egress_id="EG_dummy01")

        async def stop(request):
            egress.append(request)

        async def aclose():
            pass

        manager._fetch_credentials = credentials
        manager.lkapi = SimpleNamespace(egress=SimpleNamespace(start_room_composite_egress=start, stop_egress=stop),
                                        aclose=aclose)
        return manager

    monkeypatch.setattr(call_report, "_http_request", http)
    monkeypatch.setattr(call_report, "push_prompt", push_prompt)
    monkeypatch.setattr(call_report, "RecordingManager", recorder)
    return SimpleNamespace(sent=sent, egress=egress)


async def run_call(offline, surface, pipeline):
    """One whole call: (call.started, call.ended); each is written to LIVEKIT_EVALS_DUMP_DIR when it is set."""
    phone = SUPERBRYN_NUMBER if surface == "testing" else CALLER  # a production caller is nobody of ours
    room = Room(f"call-acme-{pipeline}", f"RM_{pipeline}Demo01", phone)
    ctx = Ctx(room, test_call=surface == "testing" and pipeline == "cascade")  # by test ID, or by our number
    session = cascade_session() if pipeline == "cascade" else realtime_session()
    report = attach(ctx, session, api_key="sbryn_dummy", agent_id=AGENT_UUID, base_url="https://api.superbryn.test",
                    prompt=PROMPT, prompt_version="v3",
                    custom_data={"enterprise_id": "ent_42", "team_id": 7, "agent_type_id": "sales_inbound"})
    await asyncio.wait_for(asyncio.shield(report._task), 2)
    report.update_custom_data({"lead_source": "website"})
    await (cascade_call if pipeline == "cascade" else realtime_call)(session, room)
    Clock.now += 0.5
    await ctx.shutdown("room disconnected")

    posts = [c for c in offline.sent if c.url.endswith("/public-api/v1/calls/agent-data")]
    assert [c.body["event"] for c in posts] == ["call.started", "call.ended"]
    out = os.getenv("LIVEKIT_EVALS_DUMP_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        for post in posts:
            name = f"{surface}_{pipeline}_{post.body['event'].split('.')[1]}.json"
            (Path(out) / name).write_text(json.dumps(post.body, indent=2, ensure_ascii=False) + "\n")
    return posts[0].body, posts[1].body


def keys_anywhere(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            if key != "custom_data":
                yield from keys_anywhere(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_anywhere(item)


@pytest.mark.parametrize("pipeline", ["cascade", "realtime"])
@pytest.mark.parametrize("surface", ["testing", "observability"])
async def test_whole_call_bodies(offline, surface, pipeline):
    started, ended = await run_call(offline, surface, pipeline)
    call_id = f"RM_{pipeline}Demo01"

    for body in (started, ended):
        assert set(body) <= TOP_LEVEL
        assert REMOVED.isdisjoint(keys_anywhere(body))
        assert (body["surface"], body["source"], body["schema_version"], body["call_id"], body["agent_id"]) == (
            surface, "livekit", 1, call_id, AGENT_UUID)
        assert body["started_at"] == "2026-10-27T10:00:00.000Z"  # when the caller joined: every *_ms counts from it
    assert started["event_id"] == f"lk:{call_id}:started" and ended["event_id"] == f"lk:{call_id}:ended"
    assert {"transcript", "diagnostics", "context", "custom_data", "ended_at"}.isdisjoint(started)

    other = "observability" if surface == "testing" else "testing"
    assert other not in started and other not in ended
    if surface == "testing":
        assert (started["from"], started["to"]) == (SUPERBRYN_NUMBER, TRUNK)
        assert started.get("testing") == ({"superbryn_call_id": TEST_CALL_ID} if pipeline == "cascade" else None)
        assert offline.egress == []  # a test call is never recorded
    else:
        assert (started["from"], started["to"]) == (CALLER, TRUNK)
        assert started["observability"] == {"direction": "inbound"}
        observability = ended["observability"]
        assert re.fullmatch(r"https://example-recordings\.s3\.ap-south-1\.amazonaws\.com/call_recordings/"
                            rf"call-acme-{pipeline}/\d{{8}}-\d{{6}}/call\.mp3", observability["recording_url"])
        assert (observability["stereo_recording_url"], observability["recording_url_source"],
                observability["direction"]) == (observability["recording_url"], "superbryn_s3", "inbound")
        # the stereo layout the simulation service reads: the agent on one channel, everyone else on the other
        start, stop = offline.egress
        assert start.audio_mixing == egress_proto.AudioMixing.Value("DUAL_CHANNEL_AGENT")
        assert stop.egress_id == "EG_dummy01"

    assert (ended["ended_reason"], ended["ended_by"]) == ("participant_disconnected", "caller")
    assert ended["custom_data"] == {"enterprise_id": "ent_42", "team_id": 7, "agent_type_id": "sales_inbound",
                                    "lead_source": "website"}
    context = ended["context"]
    assert context["system_prompt"] == "You are Ava, a support agent for Acme."  # the first agent only
    assert context["prompt_ref"] == {"hash": call_report.prompt_ref(PROMPT)["hash"], "version": "v3"}
    assert (context["room_name"], context["participant_identity"], context["ring_started_at"]) == (
        f"call-acme-{pipeline}", room_identity(surface), "2026-10-27T09:59:57.600Z")
    assert "close" not in context

    transcript, diagnostics = ended["transcript"], ended["diagnostics"]
    agent_lines = [line for line in transcript if line["role"] == "agent"]
    assert all("provider_request_id" in line for line in agent_lines[1 if pipeline == "cascade" else 0:])
    assert diagnostics["pipeline"] == ("stt_llm_tts" if pipeline == "cascade" else "realtime")
    assert all(set(entry) <= {"turn_id", "transcriber_ms", "endpointing_ms", "model_ms", "voice_ms", "turn_ms"}
               for entry in diagnostics["turn_latencies"])
    [error] = diagnostics["errors"]
    assert (error["source_model"], error["source_provider"]) == (
        ("gpt-4.1-mini", "openai") if pipeline == "cascade" else ("gpt-realtime", "openai"))
    if pipeline == "cascade":
        assert diagnostics["eou_events"] == [{"at_ms": 7480, "eou_delay_ms": 580, "transcription_delay_ms": 210},
                                             {"at_ms": 16720, "eou_delay_ms": 500, "transcription_delay_ms": 180},
                                             {"at_ms": 22030, "eou_delay_ms": 520, "transcription_delay_ms": 200}]
        assert diagnostics["dtmf"] == [{"at_ms": 12100, "digit": "1"}]
        assert diagnostics["turn_latencies"][1] == {"turn_id": "item_a1", "transcriber_ms": 210, "endpointing_ms": 370,
                                                    "model_ms": 380, "voice_ms": 190, "turn_ms": 2300}
    else:
        assert "eou_events" not in diagnostics  # the realtime model takes the turns: LiveKit sends none
        assert diagnostics["usage"]["llm_input_audio_tokens"] == 610


def room_identity(surface):
    return f"sip_{SUPERBRYN_NUMBER if surface == 'testing' else CALLER}"


async def test_production_false_sends_nothing_for_a_production_call(offline):
    room = Room("call-acme-cascade", "RM_cascadeDemo01", CALLER)
    ctx = Ctx(room, test_call=False)
    session = cascade_session()
    report = attach(ctx, session, api_key="sbryn_dummy", agent_id=AGENT_UUID, base_url="https://api.superbryn.test",
                    production=False, prompt=PROMPT)
    assert await asyncio.wait_for(asyncio.shield(report._task), 2) is False
    await cascade_call(session, room)
    await ctx.shutdown("room disconnected")

    assert [c.method for c in offline.sent] == ["GET"]  # the caller-number check alone
    assert offline.egress == []
