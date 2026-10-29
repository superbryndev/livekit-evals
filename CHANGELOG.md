# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-28

One entry point and one report per call. `attach(ctx, session)` replaces both 0.3.x paths (`create_webhook_handler`
for production calls and `attach_test_data` for SuperBryn test calls) and posts one body to
`/public-api/v1/calls/agent-data`. This is a breaking change: see "Upgrading from 0.3.x" below and in the README.

### Added
- `attach(ctx, session, *, api_key, agent_id, production=True, record=True, custom_data, prompt, prompt_version, base_url)`,
  called once per job before `session.start()`; it registers its own shutdown hook and returns the call's handle:
  `set_custom_data()`, `update_custom_data()`, `set_call_end_reason()` and `set_recording_url()`.
- A `surface` per call: a SuperBryn test call (recognised as 0.3.x did: `superbryn_call_id` in the job or room
  metadata, the `X-SuperBryn-Call-Id` SIP header, or SuperBryn's caller number) is `testing`; every other call is
  `observability`, unless `production=False`, which sends nothing for it. Each call is reported once, as
  `call.started` (identity only) and `call.ended` from the job's shutdown after the hang-up (its work finishes within
  8 s), with 0.3.x's retries and the same `event_id` on each. A console run (`agent.py console`) sends nothing.
- Production calls are recorded in stereo (the agent on one channel, the caller on the other; 0.3.x recorded mono
  unless `stereo_recording=True`, and `record=False` turns it off in place of `disable_recording=True`) and send
  `observability.recording_url`, `stereo_recording_url`, `recording_url_source` and `direction`. Test calls are not
  recorded.
- Both surfaces: `transcript[].provider_request_id` on agent lines; `diagnostics.errors[].source_model` and
  `source_provider`; `diagnostics.eou_events[]` (`at_ms`, `eou_delay_ms`, `transcription_delay_ms`);
  `diagnostics.dtmf[]` (`at_ms`, `digit`); `ended_by` (`agent`, `caller` or `system`); `context` with
  `system_prompt` (the first agent's instructions as rendered when it became active), `prompt_ref`, `room_name`,
  `participant_identity`, `ring_started_at`, `environment` and, on a session that closed on an error, `close`;
  `custom_data` at the top level, exactly as given.

### Changed
- Fewer requests per call: SuperBryn's caller numbers and the prompt versions already pushed are kept per key in a
  temp file every job on the host reads (numbers for an hour, prompts for a day), and a key refused the numbers is
  asked again after an hour, not a minute (an outage still after a minute).
- Any valid SuperBryn API key reads the caller numbers and sends production calls (as the Monitor webhooks took
  any key), so keys made before `obs:write` existed keep working; a test call's data still needs `test_data:write`,
  and without it SuperBryn's phone tests are recognised and kept out of Monitor.
- Needs livekit-agents 1.6.7 or newer (the oldest release that gives every value this body sends) and Python
  3.10+, as livekit-agents does; 0.3.x declared livekit-agents 0.8 and Python 3.9.
- `started_at` is when the caller joined; for an outbound phone call, when `sip.callStatus` turned `active` (or `automation`), so its
  dialling and ringing are not call time (`context.ring_started_at` keeps when it was dialled). Every `start_ms`,
  `end_ms` and `at_ms` counts from it. An outbound call that never connected is reported from its dialling,
  unrecorded, with `ended_reason` `user_unavailable`, `user_rejected`, `sip_trunk_failure` or `not_answered`.
- Turn latency keys are snake_case: `turn_latencies[]` has `turn_id`, `transcriber_ms`, `endpointing_ms`,
  `model_ms`, `voice_ms` and `turn_ms`. `superbryn_call_id` moves into the `testing` block. `schema_version` is 1 for
  this body.
- One key: `SUPERBRYN_API_KEY` with both `test_data:write` and `obs:write`. `SUPERBRYN_AGENT_ID` (a UUID) is sent
  when set and optional with an agent key; any other value is left out with one warning instead of disabling
  reporting. Log lines start with `SUPERBRYN_CALL_REPORT_` (logger `livekit_evals`). The handle's setters never raise:
  `custom_data` takes any mapping, as 0.3.x did, and a value a setter can't take is left out with a warning. Prompt
  pushes name the agent (`SUPERBRYN_AGENT_ID`), so an organization key can push them, and a refused push is tried
  again after an hour instead of on every call.

### Removed
- `create_webhook_handler`, `WebhookHandler` and `attach_test_data`, the 0.3.x `call{}` body and its route
  (`/webhooks/obs/livekit`), and the `WEBHOOK_URL`, `AGENT_ID`, `VERSION_ID`, `LIVEKIT_PROJECT_ID`,
  `SUPERBRYN_TEST_API_KEY` and `SUPERBRYN_CREDENTIALS_URL` settings.
- Numbers the package worked out and fields SuperBryn no longer reads: `speech_stats`, the latency, turn-detection,
  interruption and VAD summaries, network and WebRTC stats, absolute per-turn timestamps, SIP attributes, the
  `*_enabled` flags, `livekit_project_id`, `call_rate_usd`, `speaker_id`, `tokens_per_second` and `total_tokens`,
  `response_delay_ms`, `recording_url_reachable`, each caller line's `language` (0.3.4; it stayed empty with an
  STT pipeline), and the handoff fields (per-line `agent_id`, `metadata.agents`,
  `diagnostics.agent_handoffs`). SuperBryn computes what it needs from the raw records.

### Upgrading from 0.3.x
Remove `create_webhook_handler`, `attach_to_session`, `ctx.add_shutdown_callback(send_webhook)` and
`attach_test_data`; add one `report = attach(ctx, session, custom_data=...)` before `session.start()`, with the same
`custom_data` keys and value types; call `report.set_custom_data()`, `update_custom_data()` and
`set_call_end_reason()` where you called them on the webhook handler. Put SuperBryn's Agent data key (both permissions) in
`SUPERBRYN_API_KEY` and remove `SUPERBRYN_TEST_API_KEY`, set `SUPERBRYN_AGENT_ID` to the agent's SuperBryn ID instead of a
free-form `AGENT_ID` (or drop it with an agent key), drop `WEBHOOK_URL`, and pin `livekit-evals>=0.4.0`. The README
has a before/after and a prompt that has a coding agent make the change.

## [0.3.4] - 2026-10-27

Found by end-to-end calls on real models (a cascade, GPT Realtime and Mini, Grok, Gemini Live, Nova Sonic, GPT Live; handoffs, tool errors, Hindi), each value checked against LiveKit's own events and the caller's clock. Cascaded agents change too: these are fixes to values they already sent.

### Added
- Handoffs: every turn carries `agent_id`, the agent that was running, and `metadata.agents` lists each agent with its instructions when more than one ran (`system_prompt` is still the first agent's).
- Test-call (eval) payload: each caller line's `language` as the STT heard it, `configuration.stt.language` (the language the STT was set to), and `diagnostics.agent_handoffs` with when each happened.

### Fixed
- Usage reports are counted once each, not once per request ID: streamed STT sends many reports on one ID, so a 65 s call counted 5 s of audio, and Nova Sonic counted 22 of 1641 input tokens.
- One caller message is one caller turn. STT finals no longer split it into pieces with 0 ms times; finals that arrive late, or after the reply started, stay in their turn. It starts and ends where LiveKit's voice detector saw the caller (`started_speaking_at`, `stopped_speaking_at`), not at the speaking-state change about 0.5 s later.
- `response_delay_ms` runs from the end of the caller turn the reply answers to the reply starting. It is never negative (cut-off replies sent −2697 ms), never counts from the caller turn's start, and a second reply to the same caller turn (after a tool call) gets none.
- A cascaded reply's `turn_latency` is its own LLM time to first token and TTS time to first byte (LiveKit 1.8 message metrics), not the average of every reply so far.
- `confidence_score` reads LiveKit 1.8's `transcript_confidence` (it was always null); a realtime model's stand-in 1.0 is dropped.
- Models that only stand in for the caller's times (Gemini Live, Nova Sonic, GPT Live, unless the code gives the session a VAD) send no number that can't be measured: caller line times, reply waits, `turn_latency`, `latency.llm_ms` / `total_ms` / `e2e_ms` and the caller-based `speech_stats` (caller talk time, talk ratio, silence) are null. LiveKit reports their time to first token as −1, 0 or a few milliseconds. The model is read from the session, else from the realtime metrics.
- `speech_stats.avg_response_delay_ms` and `max_response_delay_ms` are null when no reply has a wait, not 0.
- Provider labels: LiveKit Inference names (`vendor/model`) give the vendor, `amazon.*` is `aws` (Nova Sonic was `cartesia`), Grok is `xai` (was `openai`). `errors[].source_provider` names the vendor when the model is known.
- A realtime model's voice is sent as `tts_voice_id`.
- Eval mode names a realtime model, not the helper model its usage lists (GPT Live).

## [0.3.3] - 2026-10-27

Realtime (speech-to-speech) models only. Cascaded (STT, LLM, TTS) agents send exactly what they did before: no new fields and the same values.

### Added
- Each agent turn carries `provider_request_id`, the provider's ID for that reply from LiveKit's message metrics. It links the turn to its response and to the provider's own trace.
- `speech_stats.speech_created_count` (replies LiveKit started, greeting included) and `speech_stats.cancelled_responses` (responses the model reported as cancelled).

### Fixed
- A reply's `turn_latency.llm_ms` is that reply's own time to first token, found by its provider request ID, instead of the average of every reply so far. On a recorded GPT Realtime 2.1 call the third reply went out as 1434 ms but took 986 ms. It is 0 (not measured) when the response is unknown, reports no time (Gemini 3.1 Flash Live, Phonic), or shares its ID with other responses (Nova Sonic, GPT Live).

## [0.3.2] - 2026-10-21

### Added
- `usage` splits a realtime model's tokens into speech and text (`llm_input_audio_tokens`, `llm_input_text_tokens`, `llm_output_audio_tokens`, `llm_output_text_tokens`), which providers price apart. Sent only when the model reports speech tokens.
- `configuration.turn_detection.preemptive_generation`: whether LiveKit drafts a reply before the caller's turn is confirmed and drops it if they go on (on by default since 1.8, up to 3 drafts per turn). Those dropped drafts are most of a cascaded agent's early starts, so SuperBryn can name the right cause instead of the end-of-turn wait.

### Changed
- Nova Sonic sends no `llm_requests`. Its usage pieces share one request ID for the whole session and are stamped as each reply's audio arrives (within about 40 ms of the reply starting to play on staging calls), so neither a request's start nor what became of it can be told; 0.3.1's one-request-per-reply rebuild mislabelled spoken replies. Its tokens still count in `usage`. GPT Live, which also keeps one ID for the session, already sent none.

### Fixed
- A call with no failures sends `errors: []` instead of leaving `errors` out, which SuperBryn read as not sent: its error metrics were skipped on every clean call instead of reading 0. Sent only when the session keeps an event record (LiveKit records every event), so an SDK without one still sends nothing.
- A realtime model's caller times count as real only with a VAD the code passed to the agent or the session. LiveKit 1.8 gives every session a default VAD when none is passed, which realtime models with their own turn detection never run, so 0.3.1 sent Nova Sonic and Gemini Live calls reply waits of about 115 ms and placeholder caller line times. Those are left out again, as 0.3.1 meant.
- A caller line the hang-up cut off no longer runs to the call's end. LiveKit ends a caller line still open at the close at the close itself (on staging a one-word "No." ran 11.5 s for 0.7 s of speech, on 4 of 21 calls); such a line now ends where the voice detector saw the caller stop, and has no `end_ms` when the caller was still talking at the close.

## [0.3.1] - 2026-10-19

### Added
- A realtime model's voice is sent as `configuration.tts` (its provider and model, with the voice).
- `errors[].end_ms` for a failure LiveKit's retry recovered: when that step next worked. `errors[].side` is `agent` for LLM and TTS failures (STT stays `caller`).

### Fixed
- Realtime reply waits are sent only when the caller's stops are real: with a VAD on the session, or for models whose own speech detection reports them (OpenAI's realtime models, xAI's Grok). Nova Sonic and Gemini Live mark the caller stopped only once their reply starts (1–230 ms before it on recorded calls), so a wait from it would be the playback delay; Phonic and GPT Live are unverified. A reply whose response is found by provider ID pairs with the caller's last stop before that response was created and sends `turnLatency` and `modelLatency` (a `ttft` of 0 is a placeholder, no `modelLatency`); without the match, with the last stop before it started speaking, `turnLatency` alone.
- Caller lines carry no `start_ms`/`end_ms` on a realtime model whose caller times are stand-ins (as above), instead of times that end when the reply starts.
- Realtime records that share one request ID for the whole session (Nova Sonic's usage pieces, GPT Live's) become one LLM request per generation: a record with a new positive `ttft` starts one, the records before it are its input (the caller's audio, the prompt) and those carrying its `ttft` its output. On a Nova Sonic call that is one request per reply, with its start, its tokens (summing exactly to the session's) and what became of it; GPT Live's records hold no generation, so none. The empty record every realtime session opens with (no request ID, tokens or time to first token; it showed as a `stopped` or `spoken` request on every realtime call) is no request.
- A retried failure is recovered unless the same step failed for good before it next worked. Before, a final failure late in the call marked every earlier retried failure of that step as unrecovered.
- An LLM request's failure window runs to its end (a realtime response's duration from creation), not 1 s past its stamp.
- A failure stamped on the monotonic clock (Nova Sonic's) is moved onto the wall clock, so its `start_ms` is no longer 0.
- A realtime model's `configuration.interruptions` is `enabled` only: its own speech detection interrupts it, so LiveKit's duration and word gates don't apply.
- `configuration.interruptions` is what the running agent uses (an Agent's `allow_interruptions` and `turn_handling` over the session's), as turn detection already was.
- `usage.stt_audio_ms` is left out when it is 0 (OpenAI's realtime transcription bills tokens and reports no audio).
- LiveKit Inference STT's stand-in confidence of 1.0 is left out, as it already was for realtime models.

## [0.3.0] - 2026-10-16

### Added
- Eval mode: `attach_test_data(ctx, session)`, called once per job before `session.start()`, sends the agent's own record of each SuperBryn test call to SuperBryn: `call.started` when the call is recognised, then `call.ended` with the transcript (LiveKit's expressive `<expr/>` markup removed from agent text), tool calls, per-reply latencies, the models that ran (the current agent's own STT/LLM/TTS over the session's, adapters and fallback lists unwrapped), turn detection and interruption settings as LiveKit resolved them, the tools offered (MCP toolsets included), token, audio and character usage, every LLM request (realtime responses included) with its start, its tokens and what became of its answer: spoken, cut off, not spoken, stopped or only a tool call, and every STT, LLM or TTS failure with whether LiveKit's retry saved it. The requests are read from the session's recorded events when the call ends, so nothing subscribes to the deprecated `metrics_collected`. No audio is sent.
- A test call is recognised by `superbryn_call_id` in the job or room metadata, by the `X-SuperBryn-Call-Id` SIP header (an attribute ending in `x-superbryn-call-id` or `superbryn_call_id`), or by SuperBryn's caller number. The caller numbers are cached for an hour in a file in the temp dir that every job on the host shares.
- Settings: `SUPERBRYN_TEST_API_KEY` (falls back to `SUPERBRYN_API_KEY`), `SUPERBRYN_AGENT_ID` (required) and `SUPERBRYN_BASE_URL`, or `api_key=`, `agent_id=` and `base_url=`. Its info, warning and error log lines start with `SUPERBRYN_TEST_DATA_`.
- Built for production agents: it runs in a background task and one shutdown callback, never raises into the agent, sends strict JSON (a section that can't be sent is left out alone), and finishes its shutdown work within 8 s (LiveKit stops a job's process 10 s into its shutdown).

## [0.2.15] - 2026-10-01

### Added
- **Prompt sync.** `create_webhook_handler(prompt=..., prompt_version=..., first_message=..., prompt_tools=...)`: each call carries `metadata.prompt_ref` (`sha256` of the template plus your label) and the template is pushed once per process per version to `POST /public-api/v1/prompts` (needs the `obs:write` scope; fails open). The push goes to `SUPERBRYN_BASE_URL`, else the webhook URL's host. `prompt_ref` / `prompt_hash` / `push_prompt` are exported.

## [0.2.14] - 2026-09-28

### Added
- **Extended capture** (`extended_capture=True`, on by default): the handler now extracts the maximum telemetry the LiveKit SDK exposes and emits it as **additive** sections in the webhook payload. All existing fields are unchanged; consumers that don't know the new keys are unaffected. Set `extended_capture=False` for the exact legacy payload.
  - `call.turn_detection` — end-of-utterance (EOU) latency per user turn (`eou_events`, capped at 200) plus `avg/max_eou_delay_ms`, `avg_transcription_delay_ms`, and end-of-turn model inference stats. This is the previously-invisible chunk of perceived response latency.
  - `call.latency` gains `eou_ms`, `transcription_ms`, and `e2e_ms` (EOU + LLM TTFT + TTS TTFB) when EOU metrics were observed. Existing `llm_ms`/`stt_ms`/`tts_ms`/`total_ms` computed exactly as before.
  - `call.speech_stats` — user/agent talk seconds and turn counts, average turn lengths, agent talk ratio, approximate silence seconds, `longest_silence_ms`, avg/max response delay. Derived from the turn timings already captured; no new listeners.
  - `call.interruptions` — `agent_turns_interrupted` (derived), `false_interruptions` (agent paused for a non-interruption), and on SDKs ≥1.6 `detected_interruptions`, `backchannels`, `detection_delay_ms` from `InterruptionMetrics`.
  - `call.network` — per-participant `connection_quality_events` + `worst_connection_quality`, reconnect/disconnect events with reasons (incl. SIP-specific ones), `track_subscription_failures`, and a `rtc` block polled every 10s via `room.get_rtc_stats()`: jitter, RTT, packet loss %, concealed samples (audio dropouts), NACKs, jitter-buffer delay, bytes in/out — summary plus a bounded timeline (≤90 samples).
  - `call.errors` — typed provider failures from the session `error` event (`stt_error`/`llm_error`/`tts_error`/realtime), with message (truncated to 500 chars), `recoverable` flag, source model + provider. Capped at 50.
  - `call.close` — LiveKit's own close `reason` + terminal error, kept separate from the app-controlled `call_end_reason`.
  - `call.sip` — full `sip.*` participant attribute snapshot (callID, callStatus, trunk IDs, ...) and DTMF keypresses with timestamps (≤100).
  - `call.vad` — VAD inference count, avg inference ms, total inference seconds, idle time.
  - `call.environment` — livekit-agents / livekit / livekit-evals / Python versions and whether RTC stats were supported, so per-version capture gaps are auditable server-side.
  - `call.usage` gains `llm_prompt_cached_tokens`, `llm_tokens_per_second`, STT/TTS `*_avg_acquire_time_ms` (connection-pool health, SDK ≥1.6), and realtime-model token splits (`realtime_input_text/audio/cached_tokens`, `realtime_output_text/audio_tokens`).
- Version tolerance: extended metric types are matched by their `type` discriminator (no imports that break on older SDKs), each listener subscription is individually feature-detected, and `get_rtc_stats` is probed with `hasattr`. Verified against livekit-agents 1.4.2 and 1.6.5 — older SDKs simply produce fewer fields.
- Fail-soft guarantees: every extended section builder is exception-isolated (a broken section becomes `null` instead of dropping the webhook), all event lists are bounded, and the extended payload is JSON-sanitized before POST.

### Notes
- Extended capture subscribes to the session-level `metrics_collected` event (deprecated upstream but still emitted) because it is the only surface carrying VAD/EOU/EOT/interruption metrics; provider metrics arriving there are de-duplicated against the per-plugin path. A one-line deprecation warning in agent logs is expected.

## [0.2.13] - 2026-07-07

### Added
- Bring your own egress: `disable_recording=True` skips SuperBryn's recording, and `set_external_recording_url()` sends your own recording's URL (any time before the webhook fires).
- Agent handoffs: `AgentHandoff` items in the conversation stream are recorded instead of being treated as messages.

### Changed
- `agent_id`, `version_id` and the phone number are no longer read from job metadata; pass them explicitly (defaults `livekit-agent`, `v1`).

### Fixed
- Metrics collection updated to the current LiveKit metrics API.

## [0.2.12] - 2026-06-18

### Added
- Tool call capture via the LiveKit `function_tools_executed` event. Every tool/function invoked by the agent during a session is now collected and emitted as `payload["call"]["tool_calls"]` — a list of objects with `id`, `function_name`, `arguments` (raw JSON string), `result`, `is_error`, `start_ms`, `end_ms`, and `timestamp_ms`. Timing offsets are derived from `FunctionCall.created_at` and `FunctionCallOutput.created_at` (ms from call start), matching the canonical shape used by VAPI and Retell in the orchestration layer.

## [0.2.9] - 2026-06-03

### Added
- `custom_data` field in webhook payload (`payload["call"]["custom_data"]`). Accepts any JSON-serializable dict and is forwarded verbatim to the webhook endpoint, letting developers attach arbitrary session-level context (e.g. ticket IDs, user tiers, CRM fields, intent labels) without modifying the package.
- `custom_data` parameter on `create_webhook_handler()` for setting initial data known at startup.
- `WebhookHandler.set_custom_data(data)` — replaces `custom_data` entirely at any point before the webhook fires.
- `WebhookHandler.update_custom_data(data)` — shallow-merges new keys into the existing `custom_data` dict, useful for incrementally enriching context as tool calls resolve during the session.

## [0.2.8] - 2026-05-13

### Fixed
- Restored `tts_characters`, `tts_audio_duration_seconds`, and TTS latency tracking for sessions whose TTS is wrapped by custom classes that don't forward `metrics_collected` (e.g. `SanitizedTTS`, `VolumeTTS`, `MixedAudioTTS`, `NetworkGlitchTTS`). The framework only re-emits the event off `session.tts`, i.e. the outer wrapper, so any wrapper that forgot to relay it caused these counters to remain at zero (most visibly for Sarvam, which is always wrapped to sanitise input). `WebhookHandler.attach_to_session` now also subscribes to the underlying base TTS/STT/LLM (resolved via the existing `_unwrap_to_base_component` walker introduced in 0.2.6) and routes events through the same handler. `_on_metrics_collected` deduplicates by `(type, request_id, segment_id)` for `LLMMetrics` / `STTMetrics` / `TTSMetrics` / `RealtimeModelMetrics`, so adapters that do forward (`FallbackAdapter`, `StreamAdapter`) don't double-count. Subscriptions are released on session close.

## [0.2.7] - 2026-05-11

### Notes
- Version published from the same commit as 0.2.6 due to the auto-publish workflow's patch bump. Contents are identical to 0.2.6 (base-provider unwrap + `SUPERBRYN_CONFIG_LOADED` log line). Documented here to keep the changelog aligned with PyPI.

## [0.2.6] - 2026-05-11

### Added
- `SUPERBRYN_CONFIG_LOADED` log line emitted right after `WebhookHandler` initialization. It reports the resolved `webhook_url`, masked `api_key`, `credentials_url`, `base_url`, `livekit_project_id`, `is_deployed_on_lk_cloud`, `call_rate_usd`, recording flags (`disable_recording`, `stereo_recording`, `defer_recording`), recording manager status, and default `agent_id` / `version_id`, so the target environment is auditable from logs.
- Base-provider resolution for TTS/STT/LLM. `_extract_session_config` now descends through wrapper chains (LiveKit `FallbackAdapter` / `StreamAdapter` and custom wrappers like `NetworkGlitchTTS`, `MixedAudioTTS`, `SanitizedTTS`, `VolumeTTS`) via known inner-instance attributes (`_tts_instances`, `_wrapped_tts`, `_inner_tts`, `_inner`, `_base_tts`, `tts`, plus STT/LLM equivalents). It stops at the first `livekit.plugins.<provider>.*` module, so `tts_provider`/`stt_provider`/`llm_provider` now report the real base provider (e.g. `cartesia`, `sarvam`) instead of wrapper names (`fallback_adapter`, `mixed_tts`, `network_glitch`, `glitch`, `config`). Cycle-safe via an `id()`-keyed visited set.

### Added (from prior unreleased work)
- Public `WebhookHandler.set_call_end_reason(reason)` method to let agents persist semantic/business end reasons such as `transfer_to_human`, `conversation_complete`, or `no_answer_timeout` in the final webhook payload
- README documentation for semantic call end reasons and graceful shutdown usage

### Fixed
- Fixed `ModuleNotFoundError: No module named 'config'` by changing absolute import to relative import in `webhook_handler.py`

## [0.1.3] - 2025-11-06

### Added
- Expanded provider detection to support 25+ providers including:
  - **LLM**: OpenAI, Anthropic, Google, Meta (Llama), Mistral, Cohere, Perplexity, Groq, Together, Replicate, Hugging Face
  - **TTS**: ElevenLabs, Cartesia, PlayHT, Resemble, Murf, WellSaid, Speechify, Sarvam, Azure, AWS Polly, Google Cloud
  - **STT**: Deepgram, AssemblyAI, Rev, Speechmatics, Gladia
  - **Realtime**: LiveKit, Twilio, Vonage
- Support for custom telephony rates via `call_rate_usd` parameter:
  - Pass custom telephony rate per minute ($/min) to `create_webhook_handler()`
  - Automatically included in cost calculations
  - Overrides default provider costs when provided
  - Useful for custom telephony providers (Twilio, Vonage, etc.)

### Fixed
- Added missing `tts_voice_id` field to `usage_metrics` dictionary initialization to prevent KeyError when extracting TTS voice configuration
- Improved `tts_voice_id` extraction with fallback to use `tts_model` when no explicit voice_id attribute exists (common for providers like Sarvam where model name IS the voice)

### Planned
- CLI tool for testing webhook connectivity
- Custom webhook URL configuration
- Retry logic for failed webhook deliveries

## [0.1.0] - 2025-10-19

### Added
- Initial release of livekit-evals package
- Core `WebhookHandler` class for tracking LiveKit agent sessions
- `create_webhook_handler()` factory function for easy setup
- Automatic extraction of models and providers from session configuration
- Precise transcript tracking using VAD state change events
- Usage metrics tracking (LLM tokens, STT duration, TTS characters)
- Latency metrics tracking (LLM, STT, TTS with averages)
- Auto-detection of:
  - Agent ID and version from job metadata
  - System prompt from session agent
  - Phone number from job metadata
  - SIP trunking from participant attributes
  - Egress recording URLs
  - LiveKit project ID from environment
- Support for both voice pipelines and realtime models
- Comprehensive documentation and examples
- Provider detection for:
  - OpenAI (GPT, Whisper, TTS-1)
  - Anthropic (Claude)
  - Google (Gemini, PaLM)
  - Sarvam (Saarika)
  - ElevenLabs
  - Cartesia (Sonic)
  - Deepgram (Nova)
- Webhook payload with complete session data
- API key authentication via environment variable or parameter
- Detailed logging with prefixed messages for easy filtering
- Examples for basic voice pipeline and realtime model agents

### Security
- API key authentication for webhook delivery
- Secure environment variable handling
- No sensitive data logged in production

[Unreleased]: https://github.com/superbryndev/livekit-evals/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/superbryndev/livekit-evals/compare/v0.2.14...v0.3.0
[0.2.14]: https://github.com/superbryndev/livekit-evals/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/superbryndev/livekit-evals/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/superbryndev/livekit-evals/compare/v0.2.9...v0.2.12
[0.2.9]: https://github.com/superbryndev/livekit-evals/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/superbryndev/livekit-evals/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/superbryndev/livekit-evals/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/superbryndev/livekit-evals/compare/v0.1.3...v0.2.6
[0.1.3]: https://github.com/superbryndev/livekit-evals/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/superbryndev/livekit-evals/releases/tag/v0.1.0
