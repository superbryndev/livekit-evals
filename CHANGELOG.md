# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Tool call capture via the LiveKit `function_tools_executed` event. Every tool/function invoked by the agent during a session is now collected and emitted as `payload["call"]["tool_calls"]` — a list of objects with `id`, `function_name`, `arguments` (raw JSON string), `result`, `is_error`, `start_ms`, `end_ms`, and `timestamp_ms`. Timing offsets are derived from `FunctionCall.created_at` and `FunctionCallOutput.created_at` (ms from call start), matching the canonical shape used by VAPI and Retell in the orchestration layer.

### Notes
- Extended capture subscribes to the session-level `metrics_collected` event (deprecated upstream but still emitted) because it is the only surface carrying VAD/EOU/EOT/interruption metrics; provider metrics arriving there are de-duplicated against the per-plugin path. A one-line deprecation warning in agent logs is expected.

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

[Unreleased]: https://github.com/superbryndev/livekit-evals/compare/v0.2.9...HEAD
[0.2.9]: https://github.com/superbryndev/livekit-evals/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/superbryndev/livekit-evals/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/superbryndev/livekit-evals/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/superbryndev/livekit-evals/compare/v0.1.3...v0.2.6
[0.1.3]: https://github.com/superbryndev/livekit-evals/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/superbryndev/livekit-evals/releases/tag/v0.1.0
