# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/superbryndev/livekit-evals/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/superbryndev/livekit-evals/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/superbryndev/livekit-evals/releases/tag/v0.1.0

