# LiveKit Evals

[![PyPI version](https://badge.fury.io/py/livekit-evals.svg)](https://badge.fury.io/py/livekit-evals)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Track and evaluate your LiveKit voice AI agents with just 3 lines of code.**

Automatically capture transcripts, usage metrics, latency data, and session analytics from your LiveKit agents. Perfect for monitoring, debugging, and optimizing your voice AI applications.

## ✨ Features

- 🎯 **3-Line Integration** - Add to any LiveKit agent in seconds
- 📝 **Precise Transcripts** - Accurate timing using VAD state change events
- 📊 **Usage Metrics** - Track LLM tokens, STT duration, TTS characters
- ⚡ **Latency Tracking** - Monitor LLM, STT, and TTS performance
- 🔍 **Auto-Detection** - Automatically extracts models, providers, and configuration
- 📞 **SIP Support** - Detects SIP trunking and phone numbers
- 🎥 **Recording URLs** - Captures egress recording links
- 🔐 **Secure** - API key authentication with webhook delivery

## 🚀 Quick Start

### Prerequisites

1. **Get your API key** from [https://your-platform.com/api-keys](https://your-platform.com/api-keys) *(placeholder)*
2. **Set environment variable:**
   ```bash
   export SUPERBRYN_API_KEY=your_api_key_here
   ```

### Installation

```bash
pip install livekit-evals
```

### Integration (3 Lines)

Add these lines to your LiveKit agent:

```python
from livekit_evals import create_webhook_handler

async def entrypoint(ctx: JobContext):
    # ... your existing setup code ...
    
    # 1. Create webhook handler
    webhook_handler = create_webhook_handler(
        room=ctx.room,
        is_deployed_on_lk_cloud=True  # Set to False if self-hosting
    )
    
    # ... create your session ...
    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=cartesia.TTS(voice="..."),
    )
    
    # ... your session setup ...
    await session.start(agent=YourAgent(), room=ctx.room)
    
    # 2. Attach to session (MUST be after session.start, before ctx.connect)
    if webhook_handler:
        webhook_handler.attach_to_session(session)
        # 3. Send webhook on shutdown
        ctx.add_shutdown_callback(webhook_handler.send_webhook)
    
    await ctx.connect()
```

**That's it!** 🎉 Your agent will now automatically track all session data and send it to your webhook endpoint.

## 📖 Full Example

Here's a complete working example:

```python
import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import cartesia, deepgram, openai, silero

# Import livekit-evals
from livekit_evals import create_webhook_handler

logger = logging.getLogger("agent")
load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions.""",
        )


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {"room": ctx.room.name}

    # Initialize webhook handler (auto-detects all metadata)
    webhook_handler = create_webhook_handler(
        room=ctx.room,
        is_deployed_on_lk_cloud=True  # Set to False if self-hosting
    )

    # Set up voice AI pipeline
    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="en"),
        tts=cartesia.TTS(voice="your-voice-id"),
        vad=silero.VAD.load(),
    )

    # Start the session
    await session.start(agent=Assistant(), room=ctx.room)

    # Attach webhook handler to capture events
    # IMPORTANT: Must be after session.start() and before ctx.connect()
    if webhook_handler:
        webhook_handler.attach_to_session(session)
        ctx.add_shutdown_callback(webhook_handler.send_webhook)

    # Connect to room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `SUPERBRYN_API_KEY` | ✅ Yes | API key for webhook authentication | - |
| `LIVEKIT_PROJECT_ID` | ⚪ Optional | LiveKit project ID | Auto-detected from `LIVEKIT_URL` |
| `AGENT_ID` | ⚪ Optional | Unique agent identifier | Auto-detected from job metadata or `"livekit-agent"` |
| `VERSION_ID` | ⚪ Optional | Agent version identifier | Auto-detected from job metadata or `"v1"` |

### Setting Environment Variables

**Linux/Mac:**
```bash
export SUPERBRYN_API_KEY=your_api_key_here
```

**Windows (CMD):**
```cmd
set SUPERBRYN_API_KEY=your_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:SUPERBRYN_API_KEY="your_api_key_here"
```

**Docker:**
```bash
docker run -e SUPERBRYN_API_KEY=your_api_key_here ...
```

**.env file:**
```env
SUPERBRYN_API_KEY=your_api_key_here
LIVEKIT_PROJECT_ID=my-project-id
AGENT_ID=my-agent
VERSION_ID=v1.0.0
```

## 📊 What Gets Tracked

### Transcript Data
- **Precise timing** using VAD state change events
- Speaker turns (user/assistant)
- Start/end timestamps (ISO 8601)
- Start/end times in milliseconds (relative to call start)
- Response delays between turns
- Interruption detection
- Confidence scores (when available)
- Language detection
- Speaker IDs

### Usage Metrics
- **LLM:** Input tokens, output tokens, total tokens, model, provider
- **STT:** Audio duration, model, provider
- **TTS:** Character count, audio duration, model, provider, voice ID

### Latency Metrics
- **LLM:** Time to first token (TTFT), total duration
- **STT:** Processing duration
- **TTS:** Time to first byte (TTFB), total duration
- **Aggregated:** Average latencies per component

### Session Metadata
- Agent ID and version
- LiveKit project ID
- System prompt
- Call duration
- Phone number (if SIP call)
- SIP trunking detection
- Egress recording URLs
- LiveKit Cloud deployment status

## 🔍 How It Works

1. **Event Listening:** Attaches to LiveKit session events (`user_state_changed`, `agent_state_changed`, `metrics_collected`, `conversation_item_added`)
2. **Data Aggregation:** Collects and processes events during the session
3. **Auto-Detection:** Extracts configuration from session objects and job metadata
4. **Webhook Delivery:** Sends comprehensive payload to webhook endpoint when session ends

### Webhook Payload Format

```json
{
  "event": "call.ended",
  "call": {
    "id": "room-name",
    "room_name": "room-name",
    "participant_identity": "user-123",
    "started_at": "2025-10-19T12:00:00.000Z",
    "ended_at": "2025-10-19T12:05:30.000Z",
    "duration_seconds": 330,
    "transcript": {
      "turns": [
        {
          "speaker": "user",
          "text": "Hello, how are you?",
          "timestamp": "2025-10-19T12:00:05.000Z",
          "start_timestamp": "2025-10-19T12:00:05.000Z",
          "end_timestamp": "2025-10-19T12:00:07.000Z",
          "start_time_ms": 5000,
          "end_time_ms": 7000,
          "interrupted": false,
          "confidence_score": 0.98,
          "language": "en"
        },
        {
          "speaker": "assistant",
          "text": "I'm doing great, thanks for asking!",
          "timestamp": "2025-10-19T12:00:08.000Z",
          "start_timestamp": "2025-10-19T12:00:08.000Z",
          "end_timestamp": "2025-10-19T12:00:11.000Z",
          "start_time_ms": 8000,
          "end_time_ms": 11000,
          "response_delay_ms": 1000,
          "interrupted": false
        }
      ]
    },
    "recording_url": "https://...",
    "metadata": {
      "agent_id": "my-agent",
      "livekit_project_id": "my-project",
      "llm_model": "gpt-4o-mini",
      "llm_provider": "openai",
      "stt_model": "nova-3",
      "stt_provider": "deepgram",
      "tts_model": "sonic-english",
      "tts_provider": "cartesia",
      "tts_voice_id": "...",
      "system_prompt": "You are a helpful assistant...",
      "sip_trunking_enabled": false,
      "egress_enabled": true,
      "lk_agent_enabled": true,
      "phone_number": null
    },
    "usage": {
      "llm_model": "gpt-4o-mini",
      "llm_provider": "openai",
      "llm_input_tokens": 1250,
      "llm_output_tokens": 850,
      "llm_total_tokens": 2100,
      "stt_provider": "deepgram",
      "stt_model": "nova-3",
      "stt_duration_seconds": 45.2,
      "audio_duration_seconds": 45.2,
      "tts_provider": "cartesia",
      "tts_model": "sonic-english",
      "tts_characters": 1200,
      "tts_audio_duration_seconds": 42.5
    },
    "latency": {
      "llm_ms": 450.5,
      "stt_ms": 120.3,
      "tts_ms": 180.7,
      "total_ms": 751.5
    }
  }
}
```

## 🛠️ Advanced Usage

### Custom API Key

Pass API key directly instead of using environment variable:

```python
webhook_handler = create_webhook_handler(
    room=ctx.room,
    is_deployed_on_lk_cloud=True,
    api_key="your_api_key_here"
)
```

### Custom LiveKit Project ID

```python
webhook_handler = create_webhook_handler(
    room=ctx.room,
    is_deployed_on_lk_cloud=True,
    livekit_project_id="my-custom-project-id"
)
```

### Self-Hosted Agents

If you're self-hosting your LiveKit agents (not using LiveKit Cloud):

```python
webhook_handler = create_webhook_handler(
    room=ctx.room,
    is_deployed_on_lk_cloud=False  # Important for cost calculation
)
```

### Custom Telephony Rates

If you're using custom telephony providers (Twilio, Vonage, etc.) with specific per-minute rates:

```python
webhook_handler = create_webhook_handler(
    room=ctx.room,
    is_deployed_on_lk_cloud=True,
    call_rate_usd=0.015  # Your custom rate per minute ($/min)
)
```

This overrides default provider costs and ensures accurate cost tracking for your telephony usage.

### Passing Metadata via Job Context

You can pass custom metadata when creating LiveKit jobs:

```python
# When creating a job
job_metadata = {
    "agent_id": "customer-support-bot",
    "version_id": "v2.1.0",
    "phone_number": "+1234567890"
}
```

The webhook handler will automatically extract these values.

## 🐛 Troubleshooting

### Webhook Not Sending

**Check API Key:**
```bash
echo $SUPERBRYN_API_KEY
```

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Look for these log messages:**
- `SUPERBRYN_WEBHOOK_HANDLER_CREATED` - Handler initialized
- `SUPERBRYN_WEBHOOK_SENT` - Webhook delivered successfully
- `SUPERBRYN_WEBHOOK_UNAUTHORIZED` - Invalid API key
- `SUPERBRYN_WEBHOOK_FAILED` - Delivery failed
- `SUPERBRYN_WEBHOOK_ERROR` - Exception occurred

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `SUPERBRYN_API_KEY not configured` | Missing API key | Set `SUPERBRYN_API_KEY` environment variable |
| `SUPERBRYN_WEBHOOK_UNAUTHORIZED` | Invalid API key | Verify your API key is correct |
| `SUPERBRYN_WEBHOOK_FORBIDDEN` | Expired/disabled key | Generate a new API key |
| `No empty turn found to fill` | State change timing issue | Usually harmless, check logs for patterns |

### Missing Transcript Data

Ensure `webhook_handler.attach_to_session(session)` is called:
- ✅ **After** `await session.start()`
- ✅ **Before** `await ctx.connect()`
- ✅ At the **end** of your entrypoint (no early returns)

### Provider Detection Issues

The package auto-detects providers from model names. Supported providers (25+):

**LLM Providers:**
- OpenAI (gpt, whisper, tts-1, o1, o3)
- Anthropic (claude)
- Google (gemini, palm, bard, gemma)
- Meta (llama, meta-llama)
- Mistral (mistral, mixtral)
- Cohere (cohere, command)
- Perplexity (perplexity, pplx)
- Groq
- Together AI (together, togethercomputer)
- Replicate
- Hugging Face (huggingface, hf-)

**TTS Providers:**
- ElevenLabs (eleven, elevenlabs)
- Cartesia (cartesia, sonic)
- PlayHT (playht, play.ht)
- Resemble AI (resemble, resembleai)
- Murf (murf, murf.ai)
- WellSaid Labs (wellsaid, wellsaidlabs)
- Speechify
- Sarvam (saarika, sarvam, bulbul)
- Azure/Microsoft (azure, microsoft)
- AWS Polly (aws, polly, amazon)
- Google Cloud (gcloud, google-cloud)

**STT Providers:**
- Deepgram (deepgram, nova, aura)
- AssemblyAI (assemblyai, assembly)
- Rev.ai (rev.ai, revai)
- Speechmatics
- Gladia

**Realtime/Multi-modal:**
- LiveKit
- Twilio
- Vonage

If your provider isn't detected, it will show as `"unknown"` but won't affect functionality.

## 📝 Migration Guide

If you're currently using the standalone `webhook_handler.py`:

**Before:**
```python
from webhook_handler import create_webhook_handler
```

**After:**
```python
from livekit_evals import create_webhook_handler
```

Everything else stays the same! The API is identical.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [GitHub Repository](https://github.com/superbryndev/livekit-evals)
- [Issue Tracker](https://github.com/superbryndev/livekit-evals/issues)
- [Get API Key](https://your-platform.com/api-keys) *(placeholder)*

## 💡 Support

- 📧 Email: support@superbryn.com
- 💬 GitHub Issues: [Report a bug](https://github.com/superbryndev/livekit-evals/issues)
- 📚 Documentation: [README](https://github.com/superbryndev/livekit-evals#readme)

---

Made with ❤️ by [SuperBryn](https://www.superbryn.com)

