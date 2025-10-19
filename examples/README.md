# LiveKit Evals Examples

This directory contains example agents showing how to integrate `livekit-evals` into your LiveKit voice AI agents.

## Examples

### 1. `basic_agent.py` - Voice Pipeline Agent

A complete example showing:
- Voice pipeline setup (LLM + STT + TTS)
- Function calling (weather lookup)
- Metrics collection
- Full livekit-evals integration

**Run:**
```bash
python basic_agent.py dev
```

### 2. `realtime_model_agent.py` - Realtime Model Agent

A simpler example using OpenAI's Realtime API:
- Single realtime model (no separate STT/TTS/LLM)
- Same livekit-evals integration
- Cleaner setup for basic use cases

**Run:**
```bash
python realtime_model_agent.py dev
```

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install livekit-evals livekit-agents livekit-plugins-openai livekit-plugins-deepgram livekit-plugins-cartesia livekit-plugins-silero python-dotenv
   ```

2. **Set up environment variables** (create `.env` file):
   ```env
   # Required for livekit-evals
   SUPERBRYN_API_KEY=your_api_key_here
   
   # LiveKit credentials
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=your_livekit_api_key
   LIVEKIT_API_SECRET=your_livekit_api_secret
   
   # Provider API keys
   OPENAI_API_KEY=your_openai_key
   DEEPGRAM_API_KEY=your_deepgram_key
   CARTESIA_API_KEY=your_cartesia_key
   ```

3. **Get API keys:**
   - LiveKit: [https://cloud.livekit.io](https://cloud.livekit.io)
   - OpenAI: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Deepgram: [https://console.deepgram.com](https://console.deepgram.com)
   - Cartesia: [https://cartesia.ai](https://cartesia.ai)
   - Superbryn (livekit-evals): [https://your-platform.com/api-keys](https://your-platform.com/api-keys) *(placeholder)*

## Testing Locally

Run in development mode to test with LiveKit's local simulator:

```bash
python basic_agent.py dev
```

This will start the agent and open a browser window where you can test the voice interaction.

## Key Integration Points

Both examples show the same 3-line integration pattern:

```python
# 1. Create handler (before session setup)
webhook_handler = create_webhook_handler(
    room=ctx.room,
    is_deployed_on_lk_cloud=True
)

# 2. Start your session
await session.start(agent=YourAgent(), room=ctx.room)

# 3. Attach handler (after session.start, before ctx.connect)
if webhook_handler:
    webhook_handler.attach_to_session(session)
    ctx.add_shutdown_callback(webhook_handler.send_webhook)

await ctx.connect()
```

## What Gets Tracked

After running these examples, you'll see webhook payloads containing:

- 📝 Full conversation transcript with precise timing
- 📊 Token usage (LLM input/output)
- ⏱️ Latency metrics (LLM, STT, TTS)
- 🎯 Model and provider information
- 📞 Call duration and metadata
- 🔍 System prompts and configuration

Check your logs for `SPEECHIFY_WEBHOOK_SENT` to confirm successful delivery!

