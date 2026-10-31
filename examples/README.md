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
   # livekit-evals: one key with test_data:write and obs:write; the agent ID is needed with an org key
   SUPERBRYN_API_KEY=your_api_key_here
   SUPERBRYN_AGENT_ID=your_superbryn_agent_uuid
   
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
   - SuperBryn (livekit-evals): in [app.superbryn.com](https://app.superbryn.com), Monitor > Add Source > LiveKit

## Testing Locally

Run in development mode to test with LiveKit's local simulator:

```bash
python basic_agent.py dev
```

This will start the agent and open a browser window where you can test the voice interaction.

## Key Integration Point

Both examples add the same one line, before `session.start()`:

```python
from livekit_evals import attach

report = attach(ctx, session)  # registers its own shutdown hook
await session.start(agent=YourAgent(), room=ctx.room)
```

## What Gets Sent

Each call is reported to SuperBryn once (`call.started`, then `call.ended` after the hang-up): the transcript with
times, tool calls, each reply's latency parts, the models and settings, usage, LLM requests and failures. SuperBryn
test calls are reported for testing; every other call is a production call, recorded in stereo.

Check your logs for `SUPERBRYN_CALL_REPORT_SENT` to confirm delivery.
