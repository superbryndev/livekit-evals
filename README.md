# LiveKit Evals

[![PyPI version](https://badge.fury.io/py/livekit-evals.svg)](https://badge.fury.io/py/livekit-evals)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**One line in your LiveKit agent reports every call to SuperBryn.**

`attach(ctx, session)` sends your agent's own record of each call: the transcript, tool calls, each reply's latency
parts, the models and settings that ran, token, audio and character usage, every LLM request and what became of it,
and every STT, LLM or TTS failure. SuperBryn does every calculation on its side.

- **SuperBryn test calls** are reported for testing (surface `testing`): the agent's record sits next to the test.
- **Every other call** is a production call (surface `observability`): it is recorded in stereo and appears in
  Monitor. `production=False` turns production reporting off.

Each call is reported once, as `call.started` when the caller is in the room (an outbound phone call: once it is
answered) and `call.ended` when the job shuts down after the hang-up (its shutdown work finishes within 8 seconds). Everything runs in the background: it adds no latency and never raises into your agent.

## Install

```bash
pip install -U "livekit-evals>=0.4.0"
```

## Setup

```bash
export SUPERBRYN_API_KEY=sbryn_...                               # one key with test_data:write and obs:write
export SUPERBRYN_AGENT_ID=3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b  # the agent's SuperBryn ID (not needed with an agent key)
```

Add one line to your entrypoint, once per job, **before** `session.start()`:

```python
from livekit_evals import attach

async def entrypoint(ctx: JobContext):
    session = AgentSession(...)
    report = attach(ctx, session)
    await session.start(agent=Assistant(), room=ctx.room)
```

That's all: `attach` registers its own shutdown hook.

### Options

| Option | Environment variable | What it does |
|--------|----------------------|--------------|
| `api_key` | `SUPERBRYN_API_KEY` | A SuperBryn API key: test calls need `test_data:write` and prompt versions `obs:write`; production calls take any valid key (the setup screens' keys hold both). Without one nothing is sent. |
| `agent_id` | `SUPERBRYN_AGENT_ID` | The agent's SuperBryn ID, a UUID. Test calls need it unless the key is the agent's own key. Any other value is left out, with one warning. |
| `production` | | `True` by default. `False` reports SuperBryn test calls only: production calls send nothing and are not recorded. |
| `record` | | `True` by default. `False` reports production calls without SuperBryn's recording (for example for compliance). |
| `custom_data` | | Your own keys for this call, a dict sent as given (see below). |
| `prompt`, `prompt_version` | | The prompt template your agent runs, and your label for this version (see Prompt sync). |
| `base_url` | `SUPERBRYN_BASE_URL` | The SuperBryn API origin, `https://api.superbryn.com` by default. |

`WEBHOOK_URL`, `AGENT_ID`, `VERSION_ID`, `LIVEKIT_PROJECT_ID` and `SUPERBRYN_TEST_API_KEY`, which 0.3.x read, are not
read any more.

### The handle

`attach` returns the call's handle. Use it for anything known only during the call, until the call ends:

```python
report.set_custom_data({"ticket_id": "TKT-9001"})          # replaces custom_data
report.update_custom_data({"appointment_booked": True})    # merges into it; other keys are kept
report.set_call_end_reason("transfer_to_human")            # sent as ended_reason instead of LiveKit's close reason
report.set_recording_url("https://my-bucket/.../call.mp3")  # production calls: your own recording (see below)
```

### Custom data

`custom_data` is sent at the top of `call.ended` exactly as you give it: the same key names and the same value types
(a value JSON can't hold, such as a `datetime`, is sent as its text). SuperBryn's pre-ingestion filters and profiles
match these key names, so keys like `enterprise_id` or `team_id` route production calls to the right source and
profile. Set it in `attach(custom_data=...)` for what you know at the start, and with `set_custom_data()` or
`update_custom_data()` for what you learn during the call.

### Prompt sync

Pass the prompt template your agent runs (placeholders unfilled) so SuperBryn knows exactly which prompt each call
ran on:

```python
report = attach(ctx, session, prompt=PROMPT_TEMPLATE, prompt_version="v13")
```

Each call carries `context.prompt_ref = {"hash": "sha256:...", "version": "v13"}`, and the template's text is pushed
to SuperBryn once per host, key and version (`POST /public-api/v1/prompts`, which needs `obs:write`; pushed again after
a day, and a refused push is tried again after an hour). A prompt change
you deploy shows up in the agent's settings in SuperBryn. Without `prompt=`, the first agent's instructions as
rendered for the call are sent as `context.system_prompt` and matched as text.

## How a SuperBryn test call is recognised

- **Phone tests** need nothing more: they come from, or call, one of SuperBryn's numbers. The numbers are fetched
  with your key (any valid SuperBryn API key) and cached for an hour in a file in the temp dir that
  every job on the host shares; a refused key is asked again after an hour, an unanswered fetch after a minute.
  A test call's data is stored only with `test_data:write`: with a Monitor-only key, SuperBryn's test calls are
  recognised and kept out of Monitor.
- **SIP endpoint tests** carry an `X-SuperBryn-Call-Id` header. Set your inbound trunk's `include_headers` to X headers
  (`SIP_X_HEADERS`), or map `X-SuperBryn-Call-Id` with `headers_to_attributes` to an attribute named
  `superbryn_call_id` (any name ending in `superbryn_call_id` or `x-superbryn-call-id` works).
- **Webhook and Custom cURL triggers**: copy `superbryn_call_id` from SuperBryn's trigger request into the job
  (dispatch) or room metadata as `{"superbryn_call_id": "..."}`.

`attach` waits up to 15 seconds for the SIP participant to join; a caller in a browser (a standard participant) ends
the wait at once. A call SuperBryn says is not one of its tests is not reported again as a production call.

## Production calls

**Recording.** Unless `record=False`, a production call is recorded in stereo to SuperBryn's storage as soon as it is recognised (an outbound phone
call from its answer): one MP3,
the agent on one channel and everyone else (the caller) on the other. No S3 settings are needed: short-lived upload
credentials are fetched with your key. It needs LiveKit Egress (on by default on LiveKit Cloud) and the standard
`LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`; the egress is stopped when the job shuts down. A recording
that can't start leaves the call unrecorded, never unreported. Test calls are never recorded by the package.

**Your own recording.** If you run your own egress, pass its URL (public or pre-signed) with
`report.set_recording_url(url)`. Set before the call is recognised (for example right after `attach`), SuperBryn's
recording is not started; it is sent with `recording_url_source: "external"`.

`observability` also carries `direction` (`inbound` when a SIP dispatch rule brought the call to your agent,
`outbound` when your agent dialled).

## What a report contains

`POST {base_url}/public-api/v1/calls/agent-data`, with the key in `X-API-Key`. Every event repeats the call's identity:
`schema_version`, `event_id` (`lk:<room SID>:started` or `:ended`, the same on a retry), `event`, `surface`,
`source: "livekit"`, `call_id` (the room SID), `agent_id`, `from`, `to`, and `started_at`: when the caller joined, or for an
outbound phone call when it was answered (dialling and ringing are not call time; `context.ring_started_at` keeps when
it was dialled). Every `start_ms`, `end_ms` and `at_ms` counts from `started_at`. `call.ended` adds:

| Field | What it holds |
|-------|---------------|
| `ended_at`, `ended_reason`, `ended_by` | When the session closed, why (LiveKit's close reason or yours; an outbound phone call that never connected says `user_unavailable`, `user_rejected`, `sip_trunk_failure` or `not_answered`), and who ended it: `agent`, `caller` or `system`. |
| `transcript[]` | Each line: `id`, `role`, `text`, `start_ms`, `end_ms`; caller lines `confidence`; agent lines `interrupted` and `provider_request_id`. |
| `tool_calls[]` | `id`, `name`, `arguments`, `result`, `is_error`, `start_ms`, `end_ms`, and `turn_id`, the agent line it fed. |
| `diagnostics` | `pipeline`; `configuration` (STT, LLM and TTS with voice and language, turn detection, interruptions, tools); `turn_latencies[]` per reply (`transcriber_ms`, `endpointing_ms`, `model_ms`, `voice_ms`, `turn_ms`); `usage`; `llm_requests[]` with each one's outcome; `errors[]` with `source_model` and `source_provider`; `eou_events[]`; `dtmf[]`. |
| `context` | `system_prompt` (the first agent's instructions as rendered), `prompt_ref`, `room_name`, `participant_identity`, `ring_started_at`, `environment` (package, livekit-agents, livekit-rtc and Python versions), and `close` when the session closed on an error. |
| `custom_data` | Your keys, as given. |
| `testing` | Test calls: `superbryn_call_id`, when the call carried one. |
| `observability` | Production calls: `direction`, and `recording_url`, `stereo_recording_url`, `recording_url_source` on `call.ended`. |

No audio is sent, and the package computes no averages: SuperBryn works them out from these records.

## Logs

Log lines (logger `livekit_evals`) start with `SUPERBRYN_CALL_REPORT_` (the recorder's own details go to
`livekit_evals.recording_manager`): `SENT` when SuperBryn accepted an event (with
any optional field it dropped), `DISABLED` without an API key, `WARNING` for an agent ID that isn't a UUID or caller
numbers the key can't read, `RETRY`, `REJECTED` (with what to change) and `FAILED` for delivery problems, and `ERROR`
for anything else. None of them affects the call.

## Upgrading from 0.3.x

0.4.0 replaces both 0.3.x entry points (the production webhook handler and `attach_test_data`) with `attach`. It posts
a new body to a new endpoint, so pin the new version and change the setup in the same deploy.

**Before (0.3.x):**

```python
from livekit_evals import attach_test_data, create_webhook_handler

async def entrypoint(ctx: JobContext):
    webhook_handler = create_webhook_handler(
        room=ctx.room,
        is_deployed_on_lk_cloud=True,
        stereo_recording=True,
        custom_data={"enterprise_id": "ent_42", "team_id": "t-7", "agent_type_id": "sales_inbound"},
    )
    session = AgentSession(...)
    attach_test_data(ctx, session)
    await session.start(agent=Assistant(), room=ctx.room)
    if webhook_handler:
        await webhook_handler.attach_to_session(session)
        ctx.add_shutdown_callback(webhook_handler.send_webhook)
    ...
    webhook_handler.update_custom_data({"lead_score": 82})
    webhook_handler.set_call_end_reason("transfer_to_human")
```

**After (0.4.0):**

```python
from livekit_evals import attach

async def entrypoint(ctx: JobContext):
    session = AgentSession(...)
    report = attach(
        ctx, session,
        custom_data={"enterprise_id": "ent_42", "team_id": "t-7", "agent_type_id": "sales_inbound"},
    )
    await session.start(agent=Assistant(), room=ctx.room)
    ...
    report.update_custom_data({"lead_score": 82})
    report.set_call_end_reason("transfer_to_human")
```

Environment: `SUPERBRYN_API_KEY` is one key with both `test_data:write` and `obs:write` (it replaces
`SUPERBRYN_TEST_API_KEY`); `SUPERBRYN_AGENT_ID` is the agent's SuperBryn ID, a UUID (a free-form `AGENT_ID` is not
read; drop it when you use the agent's own key); remove `WEBHOOK_URL`.

What else changed:

- `create_webhook_handler`'s options are gone: production calls are recorded in stereo, and `disable_recording=True`
  is `attach(..., record=False)` (`stereo_recording`, `defer_recording` and `stop_egress()` are not needed), and
  `is_deployed_on_lk_cloud`, `livekit_project_id`, `call_rate_usd`, `extended_capture`, `first_message`,
  `prompt_tools` and `sync_prompt` are not read. `set_external_recording_url(url)` is `report.set_recording_url(url)`.
- The package no longer sends averages or counts (`speech_stats`, `latency`, turn-detection, interruption and VAD
  summaries, network and WebRTC stats, SIP attributes); SuperBryn computes them from the raw records. Turn latency
  keys are snake_case (`transcriber_ms`, `endpointing_ms`, `model_ms`, `voice_ms`, `turn_ms`).
- After a handoff, `context.system_prompt` is still the first agent's prompt; lines no longer carry an `agent_id`.

**A prompt for your coding agent** (copy and paste):

```text
Upgrade this LiveKit agent from livekit-evals 0.3.x to livekit-evals 0.4.0 (pip: "livekit-evals>=0.4.0").

1. Pin livekit-evals>=0.4.0 in the project's dependencies.
2. In the agent's entrypoint, remove every use of the old API: create_webhook_handler(...),
   webhook_handler.attach_to_session(session), ctx.add_shutdown_callback(webhook_handler.send_webhook),
   webhook_handler.start_recording(), webhook_handler.stop_egress(), and attach_test_data(ctx, session).
   Remove the matching imports.
3. Add one call, `report = attach(ctx, session, ...)` (from livekit_evals import attach), right after the
   AgentSession is created and BEFORE `await session.start(...)`. It registers its own shutdown hook.
4. If create_webhook_handler was given custom_data, pass the same dict to attach(custom_data=...). Keep every
   custom_data key name and every value type exactly as it is: SuperBryn's filters and profiles match on them.
   If it was given prompt= or prompt_version=, pass them to attach the same way. If it was given
   disable_recording=True, pass record=False to attach. Drop every other create_webhook_handler option
   (is_deployed_on_lk_cloud, livekit_project_id, call_rate_usd, stereo_recording, defer_recording,
   extended_capture, first_message, prompt_tools, sync_prompt).
5. Replace webhook_handler.set_custom_data(...), webhook_handler.update_custom_data(...) and
   webhook_handler.set_call_end_reason(...) with report.set_custom_data(...), report.update_custom_data(...) and
   report.set_call_end_reason(...), with the same arguments. Replace
   webhook_handler.set_external_recording_url(url) with report.set_recording_url(url). If the handler is passed to
   other functions or tools, pass `report` instead. Remove any `if webhook_handler:` guards: attach always
   returns a handle.
6. Environment and deployment config: put SuperBryn's Agent data key (it holds both
   test_data:write and obs:write) in SUPERBRYN_API_KEY and remove SUPERBRYN_TEST_API_KEY; replace a free-form AGENT_ID with SUPERBRYN_AGENT_ID set to the agent's
   SuperBryn agent ID (a UUID), or remove it when the key is the agent's own key; remove WEBHOOK_URL, VERSION_ID
   and LIVEKIT_PROJECT_ID if they were only used by livekit-evals.
7. Do not change anything else in the agent. Show the diff when done.
```

## Links

- [LiveKit Agents documentation](https://docs.livekit.io/agents/)
- [GitHub repository](https://github.com/superbryndev/livekit-evals)
- [Issue tracker](https://github.com/superbryndev/livekit-evals/issues)
- [Get an API key](https://app.superbryn.com/api-keys)

## License

MIT, see [LICENSE](LICENSE).

Made by [SuperBryn](https://www.superbryn.com). Support: support@superbryn.com
