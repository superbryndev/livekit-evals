# livekit-evals — Agent Guide

Public PyPI package (`livekit-evals`) that tracks LiveKit agent sessions
(metrics, transcripts, usage webhooks) and optionally syncs agent config to
SuperBryn as a reviewable draft. Python >= 3.10.

## Layout

```text
livekit_evals/
  webhook_handler.py    # session tracking + webhook delivery (legacy style, ruff-excluded)
  recording_manager.py  # egress recording (legacy style, ruff-excluded)
  config_sync.py        # opt-in agent config sync (manifest build + push)
  _component_unwrap.py  # shared cycle-safe TTS/STT/LLM wrapper unwrapping
  config.py             # env-driven defaults (SUPERBRYN_BASE_URL, SUPERBRYN_API_KEY, ...)
tests/                  # pytest suite — no network, no real livekit-agents needed
```

## Non-negotiable invariants

- **No secrets are ever extracted or transmitted.** Extraction reads a fixed
  allow-list of configuration attributes (which includes private fields like
  `_opts` / `_model` / `_voice`); credential attributes (api_key, token,
  secret) must never be added to any candidate list. `tests/` locks this down.
- **Nothing syncs implicitly.** Config sync only runs when the customer calls
  `sync_config` / `async_sync_config` explicitly. Installing the webhook
  handler must never trigger a sync.
- **Extraction failures degrade, never raise.** A sparser manifest is fine;
  breaking the customer's agent is not.
- **No project-wide source scanning.** Static scanning of customer source
  trees was removed deliberately (confidentiality risk); do not reintroduce
  it. Missing sections are supplied via explicit keyword overrides.
- **Wrapped components must be unwrapped** via
  `_component_unwrap._unwrap_to_base_component` before reading provider/model
  fields; `FallbackAdapter` fallbacks are represented in the manifest's
  `fallback` sub-blocks.

## Versioning and publishing

- `.bumpversion.cfg` keeps `pyproject.toml`, `livekit_evals/__init__.py`,
  and itself in lockstep (`bump2version`).
- The publish workflow (`.github/workflows/publish.yml`) auto-bumps on merge
  to `main` based on the commit message (`[major]` / `BREAKING CHANGE`,
  `[minor]` / `feat:`, otherwise patch) and publishes to PyPI.
- Add a matching `CHANGELOG.md` section for every release. See `PUBLISH.md`
  for the release procedure.

## Development

```bash
pip install -e ".[dev]"       # or: pip install -e . pytest
pytest tests/ -v              # tests use fakes; no network or API keys
ruff format && ruff check     # webhook_handler/recording_manager are excluded
```

- The manifest schema mirrors the orchestration service's
  `AgentSyncManifest` (strict server-side validation). If the schema changes
  upstream, update the TypedDicts in `config_sync.py` and the tests together.
- `webhook_handler.py` / `recording_manager.py` carry pre-existing style
  debt; don't reformat them wholesale in unrelated changes.
- Update `CHANGELOG.md` for any behavior change.
