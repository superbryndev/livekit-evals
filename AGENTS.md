# AGENTS.md — livekit-evals

**PUBLIC PyPI package.** Customer-installed SDK: wraps a customer's LiveKit agent and ships
transcripts, usage, latency, and recordings to SuperBryn.

Workspace-wide rules: `../AGENTS.md`. Runtime topology: `../.claude/rules/cross-repo.md`.

## ⚠️ Pushing to `main` publishes to PyPI

`.github/workflows/publish.yml` runs on push to `main` and auto-increments the version.
There is no manual release gate.

It *is* path-filtered — only `livekit_evals/**`, `pyproject.toml`, `MANIFEST.in`, and
`README.md` trigger it, so a docs-only change is safe. (Note the asymmetry: the sibling SDK
`superbryn-pipecat-observer` has **no** path filter, where any push to `main` publishes.)
Don't rely on the filter without re-reading it — touching `pyproject.toml` or `README.md`
is a release.

- **Never push to `main` without explicit instruction** — a push is a public release.
- Anything committed here is world-readable: no internal URLs, keys, customer names, or
  internal service details in code, examples, tests, or commit messages.
- Breaking changes hit live customer integrations. `inbound-simulation-agent-v2` also pins
  this package (`livekit-evals==0.2.12`).

## Entrypoints

| Path | What |
|---|---|
| `livekit_evals/__init__.py` | Public API — the 3-line integration surface |
| `livekit_evals/config.py` | Base URL (`https://api.superbryn.com`), `X-API-Key` auth |
| `livekit_evals/webhook_handler.py` | Builds and posts the session payload |
| `livekit_evals/recording_manager.py` | Audio capture + S3 upload |
| `examples/` | Customer-facing usage — keep it working, it's documentation |

Ingest lands at `POST /webhooks/obs/livekit` in orchestration.

## Verify

```bash
python test_import.py     # the only check in the repo — an import smoke test
```

## Gotchas

- **No real test suite** — `test_import.py` is a smoke test, not coverage. Don't report
  "tests pass" as if it means the behaviour is verified.
- **The `/api/recording-credentials` STS path is legacy/dormant.** The live path is
  `/api/recording-upload-url` (presigned PUT). Don't build on the STS path.
- Supports Python 3.9+ — much wider than the rest of the workspace. Don't use 3.10+ syntax.
- `CHANGELOG.md` is customer-facing; update it with any behaviour change.

## PRs

Tracked branch `main`; PR base `main`. Review-before-merge discipline applies double here.
