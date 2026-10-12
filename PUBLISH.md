# Publishing to PyPI

The package publishes to PyPI from `.github/workflows/publish.yml` when a change reaches `main`.

**The version in `pyproject.toml` is the single source of truth.** CI never bumps it. On every push to `main` that touches the package, CI publishes the declared version if it is not on PyPI yet, and skips the upload if it already is. `main` is protected, so CI cannot push a bump commit back; the bump goes in your PR.

## How to release

In your PR:

```bash
pip install bump2version

# pick one: keeps pyproject.toml, livekit_evals/__init__.py and .bumpversion.cfg in lockstep
bump2version patch   # bug fixes:        0.3.0 -> 0.3.1
bump2version minor   # new features:     0.3.0 -> 0.4.0
bump2version major   # breaking changes: 0.3.0 -> 1.0.0
```

Then add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md` and merge to `main`.

CI then:
1. Checks `pyproject.toml` and `livekit_evals/__init__.py` declare the same version.
2. Skips the run if that version is already on PyPI.
3. Fails if `CHANGELOG.md` has no `## [X.Y.Z]` section.
4. Builds, runs `twine check`, and uploads.

A PR that changes code but not the version publishes nothing. Re-run a release by hand from Actions → "Publish to PyPI" → Run workflow.

## One-time setup

1. Create a PyPI API token at https://pypi.org/manage/account/token/ scoped to `livekit-evals`.
2. In GitHub: Settings → Secrets and variables → Actions → New repository secret. Name `PYPI_API_TOKEN`, value the token.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Run says "already published, nothing to do" | You did not bump the version in the PR. |
| `pyproject.toml (X) and livekit_evals/__init__.py (Y) disagree` | Bumped one file by hand. Use `bump2version`. |
| `CHANGELOG.md has no '## [X.Y.Z]' section` | Add the section and push again. |
| `Invalid or non-existent authentication` | `PYPI_API_TOKEN` is missing or expired. |
| Workflow did not trigger | It only runs on `main` when `livekit_evals/**`, `pyproject.toml`, `MANIFEST.in` or `README.md` change. |

## Verify

- https://pypi.org/project/livekit-evals/
- `pip install --upgrade livekit-evals && python -c "import livekit_evals; print(livekit_evals.__version__)"`
