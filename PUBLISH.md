# Publishing to PyPI

## How releases work

The version in `pyproject.toml` is the single source of truth. When a change to the package reaches `main`, the
"Publish to PyPI" workflow publishes **the declared version** if PyPI doesn't have it yet. It never bumps the
version and never writes to `main` (which is protected).

- Declared version already on PyPI: nothing is published. Merges that don't change the version are safe.
- New declared version: it is built and uploaded, provided `CHANGELOG.md` has a `## [<version>]` section.
- `pyproject.toml` and `livekit_evals/__init__.py` must agree, or the workflow stops.

## How to publish a release

In the pull request that ships the change:

```bash
pip install bump2version

# Pick one; it updates pyproject.toml, livekit_evals/__init__.py and .bumpversion.cfg together
bump2version patch    # bug fixes: 0.3.0 → 0.3.1
bump2version minor    # new features: 0.3.0 → 0.4.0
bump2version major    # breaking changes: 0.3.0 → 1.0.0
```

Then add a `## [<new version>] - <date>` section to `CHANGELOG.md`, and merge the pull request. The workflow publishes
the new version. It can also be started by hand from the Actions tab ("Run workflow").

## One-time setup

The workflow uploads with the `PYPI_API_TOKEN` repository secret:

1. Create an API token at https://pypi.org/manage/account/token/ (scope: the `livekit-evals` project).
2. GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:
   name `PYPI_API_TOKEN`, value the token.

## Monitoring

- **GitHub Actions**: repo → Actions → "Publish to PyPI"
- **PyPI page**: https://pypi.org/project/livekit-evals/
- **Verify install**: `pip install --upgrade livekit-evals`

## Manual publishing (alternative)

```bash
pip install build twine
rm -rf dist/ build/ *.egg-info
python -m build
twine upload dist/*
```

Publish only a version that is also declared on `main`, so the repo and PyPI stay in step.

## Troubleshooting

### "Version already published — nothing to do"
- Expected when the change didn't bump the version. Bump it in a pull request to release.

### "CHANGELOG.md has no '## [x.y.z]' section"
- Add the section for the new version and merge again.

### "pyproject.toml and livekit_evals/__init__.py disagree"
- Bump with `bump2version`, which keeps both in step, or fix the one that was edited by hand.

### "Invalid or non-existent authentication"
- Check the `PYPI_API_TOKEN` secret is set and hasn't expired.

### Workflow doesn't trigger
- It runs on pushes to `main` that change `livekit_evals/**`, `pyproject.toml`, `MANIFEST.in` or `README.md`, or by hand.
