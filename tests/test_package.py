"""Package surface: exports, version consistency, import safety."""

from __future__ import annotations

import re
import socket
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_exports_are_importable():
    import livekit_evals

    for name in livekit_evals.__all__:
        assert hasattr(livekit_evals, name), f"__all__ lists {name} but it is not importable"


def test_source_scanning_is_gone():
    import livekit_evals

    assert not hasattr(livekit_evals, "scan_source_config")
    assert "scan_source_config" not in livekit_evals.__all__
    assert not (REPO_ROOT / "livekit_evals" / "codescan.py").exists()

    import inspect

    from livekit_evals import build_manifest_from_agent

    assert "scan_root" not in inspect.signature(build_manifest_from_agent).parameters


def test_version_is_consistent_everywhere():
    import livekit_evals

    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "(.+?)"', pyproject, re.MULTILINE).group(1)
    assert livekit_evals.__version__ == declared

    bumpversion = (REPO_ROOT / ".bumpversion.cfg").read_text()
    tracked = re.search(r"current_version = (.+)", bumpversion).group(1).strip()
    assert tracked == declared

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
    assert f"## [{declared}]" in changelog, (
        "CHANGELOG.md must have a section for the declared version"
    )


def test_import_without_env_or_network(monkeypatch):
    """Importing and building a manifest must work with no env vars and no sockets.

    The sync feature being unused ("flag off") must have zero side effects:
    no credentials required, no network touched.
    """
    monkeypatch.delenv("SUPERBRYN_API_KEY", raising=False)
    monkeypatch.delenv("SUPERBRYN_BASE_URL", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted during import/build")

    monkeypatch.setattr(socket.socket, "connect", no_network)

    import importlib

    import livekit_evals

    importlib.reload(livekit_evals.config_sync)
    importlib.reload(livekit_evals)

    manifest = livekit_evals.build_manifest_from_agent(object())
    assert manifest == {"source": "livekit"}
