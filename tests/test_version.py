"""Every version source agrees (bump2version needs them to), and the package exports attach_test_data."""

import re
from pathlib import Path

import livekit_evals

ROOT = Path(__file__).resolve().parents[1]


def test_every_version_source_agrees_and_exports_attach_test_data():
    pyproject = re.search(r'^version = "(.+)"$', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    bumpversion = re.search(r"^current_version = (.+)$", (ROOT / ".bumpversion.cfg").read_text(), re.M).group(1)

    assert livekit_evals.__version__ == pyproject == bumpversion
    assert "attach_test_data" in livekit_evals.__all__ and callable(livekit_evals.attach_test_data)
