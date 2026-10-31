"""Prompt push: each version goes over the wire once per host, even with a new process per call."""

import json
import os
import tempfile
import time

import pytest

from livekit_evals import prompt_sync

BASE = "https://api.superbryn.test"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # this test's record only
    monkeypatch.setattr(prompt_sync, "_PUSH_TIMEOUT_S", 1)  # a filtered port fails fast too
    prompt_sync._pushed.clear()
    prompt_sync._refused.clear()


@pytest.fixture
async def superbryn():
    """A local SuperBryn answering prompt pushes with `status`; the bodies it got."""
    from types import SimpleNamespace

    from aiohttp import web

    got = SimpleNamespace(bodies=[], status=400)

    async def push(request):
        got.bodies.append(await request.json())
        return web.json_response({}, status=got.status)

    app = web.Application()
    app.router.add_post(prompt_sync.PUSH_PATH, push)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    got.base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    yield got
    await runner.cleanup()


async def test_a_version_pushed_by_another_process_on_the_host_is_not_pushed_again():
    digest = prompt_sync.prompt_hash("You are a support agent.")
    prompt_sync._remember_pushed(prompt_sync._pushed_file(BASE, "k"), digest)
    # nothing listens on this port: a push would fail, so True means it was skipped
    assert await prompt_sync.push_prompt(api_key="k", api_base_url="http://127.0.0.1:9", template="You are a support agent.") is False
    assert await prompt_sync.push_prompt(api_key="k", api_base_url=BASE, template="You are a support agent.") is True
    assert (prompt_sync._key_id(BASE, "k"), digest) in prompt_sync._pushed


def test_the_record_is_per_key_and_forgets_a_version_after_a_day(tmp_path):
    assert prompt_sync._pushed_file(BASE, "k") != prompt_sync._pushed_file(BASE, "another agent's key")
    path = prompt_sync._pushed_file(BASE, "k")
    prompt_sync._remember_pushed(path, "sha256:new")
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    record["sha256:old"] = time.time() - 24 * 3600 - 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    assert set(prompt_sync._read_pushed(path)) == {"sha256:new"}
    assert [p.name for p in tmp_path.iterdir()] == [os.path.basename(path)]  # no temp file left behind


def test_an_unreadable_or_linked_record_is_ignored(tmp_path):
    path = prompt_sync._pushed_file(BASE, "k")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert prompt_sync._read_pushed(path) == {}
    os.remove(path)
    os.symlink(tmp_path / "elsewhere.json", path)
    assert prompt_sync._read_pushed(path) == {}


async def test_this_process_pushes_again_for_another_key_and_after_a_day():
    digest = prompt_sync.prompt_hash("You are a support agent.")
    down = "http://127.0.0.1:9"  # nothing listens: False means a push was tried
    prompt_sync._pushed[(prompt_sync._key_id(down, "a"), digest)] = time.time()
    assert await prompt_sync.push_prompt(api_key="a", api_base_url=down, template="You are a support agent.") is True
    assert await prompt_sync.push_prompt(api_key="b", api_base_url=down, template="You are a support agent.") is False
    prompt_sync._pushed[(prompt_sync._key_id(down, "a"), digest)] = time.time() - 2 * prompt_sync._PUSHED_TTL_S
    assert await prompt_sync.push_prompt(api_key="a", api_base_url=down, template="You are a support agent.") is False


async def test_a_push_names_its_agent_and_a_refused_one_waits_an_hour(superbryn, monkeypatch):
    template = "You are a support agent."
    assert await prompt_sync.push_prompt(api_key="org-key", api_base_url=superbryn.base, template=template,
                                         agent_id="3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b") is False
    assert superbryn.bodies == [{"template": template, "agent_id": "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"}]
    prompt_sync._refused.clear()  # a new process on the host reads the refusal from the record
    assert await prompt_sync.push_prompt(api_key="org-key", api_base_url=superbryn.base, template=template) is False
    assert len(superbryn.bodies) == 1
    later = time.time() + prompt_sync._REFUSED_TTL_S + 1
    monkeypatch.setattr(prompt_sync.time, "time", lambda: later)
    superbryn.status = 201
    assert await prompt_sync.push_prompt(api_key="org-key", api_base_url=superbryn.base, template=template) is True
    assert len(superbryn.bodies) == 2
