"""HTTP behavior of sync_manifest / sync_config / async_sync_config.

The async path is exercised against a real local aiohttp server (success,
4xx, timeout). The blocking urllib path is exercised with mocks so no test
ever leaves the machine.
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
from unittest import mock

import pytest
from aiohttp import web

import livekit_evals.config_sync as config_sync
from livekit_evals.config_sync import async_sync_config, sync_config, sync_manifest

from .conftest import FakeAgent, make_llm

ACCEPTED = {
    "agent_row_id": "row-1",
    "approval_status": "pending",
    "verification_status": "pending",
    "hash": "abc",
    "change_types": ["llm"],
}


# ── credential / endpoint guards ─────────────────────────────────────────


def test_sync_manifest_requires_api_key(monkeypatch):
    monkeypatch.setitem(config_sync.WEBHOOK_CONFIG, "api_key", "")
    with pytest.raises(ValueError, match="API key missing"):
        sync_manifest({"source": "livekit"})


def test_async_sync_requires_api_key(monkeypatch):
    monkeypatch.setitem(config_sync.WEBHOOK_CONFIG, "api_key", "")
    with pytest.raises(ValueError, match="API key missing"):
        asyncio.run(async_sync_config(FakeAgent()))


def test_sync_manifest_requires_base_url(monkeypatch):
    monkeypatch.setattr(config_sync, "BASE_URL", "")
    with pytest.raises(ValueError, match="base URL"):
        sync_manifest({"source": "livekit"}, api_key="sk_agent_x")


# ── blocking path (urllib mocked) ────────────────────────────────────────


def test_sync_manifest_success():
    response = mock.MagicMock()
    response.read.return_value = json.dumps(ACCEPTED).encode()
    response.__enter__ = mock.Mock(return_value=response)
    response.__exit__ = mock.Mock(return_value=False)

    with mock.patch.object(config_sync.urllib.request, "urlopen", return_value=response) as opened:
        result = sync_manifest(
            {"source": "livekit"}, api_key="sk_agent_x", base_url="https://api.example.com"
        )

    assert result == ACCEPTED
    request = opened.call_args.args[0]
    assert request.full_url == "https://api.example.com/public-api/v1/agents/me/sync"
    assert request.get_header("X-api-key") == "sk_agent_x"


def test_sync_manifest_http_error_raises():
    error = urllib.error.HTTPError(
        url="https://api.example.com",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=io.BytesIO(b'{"error": "unknown keys"}'),
    )
    with mock.patch.object(config_sync.urllib.request, "urlopen", side_effect=error):
        with pytest.raises(urllib.error.HTTPError):
            sync_manifest(
                {"source": "livekit"}, api_key="sk_agent_x", base_url="https://api.example.com"
            )


def test_sync_manifest_timeout_raises():
    with mock.patch.object(
        config_sync.urllib.request, "urlopen", side_effect=TimeoutError("timed out")
    ):
        with pytest.raises(TimeoutError):
            sync_manifest(
                {"source": "livekit"}, api_key="sk_agent_x", base_url="https://api.example.com"
            )


def test_sync_config_builds_and_pushes():
    response = mock.MagicMock()
    response.read.return_value = json.dumps(ACCEPTED).encode()
    response.__enter__ = mock.Mock(return_value=response)
    response.__exit__ = mock.Mock(return_value=False)

    with mock.patch.object(config_sync.urllib.request, "urlopen", return_value=response) as opened:
        sync_config(
            FakeAgent(llm=make_llm()), api_key="sk_agent_x", base_url="https://api.example.com"
        )

    sent = json.loads(opened.call_args.args[0].data)
    assert sent["llm"]["provider"] == "openai"


# ── async path (real local server) ───────────────────────────────────────


async def _serve(handler):
    app = web.Application()
    app.router.add_post("/public-api/v1/agents/me/sync", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def test_async_sync_success_and_headers():
    async def scenario():
        seen = {}

        async def handler(request):
            seen["api_key"] = request.headers.get("X-API-Key")
            seen["body"] = await request.json()
            return web.json_response(ACCEPTED)

        runner, base_url = await _serve(handler)
        try:
            result = await async_sync_config(
                FakeAgent(llm=make_llm()), api_key="sk_agent_x", base_url=base_url
            )
        finally:
            await runner.cleanup()
        return result, seen

    result, seen = asyncio.run(scenario())
    assert result == ACCEPTED
    assert seen["api_key"] == "sk_agent_x"
    assert seen["body"]["llm"]["provider"] == "openai"


def test_async_sync_http_error_raises():
    async def scenario():
        async def handler(request):
            return web.json_response({"error": "org-scoped key"}, status=403)

        runner, base_url = await _serve(handler)
        try:
            await async_sync_config(FakeAgent(), api_key="sk_org_x", base_url=base_url)
        finally:
            await runner.cleanup()

    import aiohttp

    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(scenario())


def test_async_sync_timeout_raises():
    async def scenario():
        async def handler(request):
            await asyncio.sleep(5)
            return web.json_response(ACCEPTED)

        runner, base_url = await _serve(handler)
        try:
            await async_sync_config(
                FakeAgent(), api_key="sk_agent_x", base_url=base_url, timeout=0.2
            )
        finally:
            await runner.cleanup()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(scenario())
