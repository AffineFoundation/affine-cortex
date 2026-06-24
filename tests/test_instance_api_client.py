from __future__ import annotations

import pytest

from affine.core.providers.instance_api_client import (
    DEFAULT_TIMEOUT_SEC,
    InstanceAPIClient,
    InstanceAPIConfig,
    InstanceAPIHTTPError,
)


def test_instance_api_default_timeout_covers_long_provider_waits():
    cfg = InstanceAPIConfig.from_mapping("lium", {"create_path": "/pods"})

    assert cfg.timeout_sec == DEFAULT_TIMEOUT_SEC
    assert cfg.timeout_sec >= 20 * 60


@pytest.mark.asyncio
async def test_delete_treats_provider_404_as_success(monkeypatch):
    client = InstanceAPIClient(
        InstanceAPIConfig(
            provider="lium",
            api_url="https://lium.example.com",
            delete_path="/pods/{instance_id}",
        )
    )

    async def fake_request(*args, **kwargs):
        raise InstanceAPIHTTPError("DELETE", "/pods/pod-1", 404, "missing")

    monkeypatch.setattr(client, "_request", fake_request)

    assert await client.delete("pod-1") is True


@pytest.mark.asyncio
async def test_delete_reports_non_404_provider_errors(monkeypatch):
    client = InstanceAPIClient(
        InstanceAPIConfig(
            provider="lium",
            api_url="https://lium.example.com",
            delete_path="/pods/{instance_id}",
        )
    )

    async def fake_request(*args, **kwargs):
        raise InstanceAPIHTTPError("DELETE", "/pods/pod-1", 500, "boom")

    monkeypatch.setattr(client, "_request", fake_request)

    assert await client.delete("pod-1") is False
