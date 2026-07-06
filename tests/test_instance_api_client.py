from __future__ import annotations

import pytest

from affine.core.providers.instance_api_client import (
    DEFAULT_TIMEOUT_SEC,
    InstanceAPIClient,
    InstanceAPIConfig,
    InstanceAPIHTTPError,
    InstanceAPINotFoundError,
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
async def test_delete_renders_variables_and_sends_metadata(monkeypatch):
    client = InstanceAPIClient(
        InstanceAPIConfig(
            provider="lium",
            api_url="https://lium.example.com",
            delete_path="/pods/{instance_id}?purpose={purpose}",
        )
    )
    calls = []

    async def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(client, "_request", fake_request)

    assert await client.delete(
        "pod-1",
        variables={"purpose": "eval", "endpoint_name": "lium-b200-1"},
    ) is True

    args, kwargs = calls[0]
    assert args[:2] == ("DELETE", "/pods/pod-1?purpose=eval")
    assert kwargs["json"] == {
        "purpose": "eval",
        "endpoint_name": "lium-b200-1",
    }


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


@pytest.mark.asyncio
async def test_renew_raises_not_found_for_provider_404(monkeypatch):
    client = InstanceAPIClient(
        InstanceAPIConfig(
            provider="lium",
            api_url="https://lium.example.com",
            renew_path="/pods/{instance_id}/schedule-removal",
        )
    )

    async def fake_request(*args, **kwargs):
        raise InstanceAPIHTTPError(
            "POST",
            "/pods/pod-1/schedule-removal",
            404,
            '{"message":"Pod not found"}',
        )

    monkeypatch.setattr(client, "_request", fake_request)

    with pytest.raises(InstanceAPINotFoundError):
        await client.renew("pod-1")


@pytest.mark.asyncio
async def test_renew_raises_not_found_for_wrapped_provider_404(monkeypatch):
    client = InstanceAPIClient(
        InstanceAPIConfig(
            provider="lium",
            api_url="https://lium.example.com",
            renew_path="/instances/{instance_id}/renew",
        )
    )

    async def fake_request(*args, **kwargs):
        raise InstanceAPIHTTPError(
            "POST",
            "/instances/pod-1/renew",
            500,
            (
                '{"error":"LiumHTTPError","message":"POST '
                '/pods/pod-1/schedule-removal -> 404: '
                '{\\"message\\":\\"Pod not found\\",\\"status_code\\":404}"}'
            ),
        )

    monkeypatch.setattr(client, "_request", fake_request)

    with pytest.raises(InstanceAPINotFoundError):
        await client.renew("pod-1")
