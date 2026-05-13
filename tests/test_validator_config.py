from __future__ import annotations

import pytest

from affine.src.validator.main import ValidatorService


class _FakeAPIClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def get(self, path):
        self.calls.append(path)
        return self.response


class _ConfigFetchHarness:
    def __init__(self, response):
        self.api_client = _FakeAPIClient(response)
        self.watchdog_updates = []

    def update_watchdog(self, operation: str = ""):
        self.watchdog_updates.append(operation)


@pytest.mark.asyncio
async def test_validator_accepts_empty_public_config_without_retry():
    harness = _ConfigFetchHarness({"configs": {}})

    result = await ValidatorService.fetch_config_from_api(
        harness,
        max_retries=3,
        retry_interval=0,
    )

    assert result == {}
    assert harness.api_client.calls == ["/config"]


@pytest.mark.asyncio
async def test_validator_retries_malformed_config_response():
    harness = _ConfigFetchHarness({})

    result = await ValidatorService.fetch_config_from_api(
        harness,
        max_retries=2,
        retry_interval=0,
    )

    assert result is None
    assert harness.api_client.calls == ["/config", "/config"]
