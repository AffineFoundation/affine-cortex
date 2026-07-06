from __future__ import annotations

from types import SimpleNamespace

import pytest

from affine.src.validator import weight_setter as weight_setter_module
from affine.src.validator.main import ValidatorService
from affine.src.validator.weight_setter import WeightSetter


class _FakeSubtensor:
    def __init__(self, responses):
        self._responses = list(responses)
        self.set_calls = 0

    async def get_current_block(self):
        return 123

    async def set_weights(self, **kwargs):
        self.set_calls += 1
        return self._responses.pop(0)


async def _no_sleep(_seconds):
    return None


@pytest.mark.asyncio
async def test_set_weights_uses_extrinsic_success_flag(monkeypatch):
    subtensor = _FakeSubtensor(
        [
            SimpleNamespace(success=False, message="rejected", error=None),
            SimpleNamespace(success=False, message="still rejected", error=None),
        ]
    )

    async def fake_get_subtensor():
        return subtensor

    monkeypatch.setattr(weight_setter_module, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(weight_setter_module.asyncio, "sleep", _no_sleep)

    result = await WeightSetter(object(), 120).set_weights(
        {"1": {"weight": 1.0}},
        max_retries=2,
    )

    assert result is False
    assert subtensor.set_calls == 2


@pytest.mark.asyncio
async def test_set_weights_accepts_successful_extrinsic_response(monkeypatch):
    subtensor = _FakeSubtensor(
        [SimpleNamespace(success=True, message="ok", error=None)]
    )

    async def fake_get_subtensor():
        return subtensor

    monkeypatch.setattr(weight_setter_module, "get_subtensor", fake_get_subtensor)

    result = await WeightSetter(object(), 120).set_weights(
        {"1": {"weight": 1.0}},
        max_retries=2,
    )

    assert result is True
    assert subtensor.set_calls == 1


class _FakeWeightSetter:
    async def set_weights(self, api_weights, burn_percentage):
        return False


class _RunIterationHarness:
    def __init__(self):
        self.weight_setter = _FakeWeightSetter()
        self.watchdog_updates = []

    def update_watchdog(self, operation: str = ""):
        self.watchdog_updates.append(operation)

    async def fetch_weights_from_api(self):
        return {"weights": {"1": {"weight": 1.0}}}

    async def fetch_config_from_api(self):
        return {"validator_burn_percentage": 0.0}


@pytest.mark.asyncio
async def test_run_iteration_marks_failed_when_set_weights_returns_false():
    harness = _RunIterationHarness()

    await ValidatorService.run_iteration(harness)

    assert harness.watchdog_updates[-1] == "weights set failed"
