from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from affine.core.environments import (
    EvaluationDeadlineExceeded,
    SDKEnvironment,
)


@pytest.mark.asyncio
async def test_sdk_environment_exposes_a_typed_outer_deadline():
    observed_kwargs = {}

    class _RemoteEnvironment:
        async def evaluate(self, **kwargs):
            observed_kwargs.update(kwargs)
            await asyncio.Event().wait()

    environment = SDKEnvironment.__new__(SDKEnvironment)
    environment.config = SimpleNamespace(
        name="TERMINAL",
        env_type="runtime",
        eval_params={},
        proxy_timeout=0.01,
    )
    environment._env = _RemoteEnvironment()
    miner = SimpleNamespace(
        hotkey="hk",
        revision="rev",
        model="org/model",
        inference_model=None,
        base_url="https://model.invalid/v1",
    )

    with pytest.raises(EvaluationDeadlineExceeded) as raised:
        await environment._evaluate_single(miner, task_id=1)

    assert raised.value.environment == "TERMINAL"
    assert raised.value.timeout_seconds == pytest.approx(0.01)
    assert "_timeout" not in observed_kwargs
