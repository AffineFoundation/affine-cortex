import pytest

from affine.src.scheduler.deploy_failures import (
    DeployFailureClassification,
    MISSING_PREPROCESSOR_CONFIG_REASON,
    QWEN36_PREPROCESSOR_FILE,
    classify_deploy_preflight_failure,
)


@pytest.mark.asyncio
async def test_preflight_classifies_missing_qwen36_preprocessor():
    async def missing_file(model: str, revision: str, filename: str):
        assert (model, revision, filename) == (
            "org/qwen36", "rev1", QWEN36_PREPROCESSOR_FILE,
        )
        return False

    result = await classify_deploy_preflight_failure(
        model="org/qwen36",
        revision="rev1",
        model_type="qwen3_5_moe",
        hf_file_exists_fn=missing_file,
    )

    assert result == DeployFailureClassification(
        rule_name="qwen36_missing_preprocessor_config",
        reason=MISSING_PREPROCESSOR_CONFIG_REASON,
    )


@pytest.mark.asyncio
async def test_preflight_ignores_non_qwen36_models():
    async def unexpected_probe(model: str, revision: str, filename: str):
        raise AssertionError("non-qwen36 model should not probe HF files")

    assert await classify_deploy_preflight_failure(
        model="org/qwen3",
        revision="rev1",
        model_type="qwen3",
        hf_file_exists_fn=unexpected_probe,
    ) is None


@pytest.mark.asyncio
async def test_preflight_keeps_inconclusive_hf_probe_retryable():
    async def inconclusive(model: str, revision: str, filename: str):
        return None

    assert await classify_deploy_preflight_failure(
        model="org/qwen36",
        revision="rev1",
        model_type="qwen3_5_moe",
        hf_file_exists_fn=inconclusive,
    ) is None
