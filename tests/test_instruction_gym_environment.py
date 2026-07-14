"""InstructionGym registration contract in affine-cortex."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from affine.core.environments import (
    ENV_CONFIGS,
    EnvConfig,
    INSTRUCTION_GYM,
    INSTRUCTION_GYM_SUITE_ID,
    INSTRUCTION_GYM_TASK_ID_END,
    INSTRUCTION_GYM_UNIVERSE_ID,
    INSTRUCTION_GYM_SAMPLING_MANIFEST_SHA256,
    validate_execution_mode,
)


def test_env_config_preserves_legacy_positional_field_order():
    config = EnvConfig(
        "legacy",
        "example/image:tag",
        "affine",
        {},
        [],
        [],
        "7g",
        None,
        {"timeout": 12},
        34,
        "500m",
    )

    assert config.mem_limit == "7g"
    assert config.proxy_timeout == 34
    assert config.cpu_limit == "500m"
    assert config.forward_api_key is True
    assert config.supports_teacher_rollouts is True
    assert config.trusted_execution_modes == ("docker", "basilica")


def test_execution_mode_policy_error_does_not_expose_config_secrets():
    config = EnvConfig(
        name="secret-safe",
        docker_image="example/image:tag",
        env_vars={"API_KEY": "never-log-this-secret"},
        trusted_execution_modes=("docker",),
    )

    with pytest.raises(ValueError) as caught:
        validate_execution_mode(config, "basilica")

    assert "never-log-this-secret" not in str(caught.value)
    assert str(caught.value) == (
        "Environment 'secret-safe' refuses untrusted execution mode 'basilica'; "
        "allowed modes: docker"
    )


def test_instruction_gym_canonical_config_and_aliases():
    config = ENV_CONFIGS["instruction-gym"]

    assert config.name == "instruction-gym"
    assert config.forward_api_key is False
    assert config.supports_teacher_rollouts is False
    assert config.trusted_execution_modes == ("docker",)
    assert config.eval_params == {
        "protocol_version": "1.0",
        "universe_id": (
            "ifeval_templates_v4:"
            "6678152f3da165d389353a00c8b397a3fbf556f66e92e34bc1dcb194d1a6de53"
        ),
        "suite_id": "instruction_gym_ifeval_templates_v4",
        "temperature": 0.0,
        "timeout": 600,
    }
    assert config.eval_params["universe_id"] == INSTRUCTION_GYM_UNIVERSE_ID
    assert config.eval_params["suite_id"] == INSTRUCTION_GYM_SUITE_ID
    assert config.proxy_timeout > config.eval_params["timeout"] + 10

    for alias in (
        "INSTRUCTION-GYM",
        "InstructionGym",
        "instructiongym",
        "instruction_gym",
        "INSTRUCTION_GYM",
    ):
        assert ENV_CONFIGS[alias] is config
    assert callable(INSTRUCTION_GYM)


def test_instruction_gym_starts_disabled_with_exact_half_open_range():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "affine"
        / "database"
        / "system_config.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = payload["environments"]["INSTRUCTION-GYM"]

    assert runtime["enabled_for_sampling"] is False
    assert runtime["enabled_for_scoring"] is False
    assert runtime["sampling"]["dataset_range"] == [[0, INSTRUCTION_GYM_TASK_ID_END]]
    assert runtime["sampling"]["sampling_mode"] == "template_stratified_v1"
    assert (
        runtime["sampling"]["sampling_manifest_sha256"]
        == INSTRUCTION_GYM_SAMPLING_MANIFEST_SHA256
    )


@pytest.mark.parametrize("source", ["override", "hosts_config"])
def test_instruction_gym_rejects_public_execution_before_load(monkeypatch, source):
    from affine.core import environments as env_module

    env_module._ENV_CACHE.clear()
    load_calls = 0

    def forbidden_load(**kwargs):
        nonlocal load_calls
        load_calls += 1
        raise AssertionError(f"unexpected load_env call: {kwargs}")

    monkeypatch.setattr(env_module.af_env, "load_env", forbidden_load)
    if source == "override":

        def create():
            return env_module.SDKEnvironment("instruction-gym", mode="basilica")
    else:
        monkeypatch.setattr(
            env_module.SDKEnvironment,
            "_load_hosts_config",
            lambda self: {"instruction-gym": {"hosts": [], "mode": "basilica"}},
        )

        def create():
            return env_module.SDKEnvironment("instruction-gym")

    with pytest.raises(ValueError, match="refuses untrusted execution mode 'basilica'"):
        create()
    assert load_calls == 0


def test_instruction_gym_public_override_cannot_hide_behind_cached_local_env(
    monkeypatch,
):
    from affine.core import environments as env_module

    class CachedEnvironment:
        def is_ready(self):
            return True

    env_module._ENV_CACHE["instruction-gym"] = CachedEnvironment()
    monkeypatch.setattr(
        env_module.af_env,
        "load_env",
        lambda **kwargs: pytest.fail(f"unexpected load_env call: {kwargs}"),
    )
    try:
        with pytest.raises(ValueError, match="refuses untrusted execution mode"):
            env_module.SDKEnvironment("instruction-gym", mode="basilica")
    finally:
        env_module._ENV_CACHE.clear()


@pytest.mark.asyncio
async def test_evaluation_only_environment_never_enters_teacher_path(monkeypatch):
    from affine.src.teacher.worker import TeacherWorker

    worker = TeacherWorker(
        teacher_model="teacher",
        teacher_base_url="https://teacher.invalid/v1",
        api_key="teacher-secret",
        envs=["INSTRUCTION-GYM"],
    )

    class ForbiddenEnvironment:
        async def evaluate(self, **kwargs):
            pytest.fail(
                f"evaluation-only environment received evaluate kwargs: {kwargs}"
            )

    forbidden = ForbiddenEnvironment()
    worker._env_instances["INSTRUCTION-GYM"] = forbidden

    async def forbidden_sampling_list(env_name):
        pytest.fail(f"evaluation-only environment requested task IDs: {env_name}")

    monkeypatch.setattr(worker, "_get_sampling_list", forbidden_sampling_list)
    assert await worker._get_env("INSTRUCTION-GYM") is None
    assert await worker._generate_one_rollout() is None


@pytest.mark.asyncio
async def test_teacher_per_call_secret_respects_environment_policy(monkeypatch):
    from affine.core.environments import ENV_CONFIGS
    from affine.src.teacher.worker import TeacherWorker

    config = ENV_CONFIGS["knowledge-eval"]
    monkeypatch.setattr(config, "forward_api_key", False)
    worker = TeacherWorker(
        teacher_model="teacher",
        teacher_base_url="https://teacher.invalid/v1",
        api_key="teacher-secret",
        envs=["knowledge-eval"],
    )
    received = None

    class FakeEnvironment:
        async def evaluate(self, **kwargs):
            nonlocal received
            received = kwargs
            return {"score": 1.0, "extra": {"full_logprobs": [0.0]}}

    async def sampling_list(env_name):
        return range(1)

    async def get_env(env_name):
        return FakeEnvironment()

    monkeypatch.setattr(worker, "_get_sampling_list", sampling_list)
    monkeypatch.setattr(worker, "_get_env", get_env)
    rollout = await worker._generate_one_rollout()

    assert rollout is not None
    assert received == {
        "task_id": 0,
        "model": "teacher",
        "base_url": "https://teacher.invalid/v1",
        "collect_logprobs": True,
    }


@pytest.mark.asyncio
async def test_teacher_validates_mode_before_reusing_cached_environment(monkeypatch):
    from affine.core import environments as env_module
    from affine.src.teacher.worker import TeacherWorker

    config = ENV_CONFIGS["knowledge-eval"]
    monkeypatch.setattr(config, "trusted_execution_modes", ("docker",))
    monkeypatch.setattr(
        env_module.SDKEnvironment,
        "_get_hosts_and_mode",
        lambda self: ([], "basilica"),
    )
    worker = TeacherWorker(
        teacher_model="teacher",
        teacher_base_url="https://teacher.invalid/v1",
        api_key="teacher-secret",
        envs=["knowledge-eval"],
    )
    worker._env_instances["knowledge-eval"] = object()

    with pytest.raises(ValueError, match="refuses untrusted execution mode 'basilica'"):
        await worker._get_env("knowledge-eval")


def test_instruction_gym_sdk_rejects_direct_teacher_rollout() -> None:
    from affine.core.environments import ENV_CONFIGS, SDKEnvironment

    environment = SDKEnvironment.__new__(SDKEnvironment)
    environment.config = ENV_CONFIGS["instruction-gym"]

    with pytest.raises(ValueError, match="refuses collect_logprobs"):
        environment._prepare_eval_kwargs(task_id=0, collect_logprobs=True)

    prepared = environment._prepare_eval_kwargs(task_id=0, collect_logprobs=False)
    assert prepared["collect_logprobs"] is False
