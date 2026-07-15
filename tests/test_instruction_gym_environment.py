"""InstructionGym registration contract in affine-cortex."""

from __future__ import annotations

import json
from pathlib import Path

from affine.core.environments import (
    ENV_CONFIGS,
    EnvConfig,
    INSTRUCTION_GYM,
    INSTRUCTION_GYM_SUITE_ID,
    INSTRUCTION_GYM_TASK_ID_END,
    INSTRUCTION_GYM_UNIVERSE_ID,
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


def test_instruction_gym_canonical_config_and_aliases():
    config = ENV_CONFIGS["instruction-gym"]

    assert config.name == "instruction-gym"
    assert config.forward_api_key is False
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
    assert runtime["sampling"]["sampling_mode"] == "random"
