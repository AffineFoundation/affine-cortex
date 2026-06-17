import pytest

from affine.database.dao.system_config import SystemConfigDAO
from affine.src.monitor.miners_monitor import _system_miner_info


class _InMemorySystemConfigDAO(SystemConfigDAO):
    def __init__(self):
        self.system_miners = {}
        self.saved = None

    async def get_system_miners(self):
        return dict(self.system_miners)

    async def set_param(self, **kwargs):
        self.saved = kwargs
        self.system_miners = dict(kwargs["param_value"])
        return kwargs


@pytest.mark.asyncio
async def test_set_system_miner_stores_revision_and_model_type():
    dao = _InMemorySystemConfigDAO()

    await dao.set_system_miner(
        uid=1001,
        model="Qwen/Qwen3.6-35B-A3B",
        revision="main",
        model_type="qwen3_5_moe",
    )

    assert dao.system_miners == {
        "1001": {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "revision": "main",
            "model_type": "qwen3_5_moe",
        }
    }
    assert dao.saved["param_name"] == "system_miners"
    assert dao.saved["param_type"] == "dict"


def test_system_miner_info_preserves_deploy_metadata():
    info = _system_miner_info(
        1001,
        {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "revision": "main",
            "model_type": "qwen3_5_moe",
        },
    )

    assert info is not None
    assert info.uid == 1001
    assert info.hotkey == "SYSTEM-1"
    assert info.model == "Qwen/Qwen3.6-35B-A3B"
    assert info.revision == "main"
    assert info.hf_revision == "main"
    assert info.model_type == "qwen3_5_moe"
    assert info.is_valid is True


def test_system_miner_info_keeps_legacy_revision_fallback():
    info = _system_miner_info(1002, {"model": "openai/gpt-4o"})

    assert info is not None
    assert info.hotkey == "SYSTEM-2"
    assert info.revision == "SYSTEM-2"
    assert info.model_type == ""


def test_system_miner_info_ignores_regular_uids():
    assert _system_miner_info(1000, {"model": "Qwen/Qwen3.6-35B-A3B"}) is None
