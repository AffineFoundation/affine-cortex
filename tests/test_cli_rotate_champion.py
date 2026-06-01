"""CLI parsing and command behavior for manual champion rotation."""

from __future__ import annotations

from click.testing import CliRunner

from affine.cli.main import cli
import affine.src.scheduler.commands as scheduler_commands
from affine.src.scorer.window_state import (
    BattleRecord,
    ChampionRecord,
    MinerSnapshot,
    TaskIdState,
)


def test_rotate_champion_forwards_flags(monkeypatch):
    calls = []

    async def fake_rotate_champion_command(*, commit, force, full_rotate):
        calls.append((commit, force, full_rotate))

    monkeypatch.setattr(
        scheduler_commands,
        "rotate_champion_command",
        fake_rotate_champion_command,
    )

    result = CliRunner().invoke(
        cli,
        ["rotate-champion", "--commit", "--force", "--full-rotate"],
    )

    assert result.exit_code == 0
    assert calls == [(True, True, True)]


def _install_command_fakes(monkeypatch, store, *, current_block=1000):
    async def noop():
        return None

    class _Subtensor:
        async def get_current_block(self):
            return current_block

    async def fake_get_subtensor():
        return _Subtensor()

    monkeypatch.setattr(scheduler_commands, "init_client", noop)
    monkeypatch.setattr(scheduler_commands, "close_client", noop)
    monkeypatch.setattr(scheduler_commands, "SystemConfigDAO", lambda: object())
    monkeypatch.setattr(
        scheduler_commands,
        "SystemConfigKVAdapter",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(scheduler_commands, "StateStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(scheduler_commands, "get_subtensor", fake_get_subtensor)


class _Store:
    def __init__(self, *, battle=None):
        self.champion = ChampionRecord(
            uid=1, hotkey="champ_hk", revision="champ_rev", model="org/champ",
        )
        self.battle = battle
        self.predeployed = []
        self.task_state = TaskIdState(
            task_ids={"ENV_A": [1, 2, 3]},
            refreshed_at_block=990,
        )
        self.ops = []

    async def get_champion(self):
        return self.champion

    async def get_battle(self):
        return self.battle

    async def get_predeployed_challengers(self):
        return self.predeployed

    async def get_task_state(self):
        return self.task_state

    async def set_task_state(self, state):
        self.ops.append(("set_task_state", state.refreshed_at_block))
        self.task_state = state

    async def clear_battle(self):
        self.ops.append(("clear_battle",))
        self.battle = None


def _battle():
    return BattleRecord(
        challenger=MinerSnapshot(
            uid=2, hotkey="chal_hk", revision="chal_rev", model="org/chal",
        ),
        deployment_id="wrk-chal",
        base_url="http://chal/v1",
        started_at_block=900,
    )


def test_rotate_champion_stale_only_refuses_active_battle(monkeypatch):
    store = _Store(battle=_battle())
    _install_command_fakes(monkeypatch, store, current_block=1000)

    result = CliRunner().invoke(cli, ["rotate-champion", "--commit"])

    assert result.exit_code == 0
    assert "REFUSING: battle in flight" in result.output
    assert store.ops == []


def test_rotate_champion_full_rotate_releases_stales_then_clears(monkeypatch):
    store = _Store(battle=_battle())
    _install_command_fakes(monkeypatch, store, current_block=1000)

    class _Queue:
        async def release_claim(self, uid, *, hotkey=None, revision=None):
            store.ops.append(("release_claim", uid, hotkey, revision))
            return True

    monkeypatch.setattr(scheduler_commands, "MinersQueueAdapter", lambda: object())
    monkeypatch.setattr(scheduler_commands, "ChallengerQueue", lambda adapter: _Queue())

    result = CliRunner().invoke(
        cli,
        ["rotate-champion", "--commit", "--full-rotate"],
    )

    stale_block = 1000 - scheduler_commands.WINDOW_BLOCKS - 1
    assert result.exit_code == 0
    assert store.ops == [
        ("release_claim", 2, "chal_hk", "chal_rev"),
        ("set_task_state", stale_block),
        ("clear_battle",),
    ]
