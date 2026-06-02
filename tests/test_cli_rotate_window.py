"""CLI parsing and command behavior for manual window rotation."""

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


def test_rotate_window_forwards_commit_flag(monkeypatch):
    calls = []

    async def fake_rotate_window_command(*, commit):
        calls.append(commit)

    monkeypatch.setattr(
        scheduler_commands,
        "rotate_window_command",
        fake_rotate_window_command,
    )

    result = CliRunner().invoke(cli, ["db", "rotate-window", "--commit"])

    assert result.exit_code == 0
    assert calls == [True]


def test_rotate_window_dry_run_default(monkeypatch):
    calls = []

    async def fake_rotate_window_command(*, commit):
        calls.append(commit)

    monkeypatch.setattr(
        scheduler_commands,
        "rotate_window_command",
        fake_rotate_window_command,
    )

    result = CliRunner().invoke(cli, ["db", "rotate-window"])

    assert result.exit_code == 0
    assert calls == [False]


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


def test_rotate_window_dry_run_shows_plan_without_writing(monkeypatch):
    store = _Store(battle=_battle())
    _install_command_fakes(monkeypatch, store, current_block=1000)

    result = CliRunner().invoke(cli, ["db", "rotate-window"])

    assert result.exit_code == 0
    assert "=== ROTATE plan ===" in result.output
    assert "(dry-run)" in result.output
    assert store.ops == []


def test_rotate_window_commit_aborts_without_sampling_confirm(monkeypatch):
    store = _Store(battle=_battle())
    _install_command_fakes(monkeypatch, store, current_block=1000)

    # Decline the "sampling service stopped?" confirmation.
    result = CliRunner().invoke(cli, ["db", "rotate-window", "--commit"], input="n\n")

    assert result.exit_code == 0
    assert "Aborted." in result.output
    assert store.ops == []


def test_rotate_window_commit_releases_stales_then_clears(monkeypatch):
    store = _Store(battle=_battle())
    _install_command_fakes(monkeypatch, store, current_block=1000)

    class _Queue:
        async def release_claim(self, uid, *, hotkey=None, revision=None):
            store.ops.append(("release_claim", uid, hotkey, revision))
            return True

    monkeypatch.setattr(scheduler_commands, "MinersQueueAdapter", lambda: object())
    monkeypatch.setattr(scheduler_commands, "ChallengerQueue", lambda adapter: _Queue())

    # Confirm the sampling service is stopped.
    result = CliRunner().invoke(cli, ["db", "rotate-window", "--commit"], input="y\n")

    stale_block = 1000 - scheduler_commands.WINDOW_BLOCKS - 1
    assert result.exit_code == 0
    assert store.ops == [
        ("release_claim", 2, "chal_hk", "chal_rev"),
        ("set_task_state", stale_block),
        ("clear_battle",),
    ]
