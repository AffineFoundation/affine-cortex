"""CLI parsing for public miner lookup."""

from __future__ import annotations

from click.testing import CliRunner

from affine.cli.main import cli
import affine.src.miner.commands as miner_commands


def test_get_miner_accepts_positional_uid(monkeypatch):
    calls = []

    async def fake_get_miner_command(uid, hotkey):
        calls.append((uid, hotkey))

    monkeypatch.setattr(miner_commands, "get_miner_command", fake_get_miner_command)

    result = CliRunner().invoke(cli, ["get-miner", "87"])

    assert result.exit_code == 0
    assert calls == [(87, None)]


def test_get_miner_keeps_hotkey_option(monkeypatch):
    calls = []

    async def fake_get_miner_command(uid, hotkey):
        calls.append((uid, hotkey))

    monkeypatch.setattr(miner_commands, "get_miner_command", fake_get_miner_command)

    result = CliRunner().invoke(cli, ["get-miner", "--hotkey", "5Fabc"])

    assert result.exit_code == 0
    assert calls == [(None, "5Fabc")]
