"""Tests for the ``af db sample-progress`` helper bits."""

import pytest

import affine.database.cli as db_cli
from affine.database.cli import _parse_duration


def test_parse_duration_all():
    assert _parse_duration("all") == 0
    assert _parse_duration("") == 0
    assert _parse_duration("0") == 0


def test_parse_duration_seconds():
    assert _parse_duration("30s") == 30
    assert _parse_duration("1s") == 1


def test_parse_duration_minutes():
    assert _parse_duration("5m") == 300
    assert _parse_duration("90m") == 5400


def test_parse_duration_hours():
    assert _parse_duration("2h") == 7200
    assert _parse_duration("24h") == 86400


def test_parse_duration_bare_int():
    assert _parse_duration("120") == 120


def test_parse_duration_case_insensitive():
    assert _parse_duration("5M") == 300
    assert _parse_duration("ALL") == 0


class _FakeSystemConfigDAO:
    def __init__(self):
        self.values = {
            "champion": {
                "uid": 213,
                "hotkey": "champ_hk",
                "revision": "champ_rev",
            },
            "current_battle": {
                "challenger": {
                    "uid": 32,
                    "hotkey": "chal_hk",
                    "revision": "chal_rev",
                },
            },
            "current_task_ids": {
                "task_ids": {"LIVEWEB": list(range(441))},
                "refreshed_at_block": 123,
            },
            "environments": {
                "LIVEWEB": {
                    "enabled_for_sampling": True,
                    "sampling": {"sampling_count": 400},
                },
            },
            "worker_status_LIVEWEB": {
                "tasks_in_flight": 7,
                "tasks_succeeded": 10,
                "tasks_failed": 1,
                "total_execution_ms": 11000,
                "reported_at": 1000,
            },
        }

    async def get_param_value(self, name, default=None):
        return self.values.get(name, default)


class _FakeSampleResultsDAO:
    def _make_pk(self, hotkey, revision, env):
        return f"{hotkey}:{revision}:{env}"


class _FakeClient:
    def __init__(self):
        self.rows = {}

    async def query(self, **params):
        pk = params["ExpressionAttributeValues"][":pk"]["S"]
        return {"Items": self.rows.get(pk, [])}


def _rows(n: int, refresh_block: int = 123):
    return [
        {
            "refresh_block": {"N": str(refresh_block)},
            "timestamp": {"N": str(1_000_000 + i)},
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_sample_progress_default_reports_only_active_subjects(monkeypatch, capsys):
    fake_client = _FakeClient()
    fake_client.rows = {
        "champ_hk:champ_rev:LIVEWEB": _rows(441),
        "chal_hk:chal_rev:LIVEWEB": _rows(72),
    }

    async def _noop():
        return None

    monkeypatch.setattr(db_cli, "init_client", _noop)
    monkeypatch.setattr(db_cli, "close_client", _noop)
    monkeypatch.setattr(db_cli, "SystemConfigDAO", _FakeSystemConfigDAO)
    monkeypatch.setattr(db_cli, "SampleResultsDAO", _FakeSampleResultsDAO)
    monkeypatch.setattr("affine.database.client.get_client", lambda: fake_client)
    monkeypatch.setattr("affine.database.schema.get_table_name", lambda name: name)

    await db_cli._cmd_sample_progress("5m", "active")

    out = capsys.readouterr().out
    assert "uid=213" not in out
    assert "=== challenger  uid=32" in out
    assert "LIVEWEB" in out
    assert "  400    72" in out
