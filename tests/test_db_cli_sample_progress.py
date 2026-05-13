"""Tests for the ``af db sample-progress`` helper bits."""

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
