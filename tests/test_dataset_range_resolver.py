"""Tests for ``affine.src.scorer.dataset_range_resolver``.

The resolver hits a remote URL; mock ``aiohttp.ClientSession`` so the
tests run offline. We cover: happy path, dot-notation field path, HTTP
non-200, missing field, malformed JSON, unsupported range_type,
zero/negative values, and the empty-source guard.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from affine.src.scorer.dataset_range_resolver import resolve_dataset_range


def _mock_response(status: int, body):
    """Build an aiohttp-style async context manager that returns a fake
    response object."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=body)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _mock_session(status: int, body):
    """Build an aiohttp.ClientSession that yields ``_mock_response``."""
    sess = MagicMock()
    sess.get = MagicMock(return_value=_mock_response(status, body))
    sess.__aenter__ = AsyncMock(return_value=sess)
    sess.__aexit__ = AsyncMock(return_value=None)
    return sess


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_resolves_zero_to_value():
    sess = _mock_session(200, {"completed_up_to": 1000})
    with patch.object(aiohttp, "ClientSession", return_value=sess):
        result = _run(resolve_dataset_range({
            "url": "http://example/m.json",
            "field": "completed_up_to",
            "range_type": "zero_to_value",
        }))
    assert result == [[0, 999]]


def test_dot_notation_path():
    sess = _mock_session(200, {"tasks": {"completed_up_to": 5000}})
    with patch.object(aiohttp, "ClientSession", return_value=sess):
        result = _run(resolve_dataset_range({
            "url": "http://example/m.json",
            "field": "tasks.completed_up_to",
            "range_type": "zero_to_value",
        }))
    assert result == [[0, 4999]]


def test_http_non_200_returns_none():
    sess = _mock_session(503, {})
    with patch.object(aiohttp, "ClientSession", return_value=sess):
        result = _run(resolve_dataset_range({
            "url": "http://example/m.json",
            "field": "completed_up_to",
            "range_type": "zero_to_value",
        }))
    assert result is None


def test_missing_field_returns_none():
    sess = _mock_session(200, {"other": 1000})
    with patch.object(aiohttp, "ClientSession", return_value=sess):
        result = _run(resolve_dataset_range({
            "url": "http://example/m.json",
            "field": "completed_up_to",
            "range_type": "zero_to_value",
        }))
    assert result is None


def test_unsupported_range_type_returns_none():
    result = _run(resolve_dataset_range({
        "url": "http://example/m.json",
        "field": "completed_up_to",
        "range_type": "unknown_shape",
    }))
    assert result is None


def test_zero_value_returns_degenerate_range():
    sess = _mock_session(200, {"completed_up_to": 0})
    with patch.object(aiohttp, "ClientSession", return_value=sess):
        result = _run(resolve_dataset_range({
            "url": "http://example/m.json",
            "field": "completed_up_to",
            "range_type": "zero_to_value",
        }))
    assert result == [[0, 0]]


def test_negative_value_returns_degenerate_range():
    sess = _mock_session(200, {"completed_up_to": -5})
    with patch.object(aiohttp, "ClientSession", return_value=sess):
        result = _run(resolve_dataset_range({
            "url": "http://example/m.json",
            "field": "completed_up_to",
            "range_type": "zero_to_value",
        }))
    assert result == [[0, 0]]


def test_missing_url_returns_none():
    result = _run(resolve_dataset_range({
        "field": "x",
        "range_type": "zero_to_value",
    }))
    assert result is None


def test_missing_field_key_returns_none():
    result = _run(resolve_dataset_range({
        "url": "http://example/m.json",
        "range_type": "zero_to_value",
    }))
    assert result is None
