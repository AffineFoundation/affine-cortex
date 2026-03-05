"""Tests for core data models - Miner, SampleSubmission, Result."""

import json
import time
import pytest
from unittest.mock import MagicMock, patch
from affine.core.models import Miner, SampleSubmission, Result, _truncate


class TestTruncate:
    """Test the _truncate helper."""

    def test_none_returns_empty(self):
        assert _truncate(None) == ""

    def test_short_text_unchanged(self):
        assert _truncate("hello", 80) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("a" * 200, 10)
        assert len(result) <= 10
        assert "…" in result

    def test_empty_string(self):
        assert _truncate("") == ""


class TestMiner:
    """Test Miner model."""

    def test_basic_creation(self):
        miner = Miner(uid=1, hotkey="5abc123")
        assert miner.uid == 1
        assert miner.hotkey == "5abc123"
        assert miner.model is None

    def test_full_creation(self):
        miner = Miner(
            uid=42,
            hotkey="hotkey123",
            model="my_model",
            revision="v1.0",
            block=100,
            slug="my-slug",
        )
        assert miner.uid == 42
        assert miner.model == "my_model"
        assert miner.revision == "v1.0"
        assert miner.block == 100

    def test_model_dump_property(self):
        miner = Miner(uid=1, hotkey="key")
        # model_dump is an alias for dict
        data = miner.model_dump()
        assert data["uid"] == 1
        assert data["hotkey"] == "key"

    def test_optional_fields_default_none(self):
        miner = Miner(uid=0, hotkey="k")
        assert miner.model is None
        assert miner.revision is None
        assert miner.block is None
        assert miner.chute is None
        assert miner.slug is None
        assert miner.weights_shas is None


class TestSampleSubmission:
    """Test SampleSubmission model."""

    def test_basic_creation(self):
        sub = SampleSubmission(task_uuid="uuid-1", score=0.95, latency_ms=100)
        assert sub.task_uuid == "uuid-1"
        assert sub.score == 0.95
        assert sub.latency_ms == 100
        assert sub.extra == {}
        assert sub.signature == ""

    def test_negative_score_allowed(self):
        sub = SampleSubmission(task_uuid="t", score=-5.0, latency_ms=0)
        assert sub.score == -5.0

    def test_negative_latency_rejected(self):
        with pytest.raises(Exception):  # pydantic validation
            SampleSubmission(task_uuid="t", score=1.0, latency_ms=-1)

    def test_get_sign_data_deterministic(self):
        sub = SampleSubmission(
            task_uuid="test-uuid",
            score=0.123456,
            latency_ms=500,
            extra={"b": 2, "a": 1}
        )
        data1 = sub.get_sign_data()
        data2 = sub.get_sign_data()
        assert data1 == data2
        # Keys should be sorted in extra
        assert '"a": 1' in data1
        assert data1.index('"a"') < data1.index('"b"')

    def test_get_sign_data_format(self):
        sub = SampleSubmission(task_uuid="u1", score=1.5, latency_ms=100, extra={})
        data = sub.get_sign_data()
        assert data == "u1:1.500000:100:{}"

    def test_verify_without_signature_fails(self):
        sub = SampleSubmission(task_uuid="t", score=1.0, latency_ms=0, signature="")
        # Empty signature should fail verification
        result = sub.verify("some_hotkey")
        assert result is False

    def test_verify_invalid_hex_fails(self):
        sub = SampleSubmission(task_uuid="t", score=1.0, latency_ms=0, signature="not_hex")
        result = sub.verify("some_hotkey")
        assert result is False


class TestResult:
    """Test Result model."""

    def test_basic_creation(self):
        r = Result(env="coding", score=0.9, latency_seconds=1.5, success=True)
        assert r.env == "coding"
        assert r.score == 0.9
        assert r.success is True
        assert r.error is None

    def test_failed_result(self):
        r = Result(
            env="math",
            score=0.0,
            latency_seconds=0.1,
            success=False,
            error="Timeout"
        )
        assert r.success is False
        assert r.error == "Timeout"

    def test_timestamp_auto_set(self):
        before = time.time()
        r = Result(env="test", score=0.5, latency_seconds=0.1, success=True)
        after = time.time()
        assert before <= r.timestamp <= after

    def test_extra_defaults_empty(self):
        r = Result(env="e", score=0.0, latency_seconds=0.0, success=True)
        assert r.extra == {}

    def test_dict_serialization(self):
        r = Result(env="env1", score=0.5, latency_seconds=1.0, success=True)
        d = r.dict()
        assert d["env"] == "env1"
        assert d["score"] == 0.5
        assert d["success"] is True

    def test_json_serialization(self):
        r = Result(env="env1", score=0.5, latency_seconds=1.0, success=True)
        j = r.json()
        parsed = json.loads(j)
        assert parsed["env"] == "env1"

    def test_repr(self):
        r = Result(
            env="coding",
            score=0.9876,
            latency_seconds=1.0,
            success=True,
            miner_hotkey="abcdef123456789"
        )
        s = repr(r)
        assert "Result" in s
        assert "coding" in s
        assert "0.9876" in s

    def test_str_same_as_repr(self):
        r = Result(env="e", score=0.0, latency_seconds=0.0, success=True)
        assert str(r) == repr(r)
