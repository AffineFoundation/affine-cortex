"""Unit tests for cold-time tracking and auto-termination.

Covers:
- MinerStatsDAO.update_cold_tracking — cumulative cold accumulation,
  delta cap, both termination paths (cold_too_long @ 36h sampled,
  never_sampled @ 48h on-chain), backfill from sample_results,
  clock-drift / negative-delta safety, idempotency once terminated.
- MinerStatsDAO._upsert_never_sampled_termination — record creation
  for miners with no prior stats row.
- SampleResultsDAO.get_latest_sample_timestamp_ms — max timestamp
  across env partitions, None when no samples exist.

DynamoDB is mocked at the client level so no real AWS calls occur.
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.sample_results import SampleResultsDAO


HK = "5HotkeyHotkeyHotkeyHotkeyHotkeyHotkey"
REV = "rev_abcdef0123456789"


def make_mock_client():
    """Build a mock DynamoDB client with the methods we exercise."""
    mock_client = MagicMock()
    mock_client.get_item = AsyncMock()
    mock_client.update_item = AsyncMock()
    mock_client.query = AsyncMock()

    class _CCFE(Exception):
        pass

    mock_client.exceptions = MagicMock()
    mock_client.exceptions.ConditionalCheckFailedException = _CCFE
    return mock_client


def patched_clients(mock_client):
    """Context-manager bundle that re-binds get_client at every import site
    update_cold_tracking touches: base_dao (via BaseDAO.get), the
    in-function import in miner_stats, and sample_results."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch(
        "affine.database.base_dao.get_client", return_value=mock_client))
    stack.enter_context(patch(
        "affine.database.client.get_client", return_value=mock_client))
    stack.enter_context(patch(
        "affine.database.dao.sample_results.get_client", return_value=mock_client))
    return stack


def make_dao_with_mock_client():
    """Returns (dao, mock_client) ready for use under `patched_clients`."""
    mock_client = make_mock_client()
    # init_client guards prevent constructing the DAO without a client;
    # patch the client check during DAO construction too.
    with patched_clients(mock_client):
        dao = MinerStatsDAO()
    return dao, mock_client


def stats_row(**overrides):
    """Build an existing miner_stats row (already deserialized dict)."""
    base = {
        "pk": f"HOTKEY#{HK}",
        "sk": f"REV#{REV}",
        "hotkey": HK,
        "revision": REV,
        "model": "test/model",
        "challenge_status": "sampling",
        "termination_reason": "",
    }
    base.update(overrides)
    return base


def serialize_get_response(row):
    """Convert a plain dict into a DynamoDB get_item response shape
    that BaseDAO._deserialize understands."""
    if row is None:
        return {}
    item = {}
    for k, v in row.items():
        if isinstance(v, bool):
            item[k] = {"BOOL": v}
        elif isinstance(v, int):
            item[k] = {"N": str(v)}
        elif isinstance(v, float):
            item[k] = {"N": str(v)}
        elif isinstance(v, str):
            item[k] = {"S": v}
        elif isinstance(v, dict):
            item[k] = {"M": {}}
        elif v is None:
            item[k] = {"NULL": True}
        else:
            item[k] = {"S": str(v)}
    return {"Item": item}


# ─────────────────────────────────────────────────────────────────────────
# update_cold_tracking — basic / no-op paths
# ─────────────────────────────────────────────────────────────────────────


class TestUpdateColdTrackingNoOps:

    @pytest.mark.asyncio
    async def test_no_record_no_chain_age_returns_none(self):
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = {}  # missing record

        with patched_clients(client):
            result = await dao.update_cold_tracking(HK, REV, is_cold=True)

        assert result is None
        client.update_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_record_chain_age_below_threshold_skip(self):
        """No record + chain age < 48h → skip (not yet eligible)."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = {}

        with patched_clients(client):
            result = await dao.update_cold_tracking(
                HK, REV, is_cold=True,
                chain_age_seconds=(48 * 3600) - 1,
            )

        assert result is None
        client.update_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_terminated_returns_none(self):
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row(
            challenge_status="terminated",
            termination_reason="champion_loss",
            cold_seconds_total=10,
        ))

        with patched_clients(client):
            result = await dao.update_cold_tracking(HK, REV, is_cold=True)

        assert result is None
        client.update_item.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# update_cold_tracking — first-call paths
# ─────────────────────────────────────────────────────────────────────────


class TestFirstCall:

    @pytest.mark.asyncio
    async def test_first_call_hot_sets_has_been_hot(self):
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row())

        with patched_clients(client):
            r = await dao.update_cold_tracking(HK, REV, is_cold=False)

        assert r["has_been_hot"] is True
        assert r["cold_seconds_total"] == 0
        assert r["challenge_status"] == "sampling"
        client.update_item.assert_called_once()

    @pytest.mark.asyncio
    async def test_first_call_cold_no_backfill_no_credit(self):
        """No prior tracking, cold, no backfill, never sampled →
        cold time stays 0 and has_been_hot stays False."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row())

        with patched_clients(client):
            r = await dao.update_cold_tracking(HK, REV, is_cold=True)

        assert r["cold_seconds_total"] == 0
        assert r["has_been_hot"] is False
        assert r["challenge_status"] == "sampling"

    @pytest.mark.asyncio
    async def test_first_call_cold_with_backfill_3d(self):
        """Backfill from a 3-day-old sample → terminated for cold_too_long
        (3d > 36h)."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row())
        now_ms = int(time.time() * 1000)
        three_days_ago_ms = now_ms - (3 * 86400 * 1000)

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=True,
                backfill_last_seen_ms=three_days_ago_ms,
            )

        assert r["has_been_hot"] is True
        # ~3d = 259200s; allow ±5s slop for time.time() drift between calls
        assert abs(r["cold_seconds_total"] - 3 * 86400) < 5
        assert r["challenge_status"] == "terminated"
        assert r["termination_reason"] == "cold_too_long"

    @pytest.mark.asyncio
    async def test_first_call_cold_backfill_under_threshold(self):
        """Backfill 10h ago → has_been_hot=True, cold credited but under 36h."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row())
        ten_hours_ago_ms = int(time.time() * 1000) - (10 * 3600 * 1000)

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=True,
                backfill_last_seen_ms=ten_hours_ago_ms,
            )

        assert r["has_been_hot"] is True
        assert abs(r["cold_seconds_total"] - 10 * 3600) < 5
        assert r["challenge_status"] == "sampling"


# ─────────────────────────────────────────────────────────────────────────
# update_cold_tracking — incremental updates
# ─────────────────────────────────────────────────────────────────────────


class TestIncremental:

    @pytest.mark.asyncio
    async def test_subsequent_cold_within_cap(self):
        """5min after last check, cold → cold_total += 300s (within cap)."""
        dao, client = make_dao_with_mock_client()
        now = int(time.time())
        client.get_item.return_value = serialize_get_response(stats_row(
            cold_seconds_total=1000,
            has_been_hot=True,
            last_status_check_at=now - 300,
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(HK, REV, is_cold=True)

        # delta should be ~300; total ≈ 1300
        assert 1290 <= r["cold_seconds_total"] <= 1310
        assert r["has_been_hot"] is True

    @pytest.mark.asyncio
    async def test_subsequent_cold_over_cap_clamped(self):
        """If last check was 2h ago, only 600s (cap) credited."""
        dao, client = make_dao_with_mock_client()
        now = int(time.time())
        client.get_item.return_value = serialize_get_response(stats_row(
            cold_seconds_total=0,
            has_been_hot=True,
            last_status_check_at=now - 7200,
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(HK, REV, is_cold=True)

        assert r["cold_seconds_total"] == 600  # capped

    @pytest.mark.asyncio
    async def test_subsequent_hot_does_not_grow(self):
        dao, client = make_dao_with_mock_client()
        now = int(time.time())
        client.get_item.return_value = serialize_get_response(stats_row(
            cold_seconds_total=5000,
            has_been_hot=True,
            last_status_check_at=now - 300,
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(HK, REV, is_cold=False)

        assert r["cold_seconds_total"] == 5000  # unchanged
        assert r["has_been_hot"] is True

    @pytest.mark.asyncio
    async def test_negative_delta_clamped(self):
        """Clock drift: last_status_check_at in the future → delta=0."""
        dao, client = make_dao_with_mock_client()
        now = int(time.time())
        client.get_item.return_value = serialize_get_response(stats_row(
            cold_seconds_total=100,
            has_been_hot=True,
            last_status_check_at=now + 10_000,  # future
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(HK, REV, is_cold=True)

        assert r["cold_seconds_total"] == 100  # unchanged

    @pytest.mark.asyncio
    async def test_threshold_crossing_terminates(self):
        dao, client = make_dao_with_mock_client()
        now = int(time.time())
        # Just below threshold; one more 300s tick should push over.
        client.get_item.return_value = serialize_get_response(stats_row(
            cold_seconds_total=MinerStatsDAO.COLD_TERMINATION_THRESHOLD_SECONDS - 100,
            has_been_hot=True,
            last_status_check_at=now - 300,
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(HK, REV, is_cold=True)

        assert r["challenge_status"] == "terminated"
        assert r["termination_reason"] == "cold_too_long"
        assert r["cold_seconds_total"] >= MinerStatsDAO.COLD_TERMINATION_THRESHOLD_SECONDS


# ─────────────────────────────────────────────────────────────────────────
# update_cold_tracking — never-sampled chain-age path
# ─────────────────────────────────────────────────────────────────────────


class TestNeverSampledTermination:

    @pytest.mark.asyncio
    async def test_no_record_chain_age_over_48h_terminates(self):
        """No miner_stats row + chain age ≥ 48h → upsert terminated/never_sampled."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = {}  # no record

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=True,
                chain_age_seconds=48 * 3600 + 1,
            )

        assert r is not None
        assert r["challenge_status"] == "terminated"
        assert r["termination_reason"] == "never_sampled"
        client.update_item.assert_called_once()

    @pytest.mark.asyncio
    async def test_existing_record_never_hot_chain_age_over_48h(self):
        """Stub record exists (e.g., from update_challenge_state init) but
        miner has never been hot, and chain age > 48h → terminate."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row(
            has_been_hot=False,
            cold_seconds_total=0,
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=True,
                chain_age_seconds=48 * 3600 + 100,
            )

        assert r["challenge_status"] == "terminated"
        assert r["termination_reason"] == "never_sampled"

    @pytest.mark.asyncio
    async def test_chain_age_below_48h_no_termination(self):
        """Stub record, never hot, chain age < 48h → no termination."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row(
            has_been_hot=False,
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=True,
                chain_age_seconds=24 * 3600,
            )

        assert r["challenge_status"] == "sampling"

    @pytest.mark.asyncio
    async def test_has_been_hot_uses_36h_not_48h(self):
        """Once a miner has been hot, the never_sampled clock no longer
        applies. Cold path uses the 36h threshold."""
        dao, client = make_dao_with_mock_client()
        now = int(time.time())
        # cold_total just over 36h
        client.get_item.return_value = serialize_get_response(stats_row(
            has_been_hot=True,
            cold_seconds_total=MinerStatsDAO.COLD_TERMINATION_THRESHOLD_SECONDS + 50,
            last_status_check_at=now - 30,
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=True,
                chain_age_seconds=24 * 3600,  # under 48h, irrelevant
            )

        assert r["termination_reason"] == "cold_too_long"

    @pytest.mark.asyncio
    async def test_no_record_hot_miner_chain_age_over_48h_NOT_terminated(self):
        """Safety: a HOT miner with no stats row yet — even on-chain
        for 48h+ — must not be terminated. This guards against
        validator-restart-after-downtime scenarios where good miners
        haven't been picked up for sampling yet."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = {}  # no record

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=False,           # HOT
                chain_age_seconds=72 * 3600,
            )

        assert r is None
        client.update_item.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# Hot miner is never wrongly terminated
# ─────────────────────────────────────────────────────────────────────────


class TestHotMinerSafety:

    @pytest.mark.asyncio
    async def test_existing_record_hot_chain_age_over_48h_not_terminated(self):
        """An existing-record hot miner over 48h chain age must not
        terminate. has_been_hot flips True via the is_cold=False
        branch so the never_sampled condition is False."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row(
            has_been_hot=False,  # was never observed hot before
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=False,           # currently HOT now
                chain_age_seconds=72 * 3600,
            )

        assert r["challenge_status"] == "sampling"
        assert r["has_been_hot"] is True

    @pytest.mark.asyncio
    async def test_already_terminated_hot_miner_no_op(self):
        """Even if challenge_status was set to terminated externally
        (e.g., champion_loss), and miner is now hot, we don't touch
        the row."""
        dao, client = make_dao_with_mock_client()
        client.get_item.return_value = serialize_get_response(stats_row(
            challenge_status="terminated",
            termination_reason="champion_loss",
            has_been_hot=True,
        ))

        with patched_clients(client):
            r = await dao.update_cold_tracking(
                HK, REV, is_cold=False,
                chain_age_seconds=72 * 3600,
            )

        assert r is None
        client.update_item.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# Threshold constants
# ─────────────────────────────────────────────────────────────────────────


class TestThresholdConstants:

    def test_cold_threshold_is_36h(self):
        assert MinerStatsDAO.COLD_TERMINATION_THRESHOLD_SECONDS == 36 * 3600

    def test_never_sampled_threshold_is_48h(self):
        assert MinerStatsDAO.NEVER_SAMPLED_TERMINATION_THRESHOLD_SECONDS == 48 * 3600

    def test_delta_cap_is_600s(self):
        assert MinerStatsDAO.COLD_TRACKING_DELTA_CAP_SECONDS == 600


# ─────────────────────────────────────────────────────────────────────────
# get_latest_sample_timestamp_ms
# ─────────────────────────────────────────────────────────────────────────


class TestGetLatestSampleTimestamp:

    @pytest.mark.asyncio
    async def test_no_envs_returns_none(self):
        dao = SampleResultsDAO()
        client = MagicMock()
        client.query = AsyncMock(return_value={"Items": []})

        with patch("affine.database.dao.sample_results.get_client", return_value=client):
            ts = await dao.get_latest_sample_timestamp_ms(HK, REV, [])

        assert ts is None
        client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_samples_returns_none(self):
        dao = SampleResultsDAO()
        client = MagicMock()
        client.query = AsyncMock(return_value={"Items": []})

        with patch("affine.database.dao.sample_results.get_client", return_value=client):
            ts = await dao.get_latest_sample_timestamp_ms(HK, REV, ["GAME"])

        assert ts is None

    @pytest.mark.asyncio
    async def test_single_env_max_timestamp(self):
        dao = SampleResultsDAO()
        client = MagicMock()
        client.query = AsyncMock(return_value={
            "Items": [
                {"timestamp": {"N": "1700000000000"}},
                {"timestamp": {"N": "1700001000000"}},  # max
                {"timestamp": {"N": "1699999000000"}},
            ]
        })

        with patch("affine.database.dao.sample_results.get_client", return_value=client):
            ts = await dao.get_latest_sample_timestamp_ms(HK, REV, ["GAME"])

        assert ts == 1700001000000

    @pytest.mark.asyncio
    async def test_multi_env_max_across_partitions(self):
        dao = SampleResultsDAO()
        # Each call returns different max timestamps; we want overall max.
        responses = [
            {"Items": [{"timestamp": {"N": "1000"}}]},
            {"Items": [{"timestamp": {"N": "5000"}}]},  # max
            {"Items": [{"timestamp": {"N": "3000"}}]},
        ]
        client = MagicMock()
        client.query = AsyncMock(side_effect=responses)

        with patch("affine.database.dao.sample_results.get_client", return_value=client):
            ts = await dao.get_latest_sample_timestamp_ms(
                HK, REV, ["GAME", "MEMORY", "DISTILL"])

        assert ts == 5000
        assert client.query.call_count == 3

    @pytest.mark.asyncio
    async def test_skip_malformed_timestamp(self):
        """Items with non-numeric or missing timestamp are skipped."""
        dao = SampleResultsDAO()
        client = MagicMock()
        client.query = AsyncMock(return_value={
            "Items": [
                {},  # missing timestamp
                {"timestamp": {"S": "not-a-number"}},  # wrong type
                {"timestamp": {"N": "12345"}},
            ]
        })

        with patch("affine.database.dao.sample_results.get_client", return_value=client):
            ts = await dao.get_latest_sample_timestamp_ms(HK, REV, ["GAME"])

        assert ts == 12345
