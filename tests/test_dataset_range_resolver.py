import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from affine.core.dataset_range_resolver import (
    _build_range,
    _extract_field,
    expand_dataset_range,
    resolve_dataset_range_source,
)
from affine.core.range_set import RangeSet


class TestBuildRange:
    """Verify _build_range produces correct exclusive-end intervals for RangeSet."""

    def test_zero_to_value_covers_all_ids(self):
        r = _build_range(5, "zero_to_value")
        assert r == [[0, 5]]
        assert RangeSet(r).size() == 5

    def test_zero_to_value_single_id(self):
        r = _build_range(1, "zero_to_value")
        assert r == [[0, 1]]
        assert RangeSet(r).size() == 1

    def test_zero_to_value_zero(self):
        r = _build_range(0, "zero_to_value")
        assert r == [[0, 0]]
        assert RangeSet(r).size() == 0

    def test_zero_to_value_negative(self):
        r = _build_range(-3, "zero_to_value")
        assert r == [[0, 0]]
        assert RangeSet(r).size() == 0

    def test_unknown_range_type_raises(self):
        with pytest.raises(ValueError, match="Unknown range_type"):
            _build_range(5, "unknown")


class TestExpandDatasetRange:
    """Verify expand_dataset_range uses exclusive-end consistently."""

    def test_expand_covers_all_ids(self):
        expanded = expand_dataset_range([[0, 5]], 10)
        assert RangeSet(expanded).size() == 10

    def test_expand_from_empty(self):
        expanded = expand_dataset_range([], 5)
        assert expanded == [[0, 5]]
        assert RangeSet(expanded).size() == 5

    def test_no_expansion_when_value_shrinks(self):
        result = expand_dataset_range([[0, 10]], 5)
        assert result is None

    def test_no_expansion_when_value_unchanged(self):
        result = expand_dataset_range([[0, 10]], 10)
        assert result is None

    def test_large_tail_creates_new_segment(self):
        expanded = expand_dataset_range([[0, 200]], 300)
        assert expanded == [[0, 200], [200, 300]]
        assert RangeSet(expanded).size() == 300

    def test_expand_zero_value_returns_none(self):
        assert expand_dataset_range([[0, 5]], 0) is None

    def test_unknown_range_type_raises(self):
        with pytest.raises(ValueError, match="Unknown range_type"):
            expand_dataset_range([[0, 5]], 10, range_type="unknown")


class TestExtractField:
    def test_nested_dot_path(self):
        data = {"tasks": {"completed_up_to": 42}}
        assert _extract_field(data, "tasks.completed_up_to") == 42

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            _extract_field({}, "missing.path")


class TestResolveDatasetRangeSource:
    """Integration test for the async resolver."""

    @pytest.mark.asyncio
    async def test_fresh_build(self):
        source = {
            "url": "https://example.com/meta.json",
            "field": "count",
            "range_type": "zero_to_value",
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"count": 100})

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_resp

        mock_session = MagicMock()
        mock_session.get.return_value = mock_session_ctx

        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__.return_value = mock_session

        with patch("affine.core.dataset_range_resolver.aiohttp.ClientSession", return_value=mock_client_ctx):
            result = await resolve_dataset_range_source(source)

        assert result == [[0, 100]]
        assert RangeSet(result).size() == 100
