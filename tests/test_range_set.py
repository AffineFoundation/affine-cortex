"""Tests for RangeSet - interval-based set operations."""

import pytest
import random
from affine.core.range_set import RangeSet


class TestRangeSetInit:
    """Test RangeSet initialization and normalization."""

    def test_empty_ranges(self):
        rs = RangeSet([])
        assert rs.size() == 0
        assert rs.ranges == []

    def test_single_range(self):
        rs = RangeSet([[0, 10]])
        assert rs.size() == 10
        assert rs.ranges == [(0, 10)]

    def test_zero_width_ranges_filtered(self):
        rs = RangeSet([[5, 5], [3, 3], [1, 2]])
        assert rs.size() == 1
        assert rs.ranges == [(1, 2)]

    def test_overlapping_ranges_merged(self):
        rs = RangeSet([[0, 5], [3, 8]])
        assert rs.size() == 8
        assert rs.ranges == [(0, 8)]

    def test_adjacent_ranges_merged(self):
        rs = RangeSet([[0, 5], [5, 10]])
        assert rs.size() == 10
        assert rs.ranges == [(0, 10)]

    def test_disjoint_ranges_preserved(self):
        rs = RangeSet([[0, 3], [10, 15]])
        assert rs.size() == 8
        assert rs.ranges == [(0, 3), (10, 15)]

    def test_unsorted_ranges_sorted(self):
        rs = RangeSet([[10, 20], [0, 5]])
        assert rs.ranges == [(0, 5), (10, 20)]

    def test_multiple_overlapping_ranges(self):
        rs = RangeSet([[0, 5], [3, 8], [7, 12], [20, 25]])
        assert rs.ranges == [(0, 12), (20, 25)]
        assert rs.size() == 17

    def test_negative_width_range_filtered(self):
        rs = RangeSet([[10, 5]])
        assert rs.size() == 0


class TestRangeSetFromIds:
    """Test the from_ids classmethod."""

    def test_empty_ids(self):
        rs = RangeSet.from_ids([])
        assert rs.size() == 0

    def test_single_id(self):
        rs = RangeSet.from_ids([42])
        assert rs.size() == 1
        assert rs.ranges == [(42, 43)]

    def test_consecutive_ids(self):
        rs = RangeSet.from_ids([1, 2, 3, 4, 5])
        assert rs.ranges == [(1, 6)]

    def test_gaps_in_ids(self):
        rs = RangeSet.from_ids([1, 2, 3, 7, 8, 10])
        assert rs.ranges == [(1, 4), (7, 9), (10, 11)]
        assert rs.size() == 6

    def test_duplicate_ids(self):
        rs = RangeSet.from_ids([1, 1, 2, 2, 3])
        assert rs.ranges == [(1, 4)]
        assert rs.size() == 3

    def test_unsorted_ids(self):
        rs = RangeSet.from_ids([5, 1, 3, 2, 4])
        assert rs.ranges == [(1, 6)]


class TestRangeSetContains:
    """Test __contains__ (in operator)."""

    def test_contains_in_range(self):
        rs = RangeSet([[10, 20]])
        assert 10 in rs
        assert 15 in rs
        assert 19 in rs

    def test_not_contains_at_end(self):
        rs = RangeSet([[10, 20]])
        assert 20 not in rs  # end is exclusive

    def test_not_contains_before_start(self):
        rs = RangeSet([[10, 20]])
        assert 9 not in rs

    def test_contains_multiple_ranges(self):
        rs = RangeSet([[0, 5], [10, 15]])
        assert 0 in rs
        assert 4 in rs
        assert 5 not in rs
        assert 7 not in rs
        assert 10 in rs
        assert 14 in rs

    def test_contains_empty(self):
        rs = RangeSet([])
        assert 0 not in rs


class TestRangeSetLen:
    """Test __len__."""

    def test_len(self):
        rs = RangeSet([[0, 100], [200, 300]])
        assert len(rs) == 200


class TestRangeSetEquality:
    """Test __eq__."""

    def test_equal(self):
        assert RangeSet([[0, 10]]) == RangeSet([[0, 10]])

    def test_not_equal(self):
        assert RangeSet([[0, 10]]) != RangeSet([[0, 11]])

    def test_overlapping_normalized_equal(self):
        assert RangeSet([[0, 5], [3, 10]]) == RangeSet([[0, 10]])

    def test_not_equal_other_type(self):
        assert RangeSet([[0, 10]]) != "not a rangeset"


class TestRangeSetSubtractIds:
    """Test subtract_ids."""

    def test_subtract_empty(self):
        rs = RangeSet([[0, 10]])
        result = rs.subtract_ids(set())
        assert result.size() == 10

    def test_subtract_single_id_middle(self):
        rs = RangeSet([[0, 10]])
        result = rs.subtract_ids({5})
        assert result.size() == 9
        assert 5 not in result
        assert 4 in result
        assert 6 in result

    def test_subtract_first_id(self):
        rs = RangeSet([[0, 5]])
        result = rs.subtract_ids({0})
        assert result.size() == 4
        assert 0 not in result

    def test_subtract_last_id(self):
        rs = RangeSet([[0, 5]])
        result = rs.subtract_ids({4})
        assert result.size() == 4
        assert 4 not in result

    def test_subtract_all_ids(self):
        rs = RangeSet([[0, 3]])
        result = rs.subtract_ids({0, 1, 2})
        assert result.size() == 0

    def test_subtract_ids_outside_range(self):
        rs = RangeSet([[0, 5]])
        result = rs.subtract_ids({10, 20, 30})
        assert result.size() == 5

    def test_subtract_from_multiple_ranges(self):
        rs = RangeSet([[0, 5], [10, 15]])
        result = rs.subtract_ids({2, 12})
        assert result.size() == 8
        assert 2 not in result
        assert 12 not in result


class TestRangeSetRandomSample:
    """Test random_sample."""

    def test_sample_zero(self):
        rs = RangeSet([[0, 100]])
        assert rs.random_sample(0) == []

    def test_sample_exceeds_size(self):
        rs = RangeSet([[0, 5]])
        with pytest.raises(ValueError, match="Cannot sample"):
            rs.random_sample(10)

    def test_sample_all(self):
        rs = RangeSet([[0, 5]])
        samples = rs.random_sample(5)
        assert sorted(samples) == [0, 1, 2, 3, 4]

    def test_sample_within_range(self):
        rs = RangeSet([[10, 20]])
        samples = rs.random_sample(5)
        assert len(samples) == 5
        assert all(10 <= s < 20 for s in samples)
        assert len(set(samples)) == 5  # unique

    def test_sample_multiple_ranges(self):
        rs = RangeSet([[0, 5], [100, 105]])
        samples = rs.random_sample(8)
        assert len(samples) == 8
        for s in samples:
            assert (0 <= s < 5) or (100 <= s < 105)


class TestRangeSetPrioritizedSample:
    """Test prioritized_sample."""

    def test_prioritized_sample_zero(self):
        rs = RangeSet([[0, 100]])
        assert rs.prioritized_sample(0) == []

    def test_prioritized_sample_exceeds_size(self):
        rs = RangeSet([[0, 5]])
        with pytest.raises(ValueError, match="Cannot sample"):
            rs.prioritized_sample(10)

    def test_prioritized_favors_later_ranges(self):
        """When sampling fewer than a later range, all should come from later range."""
        rs = RangeSet([[0, 100], [1000, 1005]])
        # Sample 5 — should all come from the later range [1000, 1005)
        samples = rs.prioritized_sample(5)
        assert len(samples) == 5
        assert all(1000 <= s < 1005 for s in samples)

    def test_prioritized_sample_all(self):
        rs = RangeSet([[0, 3], [10, 13]])
        samples = rs.prioritized_sample(6)
        assert sorted(samples) == [0, 1, 2, 10, 11, 12]


class TestRangeSetIntersection:
    """Test intersection."""

    def test_no_overlap(self):
        a = RangeSet([[0, 5]])
        b = RangeSet([[10, 15]])
        assert a.intersection(b).size() == 0

    def test_full_overlap(self):
        a = RangeSet([[0, 10]])
        b = RangeSet([[0, 10]])
        assert a.intersection(b) == RangeSet([[0, 10]])

    def test_partial_overlap(self):
        a = RangeSet([[0, 10]])
        b = RangeSet([[5, 15]])
        result = a.intersection(b)
        assert result == RangeSet([[5, 10]])

    def test_subset(self):
        a = RangeSet([[0, 20]])
        b = RangeSet([[5, 10]])
        assert a.intersection(b) == RangeSet([[5, 10]])

    def test_multiple_intersections(self):
        a = RangeSet([[0, 5], [10, 15]])
        b = RangeSet([[3, 12]])
        result = a.intersection(b)
        assert result == RangeSet([[3, 5], [10, 12]])


class TestRangeSetUnion:
    """Test union."""

    def test_disjoint_union(self):
        a = RangeSet([[0, 5]])
        b = RangeSet([[10, 15]])
        result = a.union(b)
        assert result.size() == 10

    def test_overlapping_union(self):
        a = RangeSet([[0, 10]])
        b = RangeSet([[5, 15]])
        result = a.union(b)
        assert result == RangeSet([[0, 15]])


class TestRangeSetToList:
    """Test to_list serialization."""

    def test_round_trip(self):
        original = [[0, 5], [10, 15], [20, 25]]
        rs = RangeSet(original)
        assert rs.to_list() == original

    def test_repr(self):
        rs = RangeSet([[0, 5]])
        assert "RangeSet" in repr(rs)
        assert "size=5" in repr(rs)
