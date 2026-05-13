"""
Interval-based set normalization.

A miner dataset is described as a list of half-open intervals — possibly
overlapping, possibly redundantly ordered. :class:`RangeSet` merges and
sorts them so :class:`affine.src.scorer.sampler.WindowSampler` only has
to think about a clean, non-overlapping list.

This module intentionally has no sampling logic — that lives in
``WindowSampler`` with a seeded ``random.Random`` so the same
``(window_id, block_start, env)`` always produces the same list.
"""

from __future__ import annotations

from typing import List, Tuple


class RangeSet:
    """Normalize a list of ``[start, end)`` intervals into a sorted,
    non-overlapping ``list[(start, end)]`` available on :attr:`ranges`."""

    __slots__ = ("ranges",)

    def __init__(self, ranges: List[List[int]]):
        self.ranges: List[Tuple[int, int]] = self._normalize(ranges)

    @staticmethod
    def _normalize(ranges: List[List[int]]) -> List[Tuple[int, int]]:
        intervals = sorted((r[0], r[1]) for r in ranges if r[1] > r[0])
        if not intervals:
            return []
        merged: List[Tuple[int, int]] = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
