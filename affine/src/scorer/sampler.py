"""
Window sampler

Deterministically picks a fixed set of task_ids per (window_id, env) from each
env's dataset_range. Two modes:

- ``mode='latest'`` (SWE / DISTILL): take the highest N task_ids in
  ``dataset_range`` (the dataset's tail). These datasets append-only
  accumulate fresh tasks; only the tail is unseen.
- ``mode='random'`` (everything else): seed a local ``random.Random`` from
  ``(window_id, block_start, env)`` and sample N distinct task_ids
  weighted by range size. Idempotent across restarts.
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from affine.core.range_set import RangeSet


SAMPLING_MODE_LATEST = "latest"
SAMPLING_MODE_RANDOM = "random"


@dataclass(frozen=True)
class EnvSamplingConfig:
    env: str
    dataset_range: List[List[int]]
    sampling_count: int
    mode: str = SAMPLING_MODE_RANDOM


class WindowSampler:
    @staticmethod
    def _seed(window_id: int, block_start: int, env: str) -> int:
        digest = hashlib.sha256(
            f"{window_id}|{block_start}|{env}".encode()
        ).digest()[:16]
        return int.from_bytes(digest, "big")

    @staticmethod
    def _sample_one_env(
        ranges: List[Tuple[int, int]], n: int, rng: random.Random
    ) -> List[int]:
        total = sum(end - start for start, end in ranges)
        if n > total:
            raise ValueError(
                f"sampling_count={n} exceeds available IDs={total} in dataset_range"
            )
        if n == 0:
            return []

        weights = [end - start for start, end in ranges]
        picked: set[int] = set()

        # Weighted draw with rejection-on-duplicate; fall back to in-range
        # sequential fill if duplicates pile up (e.g. n very close to total).
        max_attempts = max(8, n // 16)
        attempts = 0
        while len(picked) < n and attempts < max_attempts:
            need = n - len(picked)
            batch = max(1, int(need * 1.3))
            for _ in range(batch):
                idx = rng.choices(range(len(ranges)), weights=weights, k=1)[0]
                start, end = ranges[idx]
                picked.add(rng.randrange(start, end))
                if len(picked) >= n:
                    break
            attempts += 1

        if len(picked) < n:
            # Exhaustive fill in deterministic range order. Within each range
            # we shuffle the remaining ids with the same rng so the tail of
            # the picked list is also seed-derived.
            for start, end in ranges:
                if len(picked) >= n:
                    break
                remaining = [i for i in range(start, end) if i not in picked]
                rng.shuffle(remaining)
                for v in remaining:
                    picked.add(v)
                    if len(picked) >= n:
                        break

        out = sorted(picked)
        return out[:n]

    @staticmethod
    def _latest_one_env(ranges: List[Tuple[int, int]], n: int) -> List[int]:
        """Return the largest ``n`` task_ids covered by ``ranges`` (sorted asc).

        ``ranges`` is the normalized list of half-open intervals from
        :class:`RangeSet`. We walk it from the back so we never materialize
        the whole interval — important for million-scale ranges.
        """
        total = sum(end - start for start, end in ranges)
        if n > total:
            raise ValueError(
                f"sampling_count={n} exceeds available IDs={total} in dataset_range"
            )
        if n == 0:
            return []
        out: List[int] = []
        remaining = n
        for start, end in reversed(ranges):
            seg_size = end - start
            take = min(remaining, seg_size)
            # Take the top ``take`` IDs from this interval (i.e. end-take..end-1).
            out.extend(range(end - take, end))
            remaining -= take
            if remaining <= 0:
                break
        return sorted(out)

    def generate(
        self,
        window_id: int,
        block_start: int,
        env_configs: Dict[str, EnvSamplingConfig],
    ) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for env, cfg in env_configs.items():
            ranges = RangeSet(cfg.dataset_range).ranges  # normalized + merged
            if cfg.mode == SAMPLING_MODE_LATEST:
                result[env] = self._latest_one_env(ranges, cfg.sampling_count)
            elif cfg.mode == SAMPLING_MODE_RANDOM:
                rng = random.Random(self._seed(window_id, block_start, env))
                result[env] = self._sample_one_env(ranges, cfg.sampling_count, rng)
            else:
                raise ValueError(
                    f"unknown sampling mode {cfg.mode!r} for env={env!r}; "
                    f"expected {SAMPLING_MODE_LATEST!r} or {SAMPLING_MODE_RANDOM!r}"
                )
        return result
