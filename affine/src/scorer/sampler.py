"""
Window sampler

Picks a fixed set of task_ids from each env's dataset_range. Three modes:

- ``mode='latest'`` (SWE / DISTILL): take the highest N task_ids in
  ``dataset_range`` (the dataset's tail). These datasets append-only
  accumulate fresh tasks; only the tail is unseen.
- ``mode='random'`` (everything else): use OS-backed randomness and sample
  N distinct task_ids weighted by range size. The generated pool is persisted
  by the scheduler, so restarts reuse the stored task ids instead of
  re-sampling.
- ``mode='template_stratified_v1'`` (InstructionGym): use a pinned release
  manifest to balance templates, then sample assignments within each selected
  template. The resulting values remain direct global task IDs.
"""

import random
import secrets
from dataclasses import dataclass
from typing import Dict, List, Tuple

from affine.core.instruction_gym_sampling import load_sampling_manifest, sample_task_ids
from affine.core.range_set import RangeSet


SAMPLING_MODE_LATEST = "latest"
SAMPLING_MODE_RANDOM = "random"
SAMPLING_MODE_TEMPLATE_STRATIFIED_V1 = "template_stratified_v1"


@dataclass(frozen=True)
class EnvSamplingConfig:
    env: str
    dataset_range: List[List[int]]
    sampling_count: int
    mode: str = SAMPLING_MODE_RANDOM
    sampling_manifest_sha256: str = ""


class WindowSampler:
    def _rng(self) -> random.Random:
        return secrets.SystemRandom()

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
            # we shuffle the remaining ids with the same rng so the tail is
            # still random-derived.
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
                rng = self._rng()
                result[env] = self._sample_one_env(ranges, cfg.sampling_count, rng)
            elif cfg.mode == SAMPLING_MODE_TEMPLATE_STRATIFIED_V1:
                manifest = load_sampling_manifest(cfg.sampling_manifest_sha256)
                from affine.core.environments import (
                    INSTRUCTION_GYM_TASK_ID_END,
                    INSTRUCTION_GYM_UNIVERSE_ID,
                )

                if (
                    manifest.universe_id != INSTRUCTION_GYM_UNIVERSE_ID
                    or manifest.case_id_end != INSTRUCTION_GYM_TASK_ID_END
                ):
                    raise ValueError(
                        "InstructionGym sampling manifest does not match the Actor contract"
                    )
                expected_ranges = [(0, manifest.case_id_end)]
                if ranges != expected_ranges:
                    raise ValueError(
                        f"dataset_range for env={env!r} must exactly match the "
                        f"InstructionGym manifest range {expected_ranges}"
                    )
                result[env] = sample_task_ids(
                    manifest,
                    cfg.sampling_count,
                    self._rng(),
                )
            else:
                raise ValueError(
                    f"unknown sampling mode {cfg.mode!r} for env={env!r}; "
                    f"expected {SAMPLING_MODE_LATEST!r}, {SAMPLING_MODE_RANDOM!r}, "
                    f"or {SAMPLING_MODE_TEMPLATE_STRATIFIED_V1!r}"
                )
        return result
