"""
Weight writer.

When the flow scheduler decides a contest, it calls :meth:`WeightWriter.write` to persist
the winning champion at weight=1.0 and every other valid miner at
weight=0.0. Validator reads ``/scores/weights/latest`` to put weights
on-chain — that endpoint is unchanged.

Also writes a ``score_snapshots`` row capturing the window's metadata.
This is the only place that writes ``scores`` / ``score_snapshots`` in
the new flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class WeightSubject:
    """Identifies one miner that should be written into the ``scores`` table.

    The full set of valid miners (champion + all losers + everyone-else)
    is supplied by the scheduler. Champion gets ``overall_score=1.0``;
    everyone else 0.0. ``scores_by_env`` is optional metadata (per-env
    score breakdown) that ranking UIs surface.
    """

    uid: int
    hotkey: str
    revision: str
    model: str
    first_block: int
    is_champion: bool
    scores_by_env: Dict[str, Any]
    total_samples: int


class ScoresWriter(Protocol):
    async def save_score(
        self,
        block_number: int,
        miner_hotkey: str,
        uid: int,
        model_revision: str,
        model: str,
        first_block: int,
        overall_score: float,
        average_score: float,
        scores_by_env: Dict[str, Any],
        total_samples: int,
    ) -> Dict[str, Any]: ...


class SnapshotWriter(Protocol):
    async def save_snapshot(
        self,
        block_number: int,
        scorer_hotkey: str,
        config: Dict[str, Any],
        statistics: Dict[str, Any],
        ttl_days: int = 30,
    ) -> Dict[str, Any]: ...


class WeightWriter:
    """Persists the per-window scoring outcome.

    All side effects (DAO writes) go through injected ``ScoresWriter`` and
    ``SnapshotWriter`` so unit tests can substitute in-memory stand-ins.
    Production wires the DynamoDB DAOs directly.
    """

    def __init__(self, scores: ScoresWriter, snapshots: SnapshotWriter):
        self._scores = scores
        self._snapshots = snapshots

    async def write(
        self,
        *,
        window_id: int,
        block_number: int,
        scorer_hotkey: str,
        envs: List[str],
        subjects: List[WeightSubject],
        outcome: Dict[str, Any],
    ) -> None:
        """Write one ``scores`` row per subject + one ``score_snapshots`` row.

        Args:
            window_id: Logical window identifier (recorded in snapshot config).
            block_number: Block at which the snapshot is taken.
            scorer_hotkey: Identity persisted into snapshot.
            envs: Environments evaluated in this window.
            subjects: Miners to score. Exactly one must be ``is_champion``.
            outcome: ``WindowComparator`` outcome payload (winner, reason, per_env).
        """
        champions = [s for s in subjects if s.is_champion]
        if len(champions) != 1:
            raise ValueError(
                f"WeightWriter.write expects exactly one champion, got {len(champions)}"
            )

        for subject in subjects:
            overall = 1.0 if subject.is_champion else 0.0
            average = _average_of_env_scores(subject.scores_by_env)
            await self._scores.save_score(
                block_number=block_number,
                miner_hotkey=subject.hotkey,
                uid=subject.uid,
                model_revision=subject.revision,
                model=subject.model,
                first_block=subject.first_block,
                overall_score=overall,
                average_score=average,
                scores_by_env=subject.scores_by_env,
                total_samples=subject.total_samples,
            )

        statistics = {
            "total_miners": len(subjects),
            "winner_uid": champions[0].uid,
            "winner_hotkey": champions[0].hotkey,
            "final_weights": {
                str(s.uid): "1.0" if s.is_champion else "0.0" for s in subjects
            },
        }
        await self._snapshots.save_snapshot(
            block_number=block_number,
            scorer_hotkey=scorer_hotkey,
            config={
                "window_id": window_id,
                "environments": list(envs),
                "outcome": outcome,
            },
            statistics=statistics,
        )


def _average_of_env_scores(scores_by_env: Dict[str, Any]) -> float:
    """Best-effort average of per-env mean scores; ``0.0`` if no numbers found."""
    means: list[float] = []
    for env_payload in scores_by_env.values():
        if isinstance(env_payload, dict):
            for key in ("score", "mean", "avg", "average"):
                if key in env_payload and isinstance(env_payload[key], (int, float)):
                    means.append(float(env_payload[key]))
                    break
        elif isinstance(env_payload, (int, float)):
            means.append(float(env_payload))
    if not means:
        return 0.0
    return sum(means) / len(means)
