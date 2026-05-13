"""Unit tests for WeightWriter."""

import pytest

from affine.src.scorer.weight_writer import WeightSubject, WeightWriter


class _FakeScores:
    def __init__(self):
        self.rows: list[dict] = []

    async def save_score(self, **kwargs):
        self.rows.append(kwargs)
        return kwargs


class _FakeSnapshots:
    def __init__(self):
        self.rows: list[dict] = []

    async def save_snapshot(self, **kwargs):
        self.rows.append(kwargs)
        return kwargs


def _subject(uid, hotkey, *, is_champion, scores_by_env=None, samples=200):
    return WeightSubject(
        uid=uid,
        hotkey=hotkey,
        revision=f"rev{uid}",
        model=f"org/m{uid}",
        first_block=100 * uid,
        is_champion=is_champion,
        scores_by_env=scores_by_env or {},
        total_samples=samples,
    )


@pytest.mark.asyncio
async def test_champion_gets_weight_1_others_0():
    scores, snaps = _FakeScores(), _FakeSnapshots()
    w = WeightWriter(scores, snaps)
    await w.write(
        window_id=10,
        block_number=72000,
        scorer_hotkey="5Scorer",
        envs=["A", "B"],
        subjects=[
            _subject(1, "champ", is_champion=True),
            _subject(2, "chal", is_champion=False),
            _subject(3, "obs", is_champion=False),
        ],
        outcome={"winner": "champion", "reason": "no_challenger"},
    )
    weights = {r["uid"]: r["overall_score"] for r in scores.rows}
    assert weights == {1: 1.0, 2: 0.0, 3: 0.0}


@pytest.mark.asyncio
async def test_snapshot_records_window_id_and_envs():
    scores, snaps = _FakeScores(), _FakeSnapshots()
    w = WeightWriter(scores, snaps)
    await w.write(
        window_id=42,
        block_number=72000,
        scorer_hotkey="5Scorer",
        envs=["X", "Y", "Z"],
        subjects=[_subject(1, "champ", is_champion=True)],
        outcome={"winner": "challenger", "reason": "all_envs_better"},
    )
    assert len(snaps.rows) == 1
    cfg = snaps.rows[0]["config"]
    assert cfg["window_id"] == 42
    assert sorted(cfg["environments"]) == ["X", "Y", "Z"]
    assert cfg["outcome"]["reason"] == "all_envs_better"
    assert snaps.rows[0]["statistics"]["winner_uid"] == 1
    assert snaps.rows[0]["statistics"]["final_weights"] == {"1": "1.0"}


@pytest.mark.asyncio
async def test_average_score_from_env_payload_dicts():
    scores, snaps = _FakeScores(), _FakeSnapshots()
    w = WeightWriter(scores, snaps)
    await w.write(
        window_id=1,
        block_number=1,
        scorer_hotkey="s",
        envs=["A", "B"],
        subjects=[
            _subject(
                1, "champ", is_champion=True,
                scores_by_env={"A": {"score": 0.6}, "B": {"mean": 0.8}},
            )
        ],
        outcome={"winner": "champion"},
    )
    avg = next(r["average_score"] for r in scores.rows if r["uid"] == 1)
    assert abs(avg - 0.7) < 1e-9


@pytest.mark.asyncio
async def test_zero_subjects_or_no_champion_raises():
    scores, snaps = _FakeScores(), _FakeSnapshots()
    w = WeightWriter(scores, snaps)
    with pytest.raises(ValueError):
        await w.write(
            window_id=1,
            block_number=1,
            scorer_hotkey="s",
            envs=[],
            subjects=[_subject(1, "loser", is_champion=False)],
            outcome={"winner": "champion"},
        )


@pytest.mark.asyncio
async def test_two_champions_raises():
    scores, snaps = _FakeScores(), _FakeSnapshots()
    w = WeightWriter(scores, snaps)
    with pytest.raises(ValueError):
        await w.write(
            window_id=1,
            block_number=1,
            scorer_hotkey="s",
            envs=[],
            subjects=[
                _subject(1, "c1", is_champion=True),
                _subject(2, "c2", is_champion=True),
            ],
            outcome={"winner": "champion"},
        )


@pytest.mark.asyncio
async def test_payload_carries_revision_model_first_block():
    scores, snaps = _FakeScores(), _FakeSnapshots()
    w = WeightWriter(scores, snaps)
    await w.write(
        window_id=1,
        block_number=1,
        scorer_hotkey="s",
        envs=[],
        subjects=[_subject(7, "h7", is_champion=True)],
        outcome={"winner": "champion"},
    )
    row = scores.rows[0]
    assert row["miner_hotkey"] == "h7"
    assert row["model_revision"] == "rev7"
    assert row["model"] == "org/m7"
    assert row["first_block"] == 700
    assert row["total_samples"] == 200
