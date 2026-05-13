from __future__ import annotations

from types import SimpleNamespace

from affine.src.scheduler.flow import _comparison_scores_by_env


def test_comparison_scores_by_env_uses_real_comparator_thresholds():
    result = SimpleNamespace(
        per_env=[
            SimpleNamespace(
                env="SWE",
                champion_avg=0.8,
                challenger_avg=0.7,
                champion_n=10,
                challenger_n=9,
                margin=0.01,
                not_worse_tolerance=0.025,
                verdict="worse",
                reason="regressed_beyond_tolerance",
            )
        ]
    )

    payload = _comparison_scores_by_env(result, role="challenger")

    assert payload == {
        "SWE": {
            "score": 0.7,
            "score_on_common": 0.7,
            "sample_count": 9,
            "common_tasks": 9,
            "not_worse_threshold": 0.78,
            "dethrone_threshold": 0.81,
            "verdict": "worse",
            "reason": "regressed_beyond_tolerance",
        }
    }


def test_comparison_scores_by_env_champion_payload_has_no_fake_thresholds():
    result = SimpleNamespace(
        per_env=[
            SimpleNamespace(
                env="SWE",
                champion_avg=0.8,
                challenger_avg=0.7,
                champion_n=10,
                challenger_n=9,
                margin=0.01,
                not_worse_tolerance=0.025,
                verdict="worse",
                reason="regressed_beyond_tolerance",
            )
        ]
    )

    payload = _comparison_scores_by_env(result, role="champion")

    assert payload == {
        "SWE": {
            "score": 0.8,
            "sample_count": 10,
            "historical_count": 10,
        }
    }
