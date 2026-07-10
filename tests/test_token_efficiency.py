from affine.src.scorer.token_efficiency import (
    SampleMetric,
    TokenEfficiencyConfig,
    compute_token_efficiency,
    extract_token_usage,
    load_token_efficiency_config,
)
from affine.src.scorer.weight_writer import _average_of_env_scores


def test_extract_token_usage_prefers_top_level_usage_over_calls():
    usage = extract_token_usage({
        "usage": {"total_tokens": 100},
        "calls": [{"usage": {"total_tokens": 999}}],
    })

    assert usage is not None
    assert usage.token_count == 100


def test_extract_token_usage_sums_calls_when_no_top_level_usage():
    usage = extract_token_usage({
        "inference_calls": [
            {"usage": {"prompt_tokens": 20, "completion_tokens": 5}},
            {"usage": {"total_tokens": 30}},
        ],
    })

    assert usage is not None
    assert usage.token_count == 55


def test_token_efficiency_skips_missing_usage_pairs():
    cfg = TokenEfficiencyConfig(
        enabled_for_sampling=True,
        enabled_for_scoring=True,
        min_pairs=2,
    )
    champion = {
        "SWE": {
            1: SampleMetric(0.8, extract_token_usage({"usage": {"total_tokens": 100}})),
            2: SampleMetric(0.9, extract_token_usage({"usage": {"total_tokens": 100}})),
            3: SampleMetric(0.9, None),
        }
    }
    challenger = {
        "SWE": {
            1: SampleMetric(0.8, extract_token_usage({"usage": {"total_tokens": 80}})),
            2: SampleMetric(0.9, extract_token_usage({"usage": {"total_tokens": 90}})),
            3: SampleMetric(0.9, extract_token_usage({"usage": {"total_tokens": 1}})),
        }
    }

    result = compute_token_efficiency(
        env="TOKEN-EFFICIENCY",
        config=cfg,
        basis_metrics_by_runtime_env=champion,
        subject_metrics_by_runtime_env=challenger,
        overlap_ids_by_runtime_env={"SWE": {1, 2, 3}},
    )

    assert result.available is True
    assert result.challenger_payload["token_pairs"] == 2
    assert result.challenger_payload["expected_pairs"] == 3
    assert result.challenger_payload["avg_tokens"] == 85.0
    assert result.challenger_payload["verdict"] == "dominant"


def test_token_efficiency_unavailable_when_min_pairs_missing():
    cfg = TokenEfficiencyConfig(enabled_for_sampling=True, min_pairs=2)
    champion = {
        "SWE": {
            1: SampleMetric(0.8, extract_token_usage({"usage": {"total_tokens": 100}})),
        }
    }
    challenger = {
        "SWE": {
            1: SampleMetric(0.8, extract_token_usage({"usage": {"total_tokens": 80}})),
        }
    }

    result = compute_token_efficiency(
        env="TOKEN-EFFICIENCY",
        config=cfg,
        basis_metrics_by_runtime_env=champion,
        subject_metrics_by_runtime_env=challenger,
        overlap_ids_by_runtime_env={"SWE": {1}},
    )

    assert result.available is False
    assert result.reason == "insufficient_token_pairs"
    assert result.challenger_score is None


def test_token_config_rejects_hidden_scoring():
    environments = {
        "TOKEN-EFFICIENCY": {
            "kind": "derived",
            "derived_metric": "token_efficiency",
            "enabled_for_sampling": False,
            "enabled_for_scoring": True,
        }
    }

    try:
        load_token_efficiency_config(environments)
    except ValueError as exc:
        assert "enabled_for_scoring=true" in str(exc)
    else:
        raise AssertionError("expected hidden token scoring config to fail")


def test_token_config_max_ratio_cannot_hide_dominant_threshold():
    cfg = load_token_efficiency_config({
        "TOKEN-EFFICIENCY": {
            "kind": "derived",
            "derived_metric": "token_efficiency",
            "enabled_for_sampling": True,
            "enabled_for_scoring": True,
            "scoring": {
                "savings_margin": 0.05,
                "max_score_ratio": 1.0,
            },
        }
    })

    assert cfg is not None
    assert cfg.max_score_ratio >= cfg.dominant_ratio_threshold


def test_token_payload_is_skipped_by_average_score():
    token_payload = {
        "avg": 1.2,
        "unit": "tokens",
        "lower_is_better": True,
        "include_in_average_score": False,
    }

    assert _average_of_env_scores({
        "SWE": {"avg": 0.8},
        "TOKEN-EFFICIENCY": token_payload,
    }) == 0.8
