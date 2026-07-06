"""Token efficiency derived scoring.

This module is the single source of truth for extracting usage, computing
token averages, building comparator inputs, and producing rank/snapshot
payloads for the TOKEN-EFFICIENCY derived env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set

from affine.src.scorer.comparator import EnvComparisonConfig


TOKEN_EFFICIENCY_ENV = "TOKEN-EFFICIENCY"
TOKEN_UNIT = "tokens"


@dataclass(frozen=True)
class TokenUsage:
    total_tokens: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    source: str

    @property
    def token_count(self) -> int | None:
        if isinstance(self.total_tokens, int) and self.total_tokens > 0:
            return self.total_tokens
        if (
            isinstance(self.prompt_tokens, int)
            and isinstance(self.completion_tokens, int)
            and self.prompt_tokens >= 0
            and self.completion_tokens >= 0
        ):
            total = self.prompt_tokens + self.completion_tokens
            return total if total > 0 else None
        return None


@dataclass(frozen=True)
class SampleMetric:
    score: float
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class TokenEfficiencyConfig:
    env: str = TOKEN_EFFICIENCY_ENV
    enabled_for_sampling: bool = False
    enabled_for_scoring: bool = False
    min_pairs: int = 100
    savings_margin: float = 0.05
    extra_token_tolerance: float = 0.02
    max_score_ratio: float = 2.0

    @property
    def dominant_ratio_threshold(self) -> float:
        return 1.0 / (1.0 - self.savings_margin)

    @property
    def not_worse_ratio_threshold(self) -> float:
        return 1.0 / (1.0 + self.extra_token_tolerance)


@dataclass(frozen=True)
class TokenEfficiencyComputation:
    env: str
    available: bool
    reason: str
    champion_score: float | None
    challenger_score: float | None
    comparison_config: EnvComparisonConfig | None
    champion_payload: Dict[str, Any]
    challenger_payload: Dict[str, Any]
    snapshot_metric: Dict[str, Any]


def load_token_efficiency_config(
    environments: Mapping[str, Any],
) -> TokenEfficiencyConfig | None:
    raw = environments.get(TOKEN_EFFICIENCY_ENV)
    if not isinstance(raw, dict):
        return None
    if str(raw.get("kind") or "").lower() != "derived":
        return None
    if str(raw.get("derived_metric") or "") != "token_efficiency":
        return None

    enabled_for_sampling = bool(raw.get("enabled_for_sampling", False))
    enabled_for_scoring = bool(raw.get("enabled_for_scoring", False))
    if enabled_for_scoring and not enabled_for_sampling:
        raise ValueError(
            "TOKEN-EFFICIENCY cannot have enabled_for_scoring=true while "
            "enabled_for_sampling=false"
        )

    scoring = raw.get("scoring") or {}
    if not isinstance(scoring, dict):
        scoring = {}
    savings_margin = max(0.0, float(scoring.get("savings_margin", 0.05) or 0.0))
    savings_margin = min(savings_margin, 0.99)
    dominant_ratio_threshold = 1.0 / (1.0 - savings_margin)
    configured_max_ratio = max(
        1.0, float(scoring.get("max_score_ratio", 2.0) or 2.0)
    )
    return TokenEfficiencyConfig(
        env=TOKEN_EFFICIENCY_ENV,
        enabled_for_sampling=enabled_for_sampling,
        enabled_for_scoring=enabled_for_scoring,
        min_pairs=max(1, int(scoring.get("min_pairs", 100) or 100)),
        savings_margin=savings_margin,
        extra_token_tolerance=max(
            0.0, float(scoring.get("extra_token_tolerance", 0.02) or 0.0)
        ),
        max_score_ratio=max(configured_max_ratio, dominant_ratio_threshold),
    )


def extract_token_usage(extra: Dict[str, Any]) -> TokenUsage | None:
    if not isinstance(extra, dict):
        return None

    for key in ("usage", "openai_usage", "inference_usage"):
        usage = _usage_from_mapping(extra.get(key), source=key)
        if usage and usage.token_count is not None:
            return usage

    for key in ("inference_calls", "calls"):
        usage = _usage_from_calls(extra.get(key), source=key)
        if usage and usage.token_count is not None:
            return usage
    return None


def compute_token_efficiency(
    *,
    env: str,
    config: TokenEfficiencyConfig,
    basis_metrics_by_runtime_env: Mapping[str, Mapping[int, SampleMetric]],
    subject_metrics_by_runtime_env: Mapping[str, Mapping[int, SampleMetric]],
    overlap_ids_by_runtime_env: Mapping[str, Set[int]],
    subject_is_reference: bool = False,
) -> TokenEfficiencyComputation:
    expected_pairs = sum(len(ids) for ids in overlap_ids_by_runtime_env.values())
    basis_total = 0
    subject_total = 0
    token_pairs = 0

    for runtime_env, overlap_ids in overlap_ids_by_runtime_env.items():
        basis_metrics = basis_metrics_by_runtime_env.get(runtime_env, {})
        subject_metrics = subject_metrics_by_runtime_env.get(runtime_env, {})
        for task_id in overlap_ids:
            basis_usage = basis_metrics.get(task_id).usage if task_id in basis_metrics else None
            subject_usage = (
                subject_metrics.get(task_id).usage if task_id in subject_metrics else None
            )
            basis_tokens = basis_usage.token_count if basis_usage else None
            subject_tokens = subject_usage.token_count if subject_usage else None
            if basis_tokens is None or subject_tokens is None:
                continue
            basis_total += basis_tokens
            subject_total += subject_tokens
            token_pairs += 1

    coverage_ratio = (token_pairs / expected_pairs) if expected_pairs > 0 else 0.0
    basis_avg = (basis_total / token_pairs) if token_pairs else None
    subject_avg = (subject_total / token_pairs) if token_pairs else None

    reason = "ok"
    available = True
    if expected_pairs <= 0:
        available = False
        reason = "no_overlap_pairs"
    elif token_pairs < config.min_pairs:
        available = False
        reason = "insufficient_token_pairs"
    elif not basis_avg or not subject_avg or basis_avg <= 0 or subject_avg <= 0:
        available = False
        reason = "invalid_token_average"

    ratio = None
    saving_rate = None
    verdict = "unavailable"
    if available and basis_avg is not None and subject_avg is not None:
        ratio = basis_avg / subject_avg
        saving_rate = 1.0 - (subject_avg / basis_avg)
        if subject_is_reference:
            verdict = "reference"
            reason = "reference_subject"
        elif ratio > config.dominant_ratio_threshold:
            verdict = "dominant"
            reason = "challenger_used_fewer_tokens"
        elif ratio >= config.not_worse_ratio_threshold:
            verdict = "not_worse"
            reason = "within_token_tolerance"
        else:
            verdict = "worse"
            reason = "used_more_tokens"

    champion_score = 1.0 if available else None
    challenger_score = (
        min(max(float(ratio or 0.0), 0.0), config.max_score_ratio)
        if available
        else None
    )
    comparison_config = None
    if available:
        comparison_config = EnvComparisonConfig(
            env=env,
            margin=config.dominant_ratio_threshold - 1.0,
            min_tasks_per_env=1,
            not_worse_tolerance=1.0 - config.not_worse_ratio_threshold,
        )

    champion_payload = _payload(
        config=config,
        available=available,
        reason="reference_subject" if available else reason,
        avg_tokens=basis_avg,
        basis_avg_tokens=basis_avg,
        ratio_to_champion=1.0 if available else None,
        saving_rate=0.0 if available else None,
        expected_pairs=expected_pairs,
        token_pairs=token_pairs,
        coverage_ratio=coverage_ratio,
        verdict="reference" if available else "unavailable",
        is_reference=True,
    )
    challenger_payload = _payload(
        config=config,
        available=available,
        reason=reason,
        avg_tokens=subject_avg,
        basis_avg_tokens=basis_avg,
        ratio_to_champion=ratio,
        saving_rate=saving_rate,
        expected_pairs=expected_pairs,
        token_pairs=token_pairs,
        coverage_ratio=coverage_ratio,
        verdict=verdict,
        is_reference=subject_is_reference,
    )
    snapshot_metric = {
        "unit": TOKEN_UNIT,
        "token_metric": "total_tokens",
        "available": available,
        "champion_avg_tokens": basis_avg,
        "challenger_avg_tokens": subject_avg,
        "ratio_to_champion": ratio,
        "saving_rate": saving_rate,
        "coverage_ratio": coverage_ratio,
        "expected_pairs": expected_pairs,
        "token_pairs": token_pairs,
        "scoring_config": _scoring_config_payload(config),
    }
    return TokenEfficiencyComputation(
        env=env,
        available=available,
        reason=reason,
        champion_score=champion_score,
        challenger_score=challenger_score,
        comparison_config=comparison_config,
        champion_payload=champion_payload,
        challenger_payload=challenger_payload,
        snapshot_metric=snapshot_metric,
    )


def _usage_from_mapping(raw: Any, *, source: str) -> TokenUsage | None:
    if not isinstance(raw, dict):
        return None
    usage = TokenUsage(
        total_tokens=_int_or_none(raw.get("total_tokens")),
        prompt_tokens=_int_or_none(raw.get("prompt_tokens")),
        completion_tokens=_int_or_none(raw.get("completion_tokens")),
        source=source,
    )
    return usage if usage.token_count is not None else None


def _usage_from_calls(raw: Any, *, source: str) -> TokenUsage | None:
    if not isinstance(raw, list):
        return None
    total = 0
    prompt = 0
    completion = 0
    found = False
    for idx, call in enumerate(raw):
        if not isinstance(call, dict):
            continue
        usage = _usage_from_mapping(call.get("usage"), source=f"{source}[{idx}].usage")
        if usage is None:
            continue
        count = usage.token_count
        if count is None:
            continue
        total += count
        prompt += usage.prompt_tokens or 0
        completion += usage.completion_tokens or 0
        found = True
    if not found:
        return None
    return TokenUsage(
        total_tokens=total,
        prompt_tokens=prompt if prompt else None,
        completion_tokens=completion if completion else None,
        source=source,
    )


def _payload(
    *,
    config: TokenEfficiencyConfig,
    available: bool,
    reason: str,
    avg_tokens: float | None,
    basis_avg_tokens: float | None,
    ratio_to_champion: float | None,
    saving_rate: float | None,
    expected_pairs: int,
    token_pairs: int,
    coverage_ratio: float,
    verdict: str,
    is_reference: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "count": int(token_pairs),
        "avg": float(ratio_to_champion) if ratio_to_champion is not None else 0.0,
        "unit": TOKEN_UNIT,
        "lower_is_better": True,
        "include_in_average_score": False,
        "is_reference": bool(is_reference),
        "available": bool(available),
        "avg_tokens": float(avg_tokens) if avg_tokens is not None else None,
        "champion_overlap_avg_tokens": (
            float(basis_avg_tokens) if basis_avg_tokens is not None else None
        ),
        "ratio_to_champion": (
            float(ratio_to_champion) if ratio_to_champion is not None else None
        ),
        "saving_rate": float(saving_rate) if saving_rate is not None else None,
        "coverage_ratio": float(coverage_ratio),
        "expected_pairs": int(expected_pairs),
        "token_pairs": int(token_pairs),
        "verdict": verdict,
        "reason": reason,
        "scoring_config": _scoring_config_payload(config),
    }
    return payload


def _scoring_config_payload(config: TokenEfficiencyConfig) -> Dict[str, Any]:
    return {
        "min_pairs": config.min_pairs,
        "savings_margin": config.savings_margin,
        "extra_token_tolerance": config.extra_token_tolerance,
        "max_score_ratio": config.max_score_ratio,
        "dominant_ratio_threshold": config.dominant_ratio_threshold,
        "not_worse_ratio_threshold": config.not_worse_ratio_threshold,
    }


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
