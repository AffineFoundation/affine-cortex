from __future__ import annotations

from affine.src.behavior_guard import (
    BehaviorGateConfig,
    BehaviorGateMode,
    DeploymentIdentity,
    ProbeClassification,
    ProbeResult,
    SampleOutcomeEvidence,
    VerdictStatus,
    aggregate_probe_results,
    classify_runtime_timeout_outcome,
    classify_sample_invariant,
    deployment_fingerprint,
    parse_behavior_gate_config,
)


def _result(probe_id: str, classification: ProbeClassification) -> ProbeResult:
    return ProbeResult(probe_id=probe_id, classification=classification)


def test_default_config_is_disabled_and_fail_open() -> None:
    config = parse_behavior_gate_config(None)

    assert config == BehaviorGateConfig()
    assert config.mode is BehaviorGateMode.SHADOW
    assert config.enforces is False
    assert config.gates_environment("SWE-INFINITE") is False


def test_config_parser_accepts_nested_block_and_clamps_related_values() -> None:
    config = parse_behavior_gate_config(
        {
            "behavior_gate": {
                "enabled": "true",
                "mode": "enforce",
                "policy_version": "guard-2",
                "gated_environments": ["SWE-INFINITE", "terminal", "terminal"],
                "probe_count": 2,
                "clean_to_pass": 99,
                "violations_to_fail": 0,
                "probe_concurrency": 10,
                "probe_timeout_seconds": 30,
                "first_response_deadline_seconds": 50,
                "first_action_deadline_seconds": 1,
                "lease_seconds": 2,
            }
        }
    )

    assert config.enabled is True
    assert config.enforces is True
    assert config.policy_version == "guard-2"
    assert config.gated_environments == ("SWE-INFINITE", "terminal")
    assert config.gates_environment("TERMINAL") is True
    assert config.gates_environment("MEMORY") is False
    assert config.probe_count == 3
    assert config.clean_to_pass == 3
    assert config.violations_to_fail == 1
    assert config.probe_concurrency == 1
    assert config.first_response_deadline_seconds == 30
    assert config.first_action_deadline_seconds == 30
    assert config.lease_seconds >= 60


def test_unknown_mode_falls_back_to_non_enforcing_shadow() -> None:
    config = BehaviorGateConfig.from_mapping({"enabled": True, "mode": "typo"})

    assert config.enabled is True
    assert config.mode is BehaviorGateMode.SHADOW
    assert config.enforces is False


def test_lease_covers_full_endpoint_admission_budget() -> None:
    config = parse_behavior_gate_config(
        {
            "admission_timeout_seconds": 600,
            "probe_timeout_seconds": 120,
            "lease_seconds": 1,
        }
    )

    assert config.admission_timeout_seconds == 600
    assert config.lease_seconds == 630


def test_admission_budget_covers_all_baseline_probes() -> None:
    config = parse_behavior_gate_config(
        {
            "probe_count": 3,
            "probe_timeout_seconds": 120,
            "admission_timeout_seconds": 300,
        }
    )

    assert config.admission_timeout_seconds == 390
    assert config.lease_seconds >= 420


def test_runtime_invariant_requires_repeated_distinct_tasks() -> None:
    config = parse_behavior_gate_config({"runtime_violations_to_fail": 1})

    assert config.runtime_violations_to_fail == 2


def test_runtime_timeout_policy_requires_a_bounded_denominator_and_two_events() -> None:
    config = parse_behavior_gate_config(
        {
            "runtime_timeout_rate_to_fail": 0,
            "runtime_timeout_min_samples": 1,
            "runtime_timeout_min_timeouts": 1,
        }
    )

    assert config.runtime_timeout_rate_to_fail == 0.01
    assert config.runtime_timeout_min_samples == 10
    assert config.runtime_timeout_min_timeouts == 2


def test_runtime_timeout_thresholds_change_policy_identity() -> None:
    baseline = BehaviorGateConfig()
    stricter = BehaviorGateConfig(runtime_timeout_rate_to_fail=0.2)

    assert baseline.policy_identity != stricter.policy_identity


def test_runtime_timeout_classifier_counts_only_model_attributed_timeouts() -> None:
    assert classify_runtime_timeout_outcome(
        extra={
            "timed_out": True,
            "harness_failure": False,
            "failure_mode": "model_timeout",
        },
        error="model exceeded deadline",
        success=False,
    ) is ProbeClassification.MODEL_NO_PROGRESS
    assert classify_runtime_timeout_outcome(
        extra={
            "timed_out": True,
            "harness_failure": True,
            "failure_mode": "harness_timeout",
        },
        error="environment exceeded deadline",
        success=False,
    ) is ProbeClassification.INFRA_FAILURE
    assert classify_runtime_timeout_outcome(
        extra={"timed_out": True},
        error="request exceeded deadline",
        success=False,
    ) is ProbeClassification.UNKNOWN
    assert classify_runtime_timeout_outcome(
        extra={},
        success=True,
    ) is ProbeClassification.CLEAN


def test_bounded_quality_failures_are_admissible_but_not_clean() -> None:
    config = BehaviorGateConfig(clean_to_pass=2)
    verdict = aggregate_probe_results(
        [
            _result("wrong", ProbeClassification.QUALITY_FAILURE),
            _result("refusal", ProbeClassification.QUALITY_FAILURE),
        ],
        config,
    )

    assert verdict.status is VerdictStatus.PASSED
    assert verdict.reason == "bounded_completion_threshold"
    assert verdict.clean_count == 0
    assert verdict.quality_failure_count == 2
    assert verdict.admissible_completion_count == 2
    assert verdict.strike_count == 0


def test_only_model_protocol_liveness_and_harness_failures_are_strikes() -> None:
    strong = (
        ProbeClassification.MODEL_NO_PROGRESS,
        ProbeClassification.MODEL_PROTOCOL_FAILURE,
        ProbeClassification.HARNESS_INVALID,
    )

    assert all(_result(str(index), item).is_strike for index, item in enumerate(strong))
    assert not _result("infra", ProbeClassification.INFRA_FAILURE).is_strike
    assert not _result("quality", ProbeClassification.QUALITY_FAILURE).is_strike
    assert ProbeClassification.NO_PROGRESS is ProbeClassification.MODEL_NO_PROGRESS
    assert (
        ProbeClassification.PROTOCOL_FAILURE
        is ProbeClassification.MODEL_PROTOCOL_FAILURE
    )


def test_infrastructure_failures_never_strike_and_eventually_defer() -> None:
    config = BehaviorGateConfig(probe_count=3, max_infra_retries=2)
    initial = [
        _result(f"infra-{index}", ProbeClassification.INFRA_FAILURE)
        for index in range(3)
    ]

    retryable = aggregate_probe_results(initial, config)
    deferred = aggregate_probe_results(
        initial
        + [
            _result("infra-3", ProbeClassification.INFRA_FAILURE),
            _result("infra-4", ProbeClassification.INFRA_FAILURE),
        ],
        config,
    )

    assert retryable.status is VerdictStatus.RUNNING
    assert retryable.strike_count == 0
    assert deferred.status is VerdictStatus.DEFERRED
    assert deferred.infra_failure_count == config.attempt_budget
    assert deferred.strike_count == 0


def test_aggregate_failure_is_deterministic_across_result_order() -> None:
    config = BehaviorGateConfig(violations_to_fail=2)
    results = [
        _result("no-progress", ProbeClassification.MODEL_NO_PROGRESS),
        _result("protocol", ProbeClassification.MODEL_PROTOCOL_FAILURE),
        _result("clean", ProbeClassification.CLEAN),
    ]

    forward = aggregate_probe_results(results, config)
    reverse = aggregate_probe_results(reversed(results), config)

    assert forward == reverse
    assert forward.status is VerdictStatus.FAILED
    assert forward.reason == "strike_threshold:model_protocol_failure"
    assert forward.strike_count == 2


def test_one_strike_never_turns_into_a_pass() -> None:
    config = BehaviorGateConfig(clean_to_pass=2, violations_to_fail=2)
    verdict = aggregate_probe_results(
        [
            _result("bad", ProbeClassification.MODEL_NO_PROGRESS),
            _result("clean-1", ProbeClassification.CLEAN),
            _result("clean-2", ProbeClassification.CLEAN),
        ],
        config,
    )

    assert verdict.status is VerdictStatus.SUSPECTED
    assert verdict.strike_count == 1


def test_deployment_fingerprint_is_order_independent_and_excludes_url_secrets() -> None:
    first = DeploymentIdentity(
        endpoint_name="primary",
        deployment_id="dep-1",
        base_url="HTTPS://user:password@GPU.EXAMPLE/v1/?token=secret-a#fragment",
    )
    second = DeploymentIdentity(
        endpoint_name="spare",
        deployment_id="dep-2",
        base_url="https://gpu-2.example/v1",
    )
    expected = deployment_fingerprint(
        hotkey="hk",
        revision="rev",
        policy_version="v1",
        deployments=[first, second],
    )
    reordered_with_other_secrets = deployment_fingerprint(
        hotkey="hk",
        revision="rev",
        policy_version="v1",
        deployments=[
            second,
            DeploymentIdentity(
                endpoint_name="primary",
                deployment_id="dep-1",
                base_url="https://other:credentials@gpu.example/v1?token=secret-b",
            ),
        ],
    )

    assert expected == reordered_with_other_secrets
    assert expected.startswith("bg1:")
    assert "secret" not in expected
    assert deployment_fingerprint(
        hotkey="hk",
        revision="rev",
        policy_version="v1",
        deployments=[
            DeploymentIdentity("dep-changed", first.base_url, "primary"), second
        ],
    ) != expected
    assert deployment_fingerprint(
        hotkey="hk",
        revision="another-revision",
        policy_version="v1",
        deployments=[first, second],
    ) != expected
    assert deployment_fingerprint(
        hotkey="hk",
        revision="rev",
        policy_version="v1",
        deployments=[first, second],
        deployment_generation=2,
    ) != deployment_fingerprint(
        hotkey="hk",
        revision="rev",
        policy_version="v1",
        deployments=[first, second],
        deployment_generation=1,
    )


def test_uid208_zero_action_positive_score_is_invalid_harness_evidence() -> None:
    uid208 = SampleOutcomeEvidence(
        score=1,
        commands_executed=0,
        llm_call_count=0,
        total_tokens=0,
        output_bytes=0,
        terminated_reason="error_exit_3",
    )

    assert classify_sample_invariant(uid208) is ProbeClassification.HARNESS_INVALID
    assert classify_sample_invariant(
        SampleOutcomeEvidence(0, 0, 0, 0, 0, "error_exit_3")
    ) is None
    assert classify_sample_invariant(
        SampleOutcomeEvidence(1, 1, 0, 0, 0, "completed")
    ) is None
    # Missing telemetry is unknown, not silently treated as zero activity.
    assert classify_sample_invariant(
        SampleOutcomeEvidence(1, None, 0, 0, 0, "error_exit_3")
    ) is None
