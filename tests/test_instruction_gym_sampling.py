"""InstructionGym manifest pinning and template-stratified sampling."""

from __future__ import annotations

import bisect
import copy
import json
import random
from pathlib import Path

import pytest

from affine.core.environments import INSTRUCTION_GYM_SAMPLING_MANIFEST_SHA256
from affine.core.instruction_gym_sampling import (
    DEFAULT_MANIFEST_PATH,
    load_sampling_manifest,
    sample_task_ids,
    validate_sampling_manifest,
)
from affine.src.scorer.sampler import (
    SAMPLING_MODE_TEMPLATE_STRATIFIED_V1,
    EnvSamplingConfig,
    WindowSampler,
)


def _manifest():
    return load_sampling_manifest(INSTRUCTION_GYM_SAMPLING_MANIFEST_SHA256)


def _template_indices(task_ids):
    manifest = _manifest()
    starts = [template.case_id_start for template in manifest.templates]
    return [bisect.bisect_right(starts, task_id) - 1 for task_id in task_ids]


def test_vendored_manifest_is_complete_and_pinned():
    manifest = _manifest()

    assert manifest.manifest_sha256 == INSTRUCTION_GYM_SAMPLING_MANIFEST_SHA256
    assert manifest.template_count == 541
    assert manifest.case_id_end == 102_636_151
    assert manifest.templates[0].case_id_start == 0
    assert manifest.templates[-1].case_id_end == manifest.case_id_end
    assert all(
        left.case_id_end == right.case_id_start
        for left, right in zip(manifest.templates, manifest.templates[1:])
    )


def test_manifest_tampering_and_wrong_configured_digest_fail_closed():
    payload = json.loads(Path(DEFAULT_MANIFEST_PATH).read_text(encoding="utf-8"))
    tampered = copy.deepcopy(payload)
    tampered["templates"][0]["cardinality"] += 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_sampling_manifest(tampered)
    with pytest.raises(ValueError, match="configured.*does not match"):
        load_sampling_manifest("0" * 64)


def test_template_stratified_sampling_is_deterministic_for_rng_stream():
    manifest = _manifest()
    first = sample_task_ids(manifest, 100, random.Random(20260714))
    second = sample_task_ids(manifest, 100, random.Random(20260714))

    assert first == second
    assert len(first) == len(set(first)) == 100
    assert all(0 <= task_id < manifest.case_id_end for task_id in first)


def test_one_round_is_exactly_uniform_over_templates():
    manifest = _manifest()
    task_ids = sample_task_ids(manifest, manifest.template_count, random.Random(7))

    indices = _template_indices(task_ids)
    assert sorted(indices) == list(range(manifest.template_count))


def test_two_rounds_are_balanced_and_assignments_are_without_replacement():
    manifest = _manifest()
    count = manifest.template_count * 2
    task_ids = sample_task_ids(manifest, count, random.Random(8))

    indices = _template_indices(task_ids)
    assert len(task_ids) == len(set(task_ids)) == count
    assert {index: indices.count(index) for index in set(indices)} == {
        index: 2 for index in range(manifest.template_count)
    }


def test_large_template_no_longer_dominates_a_sampling_window():
    manifest = _manifest()
    dominant_index = next(
        index
        for index, template in enumerate(manifest.templates)
        if template.source_key == 3710
    )
    assert manifest.templates[dominant_index].cardinality / manifest.case_id_end > 0.93

    task_ids = sample_task_ids(manifest, 100, random.Random(9))
    assert _template_indices(task_ids).count(dominant_index) <= 1


def test_window_sampler_consumes_manifest_and_preserves_direct_task_ids(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(WindowSampler, "_rng", lambda self: random.Random(10))
    config = EnvSamplingConfig(
        env="INSTRUCTION-GYM",
        dataset_range=[[0, manifest.case_id_end]],
        sampling_count=100,
        mode=SAMPLING_MODE_TEMPLATE_STRATIFIED_V1,
        sampling_manifest_sha256=manifest.manifest_sha256,
    )
    result = WindowSampler().generate(1, 100, {config.env: config})[config.env]

    assert result == sample_task_ids(manifest, 100, random.Random(10))


def test_public_window_values_do_not_seed_template_sampling(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(WindowSampler, "_rng", lambda self: random.Random(11))
    config = EnvSamplingConfig(
        env="INSTRUCTION-GYM",
        dataset_range=[[0, manifest.case_id_end]],
        sampling_count=100,
        mode=SAMPLING_MODE_TEMPLATE_STRATIFIED_V1,
        sampling_manifest_sha256=manifest.manifest_sha256,
    )

    first = WindowSampler().generate(1, 100, {config.env: config})[config.env]
    second = WindowSampler().generate(999, 9_999_999, {config.env: config})[config.env]
    assert first == second


@pytest.mark.parametrize(
    ("dataset_range", "manifest_sha256", "message"),
    [
        (
            [[1, 102_636_151]],
            INSTRUCTION_GYM_SAMPLING_MANIFEST_SHA256,
            "must exactly match",
        ),
        ([[0, 102_636_151]], "0" * 64, "configured.*does not match"),
    ],
)
def test_window_sampler_rejects_range_or_manifest_drift(
    dataset_range,
    manifest_sha256,
    message,
):
    config = EnvSamplingConfig(
        env="INSTRUCTION-GYM",
        dataset_range=dataset_range,
        sampling_count=1,
        mode=SAMPLING_MODE_TEMPLATE_STRATIFIED_V1,
        sampling_manifest_sha256=manifest_sha256,
    )
    with pytest.raises(ValueError, match=message):
        WindowSampler().generate(1, 100, {config.env: config})
