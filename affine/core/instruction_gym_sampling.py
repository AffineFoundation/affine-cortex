"""Frozen InstructionGym template layout and production sampling policy."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
SELECTION_ALGORITHM = "template_uniform_assignment_uniform_v1"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "instruction_gym_sampling_manifest_v1.json"
)


@dataclass(frozen=True, slots=True)
class TemplateRange:
    template_id: str
    source_key: int
    case_id_start: int
    case_id_end: int
    cardinality: int


@dataclass(frozen=True, slots=True)
class InstructionGymSamplingManifest:
    schema_version: str
    selection_algorithm: str
    universe_id: str
    catalog_sha256: str
    case_id_end: int
    template_count: int
    templates: tuple[TemplateRange, ...]
    manifest_sha256: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def validate_sampling_manifest(payload: Any) -> InstructionGymSamplingManifest:
    """Validate the vendored manifest before any task ID is sampled."""

    if not isinstance(payload, dict):
        raise ValueError("InstructionGym sampling manifest must be an object")
    expected_keys = {
        "schema_version",
        "selection_algorithm",
        "universe_id",
        "catalog_sha256",
        "case_id_end",
        "template_count",
        "templates",
        "manifest_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            "InstructionGym sampling manifest fields do not match schema 1.0"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported InstructionGym sampling manifest schema")
    if payload["selection_algorithm"] != SELECTION_ALGORITHM:
        raise ValueError("unsupported InstructionGym sampling algorithm")

    expected_digest = payload["manifest_sha256"]
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 hex digest")
    try:
        digest_bytes = bytes.fromhex(expected_digest)
    except ValueError as exc:
        raise ValueError(
            "manifest_sha256 must be a lowercase SHA-256 hex digest"
        ) from exc
    if digest_bytes.hex() != expected_digest:
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 hex digest")
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    actual_digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("InstructionGym sampling manifest SHA-256 mismatch")

    universe_id = payload["universe_id"]
    catalog_sha256 = payload["catalog_sha256"]
    if not isinstance(universe_id, str) or not universe_id:
        raise ValueError("universe_id must be a non-empty string")
    if not isinstance(catalog_sha256, str) or len(catalog_sha256) != 64:
        raise ValueError("catalog_sha256 must be a SHA-256 hex digest")
    try:
        catalog_digest_bytes = bytes.fromhex(catalog_sha256)
    except ValueError as exc:
        raise ValueError("catalog_sha256 must be a SHA-256 hex digest") from exc
    if catalog_digest_bytes.hex() != catalog_sha256:
        raise ValueError("catalog_sha256 must be a lowercase SHA-256 hex digest")
    if not universe_id.endswith(f":{catalog_sha256}"):
        raise ValueError("universe_id and catalog_sha256 disagree")

    case_id_end = _require_int(payload["case_id_end"], "case_id_end")
    template_count = _require_int(payload["template_count"], "template_count")
    raw_templates = payload["templates"]
    if case_id_end <= 0 or template_count <= 0:
        raise ValueError("case_id_end and template_count must be positive")
    if not isinstance(raw_templates, list) or len(raw_templates) != template_count:
        raise ValueError("template_count does not match templates")

    templates: list[TemplateRange] = []
    next_start = 0
    template_ids: set[str] = set()
    source_keys: set[int] = set()
    template_keys = {
        "template_id",
        "source_key",
        "case_id_start",
        "case_id_end",
        "cardinality",
    }
    for index, raw in enumerate(raw_templates):
        if not isinstance(raw, dict) or set(raw) != template_keys:
            raise ValueError(f"templates[{index}] fields do not match schema 1.0")
        template_id = raw["template_id"]
        if not isinstance(template_id, str) or not template_id:
            raise ValueError(f"templates[{index}].template_id must be non-empty")
        source_key = _require_int(raw["source_key"], f"templates[{index}].source_key")
        start = _require_int(raw["case_id_start"], f"templates[{index}].case_id_start")
        end = _require_int(raw["case_id_end"], f"templates[{index}].case_id_end")
        cardinality = _require_int(
            raw["cardinality"], f"templates[{index}].cardinality"
        )
        if template_id in template_ids or source_key in source_keys:
            raise ValueError("template IDs and source keys must be unique")
        if start != next_start or end <= start or cardinality != end - start:
            raise ValueError("template ranges must be contiguous and match cardinality")
        template_ids.add(template_id)
        source_keys.add(source_key)
        templates.append(
            TemplateRange(
                template_id=template_id,
                source_key=source_key,
                case_id_start=start,
                case_id_end=end,
                cardinality=cardinality,
            )
        )
        next_start = end
    if next_start != case_id_end:
        raise ValueError("template ranges do not cover [0, case_id_end)")

    return InstructionGymSamplingManifest(
        schema_version=SCHEMA_VERSION,
        selection_algorithm=SELECTION_ALGORITHM,
        universe_id=universe_id,
        catalog_sha256=catalog_sha256,
        case_id_end=case_id_end,
        template_count=template_count,
        templates=tuple(templates),
        manifest_sha256=expected_digest,
    )


@lru_cache(maxsize=4)
def load_sampling_manifest(
    expected_sha256: str,
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> InstructionGymSamplingManifest:
    """Load the vendored release manifest and require its configured identity."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = validate_sampling_manifest(payload)
    if manifest.manifest_sha256 != expected_sha256:
        raise ValueError(
            "configured InstructionGym sampling manifest SHA-256 does not match vendored data"
        )
    return manifest


def sample_task_ids(
    manifest: InstructionGymSamplingManifest,
    count: int,
    rng: random.Random,
) -> list[int]:
    """Sample templates in balanced rounds, then assignments without replacement.

    Production still supplies ``SystemRandom`` entropy and persists the resulting
    pool. Given the same RNG stream and manifest this versioned mapping is fully
    deterministic; public window/block values never become its seed.
    """

    count = _require_int(count, "sampling_count")
    if not 0 <= count <= manifest.case_id_end:
        raise ValueError(
            f"sampling_count={count} exceeds InstructionGym cases={manifest.case_id_end}"
        )
    if count == 0:
        return []

    remaining = [template.cardinality for template in manifest.templates]
    swaps: list[dict[int, int]] = [{} for _ in manifest.templates]
    selected: list[int] = []
    while len(selected) < count:
        eligible = [index for index, size in enumerate(remaining) if size]
        if (
            not eligible
        ):  # pragma: no cover - count bound and manifest coverage guard this.
            raise RuntimeError("InstructionGym sampling manifest exhausted early")
        rng.shuffle(eligible)
        for index in eligible:
            size = remaining[index]
            draw = rng.randrange(size)
            mapping = swaps[index]
            local_rank = mapping.get(draw, draw)
            last = size - 1
            mapping[draw] = mapping.get(last, last)
            mapping.pop(last, None)
            remaining[index] = last
            selected.append(manifest.templates[index].case_id_start + local_rank)
            if len(selected) == count:
                return sorted(selected)
    raise AssertionError("unreachable")


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "InstructionGymSamplingManifest",
    "SCHEMA_VERSION",
    "SELECTION_ALGORITHM",
    "TemplateRange",
    "load_sampling_manifest",
    "sample_task_ids",
    "validate_sampling_manifest",
]
