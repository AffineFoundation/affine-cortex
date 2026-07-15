#!/usr/bin/env python3
"""Generate/check the vendored manifest from an InstructionGym release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from instruction_gym.ifeval_seeded import load_template_catalog


SCHEMA_VERSION = "1.0"
SELECTION_ALGORITHM = "template_uniform_assignment_uniform_v1"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "affine"
    / "core"
    / "data"
    / "instruction_gym_sampling_manifest_v1.json"
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_manifest(ifeval_path: Path) -> dict[str, Any]:
    catalog = load_template_catalog(ifeval_path)
    templates = []
    for index, template in enumerate(catalog.templates):
        start = catalog.prefix_offsets[index]
        end = catalog.prefix_offsets[index + 1]
        templates.append(
            {
                "template_id": template.template_id,
                "source_key": template.source_key,
                "case_id_start": start,
                "case_id_end": end,
                "cardinality": template.domain_cardinality,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "universe_id": catalog.universe_id,
        "catalog_sha256": catalog.catalog_sha256,
        "case_id_end": catalog.universe_cardinality,
        "template_count": len(catalog.templates),
        "templates": templates,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return manifest


def serialized(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifeval-path", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = serialized(build_manifest(args.ifeval_path))
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != expected:
            raise SystemExit(f"sampling manifest is stale: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
