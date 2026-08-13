from __future__ import annotations

import pytest

from scripts.evaluate_flower_relighting_window import (
    AUTOMATIC_GATES,
    SEMANTIC_GATES,
    validate_geometry_evaluation,
    validate_manifest,
    validate_review,
)


def test_manifest_requires_exact_relighting_gate_contract() -> None:
    manifest = {
        "automatic_gates": {name: True for name in AUTOMATIC_GATES},
        "source_frame_indices": [272, 275],
        "outputs": {"candidate": {"path": "/tmp/candidate.mp4", "sha256": "a" * 64}},
    }
    assert validate_manifest(manifest) == manifest
    manifest["automatic_gates"].pop(AUTOMATIC_GATES[-1])
    with pytest.raises(ValueError, match="contract"):
        validate_manifest(manifest)


def test_geometry_and_review_are_strictly_bound() -> None:
    geometry = {"decision": "ALLOW_RELIGHTING_WINDOW", "geometry_gate_pass": True}
    assert validate_geometry_evaluation(geometry) == geometry
    review = {
        "reviewer": "frame audit",
        "reviewed_source_frames": [272, 275],
        "candidate_sha256": "a" * 64,
        "semantic_gates": {name: True for name in SEMANTIC_GATES},
    }
    assert validate_review(
        review, expected_frames=[272, 275], candidate_sha256="a" * 64
    ) == review
    review["semantic_gates"][SEMANTIC_GATES[0]] = 1
    with pytest.raises(ValueError, match="JSON boolean"):
        validate_review(review, expected_frames=[272, 275], candidate_sha256="a" * 64)


def test_review_accepts_compact_contiguous_range() -> None:
    review = {
        "reviewer": "dense audit",
        "reviewed_source_frame_range": [272, 275, 1],
        "candidate_sha256": "a" * 64,
        "semantic_gates": {name: True for name in SEMANTIC_GATES},
    }

    validated = validate_review(
        review, expected_frames=[272, 273, 274], candidate_sha256="a" * 64
    )

    assert validated["reviewed_source_frames"] == [272, 273, 274]
