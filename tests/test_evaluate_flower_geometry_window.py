from __future__ import annotations

import pytest

from scripts.evaluate_flower_geometry_window import (
    AUTOMATIC_GATES,
    SEMANTIC_GATES,
    validate_manifest,
    validate_review,
)


def _manifest() -> dict[str, object]:
    return {
        "source_frame_indices": [272, 275, 278],
        "automatic_gates": {name: True for name in AUTOMATIC_GATES},
        "outputs": {
            "geometry_candidate": {"path": "/remote/candidate.mp4", "sha256": "a" * 64}
        },
    }


def test_manifest_requires_exact_boolean_gate_contract() -> None:
    manifest = _manifest()
    assert validate_manifest(manifest) == manifest
    manifest["automatic_gates"][AUTOMATIC_GATES[0]] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="JSON boolean"):
        validate_manifest(manifest)


def test_review_is_bound_to_frames_and_candidate_hash() -> None:
    review = {
        "reviewer": "frame audit",
        "reviewed_source_frames": [272, 275, 278],
        "candidate_sha256": "a" * 64,
        "semantic_gates": {name: True for name in SEMANTIC_GATES},
    }
    assert validate_review(
        review, expected_frames=[272, 275, 278], candidate_sha256="a" * 64
    ) == review
    review["reviewed_source_frames"] = [272, 275]
    with pytest.raises(ValueError, match="cover"):
        validate_review(
            review, expected_frames=[272, 275, 278], candidate_sha256="a" * 64
        )


def test_review_accepts_compact_source_frame_range() -> None:
    review = {
        "reviewer": "dense frame audit",
        "reviewed_source_frame_range": [272, 275, 1],
        "candidate_sha256": "a" * 64,
        "semantic_gates": {name: True for name in SEMANTIC_GATES},
    }

    validated = validate_review(
        review, expected_frames=[272, 273, 274], candidate_sha256="a" * 64
    )

    assert validated["reviewed_source_frames"] == [272, 273, 274]
