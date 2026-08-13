from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from phiagent.perception.multistem_rod_optimizer import (
    MultiStemRodContract,
    audit_multistem_identity_swaps,
    optimize_multistem_rod_trajectories,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, MultiStemRodContract]:
    rng = np.random.default_rng(7)
    frames, stems, nodes = 40, 2, 6
    truth = np.zeros((frames, stems, nodes, 3), dtype=np.float64)
    for stem, y in enumerate((-0.10, 0.10)):
        truth[:, stem, :, 1] = y
        truth[:, stem, :, 2] = np.linspace(0.0, 0.5, nodes)
        motion = 0.025 * np.sin(np.linspace(0.0, 2.0 * np.pi, frames))
        truth[:, stem, 1:, 0] = motion[:, None] * np.linspace(
            0.2,
            1.0,
            nodes - 1,
        )[None, :]
    observations = truth + rng.normal(0.0, 0.003, truth.shape)
    confidence = np.ones((frames, stems, nodes), dtype=np.float64)
    observations[10:20, 0] = np.nan
    confidence[10:20, 0] = 0.0
    contract = MultiStemRodContract(
        instance_ids=("stem-left", "stem-right"),
        coordinate_frame="world:relative_proposal",
        timeline="frame:source_video",
        fps=24.0,
        nodes_per_stem=nodes,
        root_nodes=(0, 0),
        root_modes=("fixed", "fixed"),
        geometry_evidence="foundation_model_estimate",
        metric_scale_verified=False,
    )
    return observations, confidence, contract


def test_multistem_optimizer_preserves_roots_and_occlusion_uncertainty() -> None:
    observations, confidence, contract = _fixture()
    result = optimize_multistem_rod_trajectories(
        np,
        observations=observations,
        confidence=confidence,
        contract=contract,
    )
    report = result["report"]

    assert report["structural_passed"] is True
    assert report["promotion_eligible"] is False
    assert report["status"] == "PARTIAL"
    assert report["maximum_segment_length_cv"] <= 0.12
    assert report["maximum_fixed_root_error_m"] == 0.0
    assert report["identity_audit"]["passed"] is True
    assert report["fully_occluded_frames_by_stem"]["stem-left"] == list(
        range(10, 20)
    )
    visible_variance = result["position_covariance"][5, 0, 2, 0, 0]
    occluded_variance = result["position_covariance"][15, 0, 2, 0, 0]
    assert occluded_variance > visible_variance


def test_multistem_optimizer_allows_metric_only_with_independent_groups() -> None:
    observations, confidence, contract = _fixture()
    metric_contract = replace(
        contract,
        coordinate_frame="world:calibrated",
        geometry_evidence="calibrated_geometry",
        metric_scale_verified=True,
        independent_calibration_groups=2,
    )
    result = optimize_multistem_rod_trajectories(
        np,
        observations=observations,
        confidence=confidence,
        contract=metric_contract,
    )

    assert result["report"]["promotion_eligible"] is True
    assert result["report"]["status"] == "WORKING"


def test_identity_audit_detects_midsequence_stem_swap() -> None:
    observations, confidence, _ = _fixture()
    filled = observations.copy()
    for frame in range(10, 20):
        filled[frame, 0] = filled[9, 0]
    filled[20:, [0, 1]] = filled[20:, [1, 0]]
    result = audit_multistem_identity_swaps(
        np,
        centerlines=filled,
        confidence=np.maximum(confidence, 1e-8),
    )

    assert result["passed"] is False
    assert result["suspected_id_swap_count"] >= 1
    assert any(row["frame"] == 20 for row in result["flagged_transitions"])


def test_foundation_geometry_cannot_self_declare_metric() -> None:
    _, _, contract = _fixture()
    invalid = replace(
        contract,
        metric_scale_verified=True,
        independent_calibration_groups=2,
    )

    with pytest.raises(ValueError, match="cannot use foundation-model"):
        invalid.validate()
