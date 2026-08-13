"""Joint persistent multi-stem reconstruction from uncertain 4-D proposals."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MultiStemRodContract:
    instance_ids: tuple[str, ...]
    coordinate_frame: str
    timeline: str
    fps: float
    nodes_per_stem: int
    root_nodes: tuple[int, ...]
    root_modes: tuple[str, ...]
    geometry_evidence: str
    metric_scale_verified: bool
    independent_calibration_groups: int = 0

    def validate(self) -> None:
        stems = len(self.instance_ids)
        if stems == 0 or len(set(self.instance_ids)) != stems:
            raise ValueError("multi-stem contract requires unique instance IDs")
        if any(not value.strip() for value in self.instance_ids):
            raise ValueError("multi-stem instance IDs cannot be blank")
        if not self.coordinate_frame.strip() or not self.timeline.strip():
            raise ValueError("multi-stem coordinate frame and timeline must be named")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("multi-stem FPS must be finite and positive")
        if self.nodes_per_stem < 3:
            raise ValueError("each stem requires at least three material nodes")
        if len(self.root_nodes) != stems or len(self.root_modes) != stems:
            raise ValueError("every stem requires one root node and root mode")
        if any(
            root not in (0, self.nodes_per_stem - 1)
            for root in self.root_nodes
        ):
            raise ValueError("stem roots must be material endpoints")
        if any(mode not in {"fixed", "free"} for mode in self.root_modes):
            raise ValueError("stem root mode must be fixed or free")
        if not self.geometry_evidence.strip():
            raise ValueError("multi-stem geometry evidence must be named")
        if self.independent_calibration_groups < 0:
            raise ValueError("calibration group count cannot be negative")
        if self.metric_scale_verified and self.independent_calibration_groups < 2:
            raise ValueError(
                "verified metric scale requires at least two independent groups"
            )
        if self.metric_scale_verified and self.geometry_evidence not in {
            "calibrated_geometry",
            "sensor_measurement",
            "physics_solver_estimate",
        }:
            raise ValueError(
                "verified metric scale cannot use foundation-model evidence"
            )


def _interpolate_missing(np: Any, observations: Any, visible: Any) -> Any:
    frames, stems, nodes, _ = observations.shape
    result = np.empty_like(observations, dtype=np.float64)
    timeline = np.arange(frames, dtype=np.float64)
    for stem in range(stems):
        for node in range(nodes):
            selected = np.flatnonzero(visible[:, stem, node])
            if not len(selected):
                raise ValueError(
                    f"stem {stem} node {node} has no visible observation"
                )
            for axis in range(3):
                values = observations[selected, stem, node, axis]
                result[:, stem, node, axis] = np.interp(
                    timeline,
                    selected.astype(np.float64),
                    values,
                )
    return result


def _reference_segment_lengths(np: Any, observations: Any, visible: Any) -> Any:
    _, stems, nodes, _ = observations.shape
    lengths = np.empty((stems, nodes - 1), dtype=np.float64)
    for stem in range(stems):
        for segment in range(nodes - 1):
            selected = (
                visible[:, stem, segment]
                & visible[:, stem, segment + 1]
            )
            values = np.linalg.norm(
                observations[selected, stem, segment + 1]
                - observations[selected, stem, segment],
                axis=1,
            )
            values = values[np.isfinite(values) & (values > 1e-8)]
            if not len(values):
                raise ValueError(
                    f"stem {stem} segment {segment} has no joint visible support"
                )
            lengths[stem, segment] = float(np.median(values))
    return lengths


def audit_multistem_identity_swaps(
    np: Any,
    *,
    centerlines: Any,
    confidence: Any,
    improvement_margin: float = 0.20,
    maximum_stems_for_exact_audit: int = 6,
) -> dict[str, object]:
    """Detect transitions where a non-identity assignment is materially cheaper."""

    centers = np.asarray(centerlines, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    if centers.ndim != 4 or centers.shape[-1] != 3:
        raise ValueError("identity audit centerlines must have shape TxSxNx3")
    if conf.shape != centers.shape[:-1]:
        raise ValueError("identity audit confidence must have shape TxSxN")
    frames, stems, _, _ = centers.shape
    if stems > maximum_stems_for_exact_audit:
        raise ValueError(
            "exact identity audit stem count exceeds the declared factorial bound"
        )
    if not (
        np.all(np.isfinite(centers))
        and np.all(np.isfinite(conf))
        and np.all(conf >= 0)
    ):
        raise ValueError("identity audit inputs must be finite")
    weights = np.maximum(conf, 1e-8)
    centroids = np.sum(centers * weights[..., None], axis=2) / np.sum(
        weights,
        axis=2,
    )[..., None]
    material_lengths = np.sum(
        np.linalg.norm(centers[:, :, 1:] - centers[:, :, :-1], axis=-1),
        axis=2,
    )
    scale = max(float(np.median(material_lengths)), 1e-6)
    permutations = tuple(itertools.permutations(range(stems)))
    identity = tuple(range(stems))
    flagged = []
    for frame in range(1, frames):
        costs = {}
        for permutation in permutations:
            costs[permutation] = float(
                sum(
                    np.linalg.norm(
                        centroids[frame, target]
                        - centroids[frame - 1, source]
                    )
                    for source, target in enumerate(permutation)
                )
                / (stems * scale)
            )
        best = min(costs, key=costs.get)
        identity_cost = costs[identity]
        best_cost = costs[best]
        if (
            best != identity
            and best_cost + improvement_margin < identity_cost
        ):
            flagged.append(
                {
                    "frame": frame,
                    "identity_cost_normalized": identity_cost,
                    "best_cost_normalized": best_cost,
                    "best_assignment": list(best),
                }
            )
    return {
        "frames": frames,
        "stems": stems,
        "flagged_transitions": flagged,
        "suspected_id_swap_count": len(flagged),
        "passed": not flagged,
    }


def optimize_multistem_rod_trajectories(
    np: Any,
    *,
    observations: Any,
    confidence: Any,
    contract: MultiStemRodContract,
    proposal_sigma: Any | None = None,
    iterations: int = 80,
    observation_gain: float = 0.35,
    temporal_gain: float = 0.18,
    length_projection_passes: int = 4,
    maximum_segment_length_cv: float = 0.12,
    maximum_observation_residual_fraction_p95: float = 0.10,
) -> dict[str, Any]:
    """Optimize all stems and frames while preserving material identity and arc length."""

    contract.validate()
    if iterations <= 0:
        raise ValueError("multi-stem optimization iterations must be positive")
    if length_projection_passes <= 0:
        raise ValueError("length projection passes must be positive")
    if not 0 < observation_gain <= 1 or not 0 <= temporal_gain < 0.5:
        raise ValueError("multi-stem observation and temporal gains are invalid")
    if maximum_observation_residual_fraction_p95 <= 0:
        raise ValueError("observation residual limit must be positive")
    observed = np.asarray(observations, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    expected_tail = (
        len(contract.instance_ids),
        contract.nodes_per_stem,
        3,
    )
    if observed.ndim != 4 or observed.shape[1:] != expected_tail:
        raise ValueError(
            f"multi-stem observations must have shape Tx{expected_tail}"
        )
    if conf.shape != observed.shape[:-1]:
        raise ValueError("multi-stem confidence must have shape TxSxN")
    if not bool(np.all(np.isfinite(conf)) and np.all(conf >= 0)):
        raise ValueError("multi-stem confidence must be finite and non-negative")
    visible = np.isfinite(observed).all(axis=-1) & (conf > 0)
    filled = _interpolate_missing(np, observed, visible)
    reference_lengths = _reference_segment_lengths(np, observed, visible)
    normalized_confidence = conf / np.maximum(
        np.max(conf, axis=0, keepdims=True),
        1e-8,
    )
    normalized_confidence[~visible] = 0.0
    fixed_roots = {}
    for stem, (root, mode) in enumerate(
        zip(contract.root_nodes, contract.root_modes)
    ):
        if mode != "fixed":
            continue
        selected = visible[:, stem, root]
        if not bool(np.any(selected)):
            raise ValueError(f"fixed root for stem {stem} is never visible")
        weights = conf[selected, stem, root]
        fixed_roots[stem] = np.average(
            observed[selected, stem, root],
            axis=0,
            weights=weights,
        )

    state = filled.copy()

    def project_lengths(passes: int) -> None:
        nonlocal state
        for _ in range(passes):
            for frame in range(len(state)):
                for stem in range(len(contract.instance_ids)):
                    root = contract.root_nodes[stem]
                    fixed = contract.root_modes[stem] == "fixed"
                    for segment in range(contract.nodes_per_stem - 1):
                        left = segment
                        right = segment + 1
                        delta = (
                            state[frame, stem, right]
                            - state[frame, stem, left]
                        )
                        length = float(np.linalg.norm(delta))
                        if length <= 1e-12:
                            continue
                        correction = (
                            length - reference_lengths[stem, segment]
                        ) * (delta / length)
                        if fixed and left == root:
                            state[frame, stem, right] -= correction
                        elif fixed and right == root:
                            state[frame, stem, left] += correction
                        else:
                            state[frame, stem, left] += 0.5 * correction
                            state[frame, stem, right] -= 0.5 * correction
                    if fixed:
                        state[frame, stem, root] = fixed_roots[stem]

    for _ in range(iterations):
        if len(state) > 2 and temporal_gain:
            smoothed = state.copy()
            smoothed[1:-1] += temporal_gain * (
                0.5 * (state[:-2] + state[2:]) - state[1:-1]
            )
            state = smoothed
        observation_delta = np.zeros_like(state)
        observation_delta[visible] = observed[visible] - state[visible]
        state += (
            observation_gain
            * normalized_confidence[..., None]
            * observation_delta
        )
        project_lengths(length_projection_passes)
    project_lengths(max(16, length_projection_passes * 4))
    for frame in range(len(state)):
        for stem in range(len(contract.instance_ids)):
            root = contract.root_nodes[stem]
            if contract.root_modes[stem] == "fixed":
                state[frame, stem, root] = fixed_roots[stem]
            if root == 0:
                segments = range(contract.nodes_per_stem - 1)
                for segment in segments:
                    direction = (
                        state[frame, stem, segment + 1]
                        - state[frame, stem, segment]
                    )
                    direction /= max(float(np.linalg.norm(direction)), 1e-12)
                    state[frame, stem, segment + 1] = (
                        state[frame, stem, segment]
                        + reference_lengths[stem, segment] * direction
                    )
            else:
                segments = range(contract.nodes_per_stem - 2, -1, -1)
                for segment in segments:
                    direction = (
                        state[frame, stem, segment]
                        - state[frame, stem, segment + 1]
                    )
                    direction /= max(float(np.linalg.norm(direction)), 1e-12)
                    state[frame, stem, segment] = (
                        state[frame, stem, segment + 1]
                        + reference_lengths[stem, segment] * direction
                    )

    identity = audit_multistem_identity_swaps(
        np,
        centerlines=state,
        confidence=np.maximum(normalized_confidence, 1e-8),
    )
    segment_lengths = np.linalg.norm(
        state[:, :, 1:] - state[:, :, :-1],
        axis=-1,
    )
    segment_cv = np.std(segment_lengths, axis=0) / np.maximum(
        np.mean(segment_lengths, axis=0),
        1e-8,
    )
    maximum_cv = float(np.max(segment_cv))
    root_error = {}
    for stem, root_position in fixed_roots.items():
        root = contract.root_nodes[stem]
        root_error[contract.instance_ids[stem]] = float(
            np.max(
                np.linalg.norm(
                    state[:, stem, root] - root_position,
                    axis=1,
                )
            )
        )

    if proposal_sigma is None:
        sigma = np.full(conf.shape, 0.005, dtype=np.float64)
        sigma[visible] /= np.sqrt(np.maximum(conf[visible], 1e-6))
    else:
        sigma = np.asarray(proposal_sigma, dtype=np.float64)
        if sigma.shape != conf.shape or not bool(
            np.all(np.isfinite(sigma)) and np.all(sigma >= 0)
        ):
            raise ValueError(
                "proposal sigma must be finite, non-negative, and shaped TxSxN"
            )
    residual = np.zeros(conf.shape, dtype=np.float64)
    residual[visible] = np.linalg.norm(
        state[visible] - observed[visible],
        axis=1,
    )
    frames = len(state)
    for stem in range(len(contract.instance_ids)):
        for node in range(contract.nodes_per_stem):
            selected = np.flatnonzero(visible[:, stem, node])
            for frame in np.flatnonzero(~visible[:, stem, node]):
                distance = int(np.min(np.abs(selected - frame)))
                sigma[frame, stem, node] += (
                    distance / contract.fps
                ) * 0.02
    variance = sigma**2 + residual**2
    covariance = np.zeros((*variance.shape, 3, 3), dtype=np.float64)
    diagonal = np.arange(3)
    covariance[..., diagonal, diagonal] = variance[..., None]
    fully_occluded = ~np.any(visible, axis=2)
    root_passed = max(root_error.values(), default=0.0) <= 1e-9
    material_scale = max(float(np.median(np.sum(reference_lengths, axis=1))), 1e-8)
    observation_residual_fraction_p95 = float(
        np.percentile(residual[visible], 95) / material_scale
    )
    structural_passed = bool(
        maximum_cv <= maximum_segment_length_cv
        and root_passed
        and identity["passed"]
        and observation_residual_fraction_p95
        <= maximum_observation_residual_fraction_p95
    )
    promotion_eligible = bool(
        structural_passed
        and contract.metric_scale_verified
        and contract.independent_calibration_groups >= 2
    )
    return {
        "centerlines": state,
        "velocity": np.gradient(state, 1.0 / contract.fps, axis=0),
        "position_covariance": covariance,
        "visible": visible,
        "reference_segment_lengths": reference_lengths,
        "report": {
            "frames": frames,
            "stems": len(contract.instance_ids),
            "instance_ids": list(contract.instance_ids),
            "coordinate_frame": contract.coordinate_frame,
            "timeline": contract.timeline,
            "geometry_evidence": contract.geometry_evidence,
            "metric_scale_verified": contract.metric_scale_verified,
            "independent_calibration_groups": (
                contract.independent_calibration_groups
            ),
            "visible_fraction_by_stem": [
                float(value)
                for value in np.mean(visible, axis=(0, 2))
            ],
            "fully_occluded_frames_by_stem": {
                contract.instance_ids[stem]: [
                    int(frame)
                    for frame in np.flatnonzero(fully_occluded[:, stem])
                ]
                for stem in range(len(contract.instance_ids))
            },
            "maximum_segment_length_cv": maximum_cv,
            "maximum_fixed_root_error_m": max(
                root_error.values(),
                default=0.0,
            ),
            "observation_residual_fraction_p95": (
                observation_residual_fraction_p95
            ),
            "maximum_observation_residual_fraction_p95": (
                maximum_observation_residual_fraction_p95
            ),
            "identity_audit": identity,
            "structural_passed": structural_passed,
            "promotion_eligible": promotion_eligible,
            "status": "WORKING" if promotion_eligible else "PARTIAL",
            "limitations": (
                []
                if promotion_eligible
                else [
                    "Structure may be usable, but metric promotion requires "
                    "independent calibrated scale and zero identity ambiguity."
                ]
            ),
        },
    }
