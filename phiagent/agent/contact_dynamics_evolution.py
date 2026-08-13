"""Architecture-level evolution for contact/dynamics video pipelines.

This module deliberately does not search numeric generator settings.  It
compares complete algorithm families on complete held-group tournaments and
requires every immutable hard gate before utility or cost can break ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_STRUCTURAL_REPAIRS = {
    "articulated_metric_hand": {
        "component": "hand_state",
        "change": "replace raster hand synthesis with a fixed articulated metric joint tree",
    },
    "metric_force_closure": {
        "component": "contact_state",
        "change": "add calibrated depth, occlusion order, and force-position contact evidence",
    },
    "causal_stem_motion": {
        "component": "deformable_object_state",
        "change": "drive a rooted deformable-rod backbone from explicit contact forces",
    },
    "persistent_instance_identity": {
        "component": "object_memory",
        "change": "maintain one immutable state trajectory per named flower stem",
    },
    "adversarial_audit": {
        "component": "acceptance",
        "change": "add attacks that erase response, spoof contact, and corrupt topology",
    },
    "human_high_resolution_review": {
        "component": "acceptance",
        "change": "retain full-resolution finger/contact review as a non-overridable veto",
    },
}


@dataclass(frozen=True)
class ArchitectureAssessment:
    group_id: str
    architecture_id: str
    hard_gates: tuple[tuple[str, bool], ...]
    utility: float
    cost_units: float
    evidence_path: str

    def validate(self, required_gates: tuple[str, ...]) -> None:
        if not self.group_id.strip() or not self.architecture_id.strip():
            raise ValueError("group and architecture IDs must be non-empty")
        gates = dict(self.hard_gates)
        if len(gates) != len(self.hard_gates):
            raise ValueError("hard gate names must be unique")
        missing = set(required_gates) - gates.keys()
        extra = gates.keys() - set(required_gates)
        if missing or extra:
            raise ValueError(
                f"hard gates do not match contract; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if not self.evidence_path.strip() or self.cost_units < 0:
            raise ValueError("assessment evidence and non-negative cost are required")

    @property
    def passed(self) -> bool:
        return all(value for _, value in self.hard_gates)


@dataclass(frozen=True)
class ArchitectureEvolutionContract:
    required_gates: tuple[str, ...]
    required_groups: tuple[str, ...]
    architecture_ids: tuple[str, ...]
    maximum_cost_units: float

    def validate(self) -> None:
        for label, values in (
            ("required_gates", self.required_gates),
            ("required_groups", self.required_groups),
            ("architecture_ids", self.architecture_ids),
        ):
            if not values or len(set(values)) != len(values) or any(not value.strip() for value in values):
                raise ValueError(f"{label} must contain unique non-empty values")
        if len(self.required_groups) < 2:
            raise ValueError("evolution requires at least two held groups")
        if len(self.architecture_ids) < 2:
            raise ValueError("evolution requires at least two complete architectures")
        if self.maximum_cost_units <= 0:
            raise ValueError("maximum cost must be positive")


@dataclass(frozen=True)
class FoundationPipelineExperiment:
    """One architecture experiment derived from an observed physical-stage failure."""

    experiment_id: str
    failed_stage: str
    root_cause: str
    architecture_change: str
    required_evidence: tuple[str, ...]
    promotion_gates: tuple[str, ...]
    blocked_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        values = (
            self.experiment_id,
            self.failed_stage,
            self.root_cause,
            self.architecture_change,
        )
        if any(not value.strip() for value in values):
            raise ValueError("foundation-pipeline experiment fields must be non-empty")
        if not self.required_evidence or not self.promotion_gates:
            raise ValueError("experiments require evidence and promotion gates")
        return {
            "experiment_id": self.experiment_id,
            "failed_stage": self.failed_stage,
            "root_cause": self.root_cause,
            "architecture_change": self.architecture_change,
            "required_evidence": list(self.required_evidence),
            "promotion_gates": list(self.promotion_gates),
            "blocked_by": list(self.blocked_by),
            "mutation_class": "architecture_not_hyperparameter",
        }


def _stage_passed(stages: Mapping[str, Any], stage: str) -> bool:
    row = stages.get(stage)
    return isinstance(row, Mapping) and bool(row.get("passed"))


def derive_foundation_pipeline_experiments(
    pipeline_report: Mapping[str, Any],
) -> dict[str, object]:
    """Compile failed evidence gates into ordered, architecture-level experiments.

    The proposals change observability or state representation.  They never
    relax an acceptance threshold, and force inference remains blocked until
    geometry and kinematics are accepted.
    """

    stages = pipeline_report.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("pipeline report requires a stages mapping")
    required = (
        "metric_camera",
        "robot_trajectory",
        "stem_centerlines",
        "contact_forces",
    )
    missing = [stage for stage in required if stage not in stages]
    if missing:
        raise ValueError(f"pipeline report is missing required stages: {missing}")

    failed = tuple(stage for stage in required if not _stage_passed(stages, stage))
    experiments: list[FoundationPipelineExperiment] = []
    if "metric_camera" in failed:
        experiments.append(
            FoundationPipelineExperiment(
                experiment_id="calibrated-metric-camera-bridge-v1",
                failed_stage="metric_camera",
                root_cause="learned metric depth has no independent absolute-scale observation",
                architecture_change=(
                    "fuse DA3/V-DPM camera proposals with RGB-D, a fiducial, or a known-length "
                    "hashed robot link in one named world frame"
                ),
                required_evidence=(
                    "synchronized metric observation",
                    "camera intrinsics and world_from_camera covariance",
                    "held-view reprojection evidence",
                ),
                promotion_gates=(
                    "absolute scale error is independently bounded",
                    "SE(3) and depth-coverage validators pass",
                    "scale-spoof attack is rejected",
                ),
            )
        )
    if "robot_trajectory" in failed:
        experiments.append(
            FoundationPipelineExperiment(
                experiment_id="exact-asset-full-q-analysis-by-synthesis-v1",
                failed_stage="robot_trajectory",
                root_cause="image-space wrist targets do not identify every URDF coordinate",
                architecture_change=(
                    "initialize from visual pose proposals, then optimize the complete G1 plus "
                    "Sharpa generalized coordinates through the exact hashed kinematic renderer"
                ),
                required_evidence=(
                    "exact G1 and bilateral Sharpa asset hashes",
                    "full joint-position trajectory with named robot_base frame",
                    "rendered silhouette, link, and fingertip reprojection residuals",
                ),
                promotion_gates=(
                    "every scalar joint is present for every frame",
                    "joint limits and velocity limits pass",
                    "held-frame render reprojection p95 is at most 8 pixels",
                    "partial-q and wrong-asset attacks are rejected",
                ),
                blocked_by=(("metric_camera",) if "metric_camera" in failed else ()),
            )
        )
    if "stem_centerlines" in failed:
        experiments.append(
            FoundationPipelineExperiment(
                experiment_id="persistent-dynamic-rod-state-v1",
                failed_stage="stem_centerlines",
                root_cause=(
                    "per-frame mask skeletons cross depth discontinuities and change visible arc length"
                ),
                architecture_change=(
                    "replace independent 2-D skeleton lifting with V-DPM/SpaTracker point identities "
                    "and a rooted Cosserat-rod state optimized jointly across visible and occluded frames"
                ),
                required_evidence=(
                    "one immutable ID for every manipulated stem",
                    "tracked surface/centerline point identities with uncertainty",
                    "metric root pose and arc-length/material prior",
                ),
                promotion_gates=(
                    "visible coverage is at least 80 percent per stem",
                    "maximum temporal segment-length CV is at most 0.12",
                    "occlusion, mask-truncation, and identity-swap attacks are rejected",
                    "the gate threshold is unchanged from the rejected baseline",
                ),
                blocked_by=(("metric_camera",) if "metric_camera" in failed else ()),
            )
        )
    if "contact_forces" in failed:
        dependencies = tuple(
            stage
            for stage in ("metric_camera", "robot_trajectory", "stem_centerlines")
            if stage in failed
        )
        experiments.append(
            FoundationPipelineExperiment(
                experiment_id="sensor-or-inverse-dynamics-force-fusion-v1",
                failed_stage="contact_forces",
                root_cause="RGB likelihood and 2-D adjacency are not observations of force",
                architecture_change=(
                    "prefer calibrated tactile/force-torque measurements; otherwise solve rod plus "
                    "MuJoCo inverse dynamics with contact complementarity and propagated covariance"
                ),
                required_evidence=(
                    "accepted metric camera, full-q robot, and per-stem rod trajectories",
                    "named contact geometry, normals, friction, material, and support reactions",
                    "sensor calibration or physics-solver residual and covariance",
                ),
                promotion_gates=(
                    "force source is sensor_measurement or physics_solver_estimate",
                    "solver residual p95 is at most 0.08 N",
                    "friction, non-penetration, and force-closure checks pass",
                    "2-D-overlap-with-zero-force and force-spoof attacks are rejected",
                ),
                blocked_by=dependencies,
            )
        )

    return {
        "input_status": str(pipeline_report.get("status", "UNKNOWN")),
        "promotable": not failed,
        "failed_stages": list(failed),
        "experiments": [experiment.to_dict() for experiment in experiments],
        "execution_order": [
            ["metric_camera"],
            ["robot_trajectory", "stem_centerlines"],
            ["contact_forces"],
            ["adversarial_full_video_render_acceptance"],
        ],
        "global_attacks": [
            "learned-depth common-mode scale spoof",
            "wrong robot asset hash and partial-q substitution",
            "stem mask truncation, identity swap, occlusion, and frozen response",
            "2-D contact adjacency with invalid 3-D depth or zero force",
            "force covariance removal and solver-residual inflation",
        ],
        "promotion_rule": (
            "promote only after every physical stage and every adversarial attack passes on "
            "a newly rendered full-length candidate; mean scores cannot override a failed gate"
        ),
    }


def select_architecture(
    assessments: Iterable[ArchitectureAssessment],
    contract: ArchitectureEvolutionContract,
) -> dict[str, object]:
    """Select only an architecture that passes every gate in every group."""

    contract.validate()
    rows = list(assessments)
    indexed: dict[tuple[str, str], ArchitectureAssessment] = {}
    for row in rows:
        row.validate(contract.required_gates)
        key = (row.group_id, row.architecture_id)
        if key in indexed:
            raise ValueError(f"duplicate tournament row: {key}")
        indexed[key] = row
    required = {
        (group, architecture)
        for group in contract.required_groups
        for architecture in contract.architecture_ids
    }
    missing = required - indexed.keys()
    extra = indexed.keys() - required
    if missing or extra:
        raise ValueError(
            f"incomplete architecture tournament; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    summaries = []
    promotable = []
    for architecture in contract.architecture_ids:
        selected = [indexed[(group, architecture)] for group in contract.required_groups]
        all_hard_gates = all(row.passed for row in selected)
        mean_cost = sum(row.cost_units for row in selected) / len(selected)
        mean_utility = sum(row.utility for row in selected) / len(selected)
        accepted = all_hard_gates and mean_cost <= contract.maximum_cost_units
        summary = {
            "architecture_id": architecture,
            "all_hard_gates_pass": all_hard_gates,
            "mean_utility": mean_utility,
            "mean_cost_units": mean_cost,
            "cost_gate_pass": mean_cost <= contract.maximum_cost_units,
            "promotable": accepted,
            "failed_gates_by_group": {
                row.group_id: [name for name, passed in row.hard_gates if not passed]
                for row in selected
                if not row.passed
            },
            "evidence": [row.evidence_path for row in selected],
        }
        summaries.append(summary)
        if accepted:
            promotable.append(summary)
    winner = (
        max(
            promotable,
            key=lambda item: (
                float(item["mean_utility"]),
                -float(item["mean_cost_units"]),
                str(item["architecture_id"]),
            ),
        )
        if promotable
        else None
    )
    return {
        "promoted": winner is not None,
        "selected_architecture": winner["architecture_id"] if winner else None,
        "architectures": summaries,
    }


def derive_structural_repairs(
    selection: dict[str, object],
) -> list[dict[str, object]]:
    """Derive architecture mutations from failed gates, never numeric sweeps.

    A repair is emitted only from evidence present in a complete tournament.
    It names the architectures and groups that failed the gate, making the
    mutation auditable and keeping it separate from promotion.
    """

    failures: dict[str, dict[str, set[str]]] = {}
    for architecture in selection.get("architectures", []):
        if not isinstance(architecture, dict):
            raise ValueError("architecture summaries must be mappings")
        architecture_id = str(architecture.get("architecture_id", ""))
        grouped = architecture.get("failed_gates_by_group", {})
        if not isinstance(grouped, dict):
            raise ValueError("failed_gates_by_group must be a mapping")
        for group_id, gates in grouped.items():
            if not isinstance(gates, list):
                raise ValueError("failed gate values must be lists")
            for gate in gates:
                gate_name = str(gate)
                item = failures.setdefault(
                    gate_name,
                    {"architectures": set(), "groups": set()},
                )
                item["architectures"].add(architecture_id)
                item["groups"].add(str(group_id))
    repairs = []
    for gate_name in sorted(failures):
        structural = _STRUCTURAL_REPAIRS.get(gate_name)
        if structural is None:
            raise ValueError(f"no architecture-level repair is registered for gate {gate_name!r}")
        repairs.append(
            {
                "failed_gate": gate_name,
                **structural,
                "failed_architectures": sorted(failures[gate_name]["architectures"]),
                "failed_groups": sorted(failures[gate_name]["groups"]),
                "mutation_class": "architecture_not_hyperparameter",
            }
        )
    return repairs
