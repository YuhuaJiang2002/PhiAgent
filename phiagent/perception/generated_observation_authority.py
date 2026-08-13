"""Authority and ensemble audits for model-completed physical observations.

Generative video and vision-language models can propose observations that are
useful for triage and weak supervision.  They cannot create a new physical
acquisition group from pixels that were already recorded.  This module keeps
that distinction explicit and has no heavyweight runtime dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class ObservationSourceKind(str, Enum):
    APRILTAG_CAPTURE = "apriltag_capture"
    RGBD_SENSOR = "rgbd_sensor"
    EXTRA_CAMERA_CAPTURE = "extra_camera_capture"
    ROBOT_TELEMETRY = "robot_telemetry"
    FORCE_OR_TACTILE_SENSOR = "force_or_tactile_sensor"
    VALIDATED_PHYSICS_SOLVER = "validated_physics_solver"
    MODEL_DERIVED_RGBD = "model_derived_rgbd"
    GENERATIVE_NOVEL_VIEW = "generative_novel_view"
    VLM_INFERENCE = "vlm_inference"
    SYNTHETIC_FIDUCIAL = "synthetic_fiducial"


PHYSICAL_CAPTURE_KINDS = {
    ObservationSourceKind.APRILTAG_CAPTURE,
    ObservationSourceKind.RGBD_SENSOR,
    ObservationSourceKind.EXTRA_CAMERA_CAPTURE,
    ObservationSourceKind.ROBOT_TELEMETRY,
    ObservationSourceKind.FORCE_OR_TACTILE_SENSOR,
}

MODEL_COMPLETION_KINDS = {
    ObservationSourceKind.MODEL_DERIVED_RGBD,
    ObservationSourceKind.GENERATIVE_NOVEL_VIEW,
    ObservationSourceKind.VLM_INFERENCE,
    ObservationSourceKind.SYNTHETIC_FIDUCIAL,
}


@dataclass(frozen=True)
class ObservationSource:
    """One immutable observation source and its physical lineage."""

    source_id: str
    kind: ObservationSourceKind
    acquisition_group_id: str | None
    source_sha256: str
    timeline: str
    coordinate_frame: str
    synchronized: bool
    physically_captured: bool
    generated_from_source_sha256: str | None = None
    metric_calibration_passed: bool = False
    exact_asset_bound: bool = False
    solver_inputs_physically_accepted: bool = False

    def validate(self) -> None:
        if not self.source_id.strip() or not self.timeline.strip():
            raise ValueError("source ID and timeline must be named")
        if not self.coordinate_frame.strip():
            raise ValueError("observation coordinate frame must be named")
        if len(self.source_sha256) != 64:
            raise ValueError("source digest must be SHA-256")
        if self.generated_from_source_sha256 is not None and len(
            self.generated_from_source_sha256
        ) != 64:
            raise ValueError("generated-source digest must be SHA-256")
        if self.kind in PHYSICAL_CAPTURE_KINDS and not self.physically_captured:
            raise ValueError("physical source kind requires a physical capture")
        if self.kind in MODEL_COMPLETION_KINDS and self.physically_captured:
            raise ValueError("model completion cannot be marked physically captured")
        if self.kind in MODEL_COMPLETION_KINDS and self.metric_calibration_passed:
            raise ValueError("model completion cannot declare metric calibration")
        if self.physically_captured and not self.acquisition_group_id:
            raise ValueError("physical capture requires an acquisition group")


def audit_observation_sources(
    sources: Sequence[ObservationSource],
) -> dict[str, object]:
    """Compute which physical gates the supplied sources may support.

    The result expresses authority, not whether the numerical quality gates pass.
    In particular, a validated solver can support the force stage only after all
    of its state inputs have independently passed their own physical contracts.
    """

    if not sources:
        raise ValueError("at least one observation source is required")
    for source in sources:
        source.validate()
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("observation source IDs must be unique")

    physical_groups = {
        str(source.acquisition_group_id)
        for source in sources
        if source.physically_captured and source.acquisition_group_id
    }
    metric_camera = any(
        source.physically_captured
        and source.synchronized
        and source.metric_calibration_passed
        and source.kind
        in {
            ObservationSourceKind.APRILTAG_CAPTURE,
            ObservationSourceKind.RGBD_SENSOR,
            ObservationSourceKind.EXTRA_CAMERA_CAPTURE,
        }
        for source in sources
    )
    robot_trajectory = any(
        source.physically_captured
        and source.synchronized
        and source.exact_asset_bound
        and source.kind is ObservationSourceKind.ROBOT_TELEMETRY
        for source in sources
    )
    measured_force = any(
        source.physically_captured
        and source.synchronized
        and source.kind is ObservationSourceKind.FORCE_OR_TACTILE_SENSOR
        for source in sources
    )
    solver_force = any(
        source.kind is ObservationSourceKind.VALIDATED_PHYSICS_SOLVER
        and source.solver_inputs_physically_accepted
        for source in sources
    )
    model_sources = [
        source.source_id for source in sources if source.kind in MODEL_COMPLETION_KINDS
    ]
    return {
        "source_count": len(sources),
        "physical_acquisition_groups": sorted(physical_groups),
        "independent_physical_group_count": len(physical_groups),
        "model_completion_sources": model_sources,
        "authority": {
            "metric_camera": metric_camera,
            "robot_trajectory": robot_trajectory,
            "contact_forces": measured_force or solver_force,
        },
        "model_completion_can_satisfy_physical_gate": False,
        "required_new_sources": {
            "metric_camera": (
                []
                if metric_camera
                else [
                    "physically captured AprilTag/known-scale calibration, RGB-D, or calibrated extra camera",
                    "a second independent calibration group for promotion",
                ]
            ),
            "robot_trajectory": (
                []
                if robot_trajectory
                else [
                    "synchronized complete q/qdot telemetry bound to the exact URDF/MJCF asset"
                ]
            ),
            "contact_forces": (
                []
                if measured_force or solver_force
                else [
                    "synchronized force/torque or tactile sensing, or a validated solver after physical state gates pass"
                ]
            ),
        },
        "reason": (
            "generated observations preserve the causal lineage of their input pixels; "
            "they add hypotheses, not independent physical measurements"
        ),
    }


_VISIBILITY = {"clear", "partial", "occluded", "absent"}
_FINGER = {"normal", "ambiguous", "deformed", "motion_blur", "not_visible"}
_CONTACT = {"none", "near", "touching", "grasping", "unknown"}
_FLOWER_MOTION = {"moving", "static", "ambiguous"}
_CAMERA_MOTION = {"static", "moving", "ambiguous"}


def validate_vlm_observation_report(
    report: Mapping[str, Any], *, expected_source_sha256: str | None = None
) -> dict[str, object]:
    """Validate a strict VLM report while refusing physical authority claims."""

    if str(report.get("evidence_class")) != "foundation_model_estimate":
        raise ValueError("VLM report must remain a foundation-model estimate")
    source_hash = str(report.get("source_video_sha256", ""))
    if len(source_hash) != 64:
        raise ValueError("VLM report requires the source-video SHA-256")
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError("VLM report is bound to a different source video")
    if report.get("independent_physical_groups") != 0:
        raise ValueError("VLM inference cannot declare an independent physical group")
    if report.get("physical_gate_eligible") is not False:
        raise ValueError("VLM inference must be explicitly ineligible for physical gates")

    model = report.get("model")
    if not isinstance(model, Mapping) or not str(model.get("name", "")).strip():
        raise ValueError("VLM model provenance is required")
    observations = report.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("VLM report requires observations")
    seen: set[int] = set()
    previous = -1
    risky = 0
    static_contact = 0
    for row in observations:
        if not isinstance(row, Mapping):
            raise ValueError("every VLM observation must be an object")
        frame = int(row["frame_index"])
        timestamp = float(row["timestamp_s"])
        if frame <= previous or frame in seen or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("VLM frames must be unique, increasing, and have finite time")
        seen.add(frame)
        previous = frame
        if row.get("left_hand_visibility") not in _VISIBILITY:
            raise ValueError("invalid left-hand visibility enum")
        if row.get("right_hand_visibility") not in _VISIBILITY:
            raise ValueError("invalid right-hand visibility enum")
        if row.get("finger_integrity") not in _FINGER:
            raise ValueError("invalid finger-integrity enum")
        if row.get("left_contact") not in _CONTACT or row.get("right_contact") not in _CONTACT:
            raise ValueError("invalid contact enum")
        if row.get("flower_motion") not in _FLOWER_MOTION:
            raise ValueError("invalid flower-motion enum")
        if row.get("camera_motion") not in _CAMERA_MOTION:
            raise ValueError("invalid camera-motion enum")
        if row.get("finger_integrity") in {"ambiguous", "deformed", "motion_blur", "not_visible"}:
            risky += 1
        contact = row.get("left_contact") in {"touching", "grasping"} or row.get(
            "right_contact"
        ) in {"touching", "grasping"}
        if contact and row.get("flower_motion") == "static":
            static_contact += 1
    non_observable = report.get("non_observable")
    required_non_observable = {
        "metric_depth",
        "absolute_camera_scale",
        "full_q_qdot",
        "contact_force",
        "force_closure",
    }
    if not isinstance(non_observable, Mapping) or any(
        non_observable.get(name) is not True for name in required_non_observable
    ):
        raise ValueError("VLM must mark all unmeasured physical quantities non-observable")
    return {
        "passed": True,
        "model": str(model["name"]),
        "frames": len(observations),
        "finger_risk_frames": risky,
        "contact_with_static_flower_frames": static_contact,
        "physical_gate_eligible": False,
    }


def audit_vlm_ensemble(reports: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Measure model agreement and produce triage intervals, never new sensors."""

    if len(reports) < 2:
        raise ValueError("ensemble audit requires at least two model reports")
    source_hash = str(reports[0].get("source_video_sha256", ""))
    validation = [
        validate_vlm_observation_report(report, expected_source_sha256=source_hash)
        for report in reports
    ]
    model_names = [str(row["model"]) for row in validation]
    if len(set(model_names)) != len(model_names):
        raise ValueError("ensemble members must have distinct model names")
    observations = [
        {int(row["frame_index"]): row for row in report["observations"]}
        for report in reports
    ]
    common = sorted(set.intersection(*(set(rows) for rows in observations)))
    if not common:
        raise ValueError("ensemble reports have no common sampled frames")
    agreement_fields = ("left_contact", "right_contact", "flower_motion", "finger_integrity")
    agreements = 0
    comparisons = 0
    review_frames = []
    for frame in common:
        rows = [model_rows[frame] for model_rows in observations]
        frame_disagreement = False
        for field in agreement_fields:
            values = {str(row[field]) for row in rows}
            comparisons += 1
            agreements += len(values) == 1
            frame_disagreement = frame_disagreement or len(values) > 1
        risky = any(
            row["finger_integrity"] in {"ambiguous", "deformed", "motion_blur", "not_visible"}
            for row in rows
        )
        static_contact = any(
            (
                row["left_contact"] in {"touching", "grasping"}
                or row["right_contact"] in {"touching", "grasping"}
            )
            and row["flower_motion"] == "static"
            for row in rows
        )
        if frame_disagreement or risky or static_contact:
            review_frames.append(
                {
                    "frame_index": frame,
                    "timestamp_s": float(rows[0]["timestamp_s"]),
                    "model_disagreement": frame_disagreement,
                    "finger_risk": risky,
                    "contact_with_static_flower": static_contact,
                }
            )
    return {
        "passed": True,
        "source_video_sha256": source_hash,
        "models": model_names,
        "common_frames": len(common),
        "categorical_agreement_fraction": agreements / comparisons,
        "review_frames": review_frames,
        "review_frame_fraction": len(review_frames) / len(common),
        "evidence_class": "foundation_model_estimate",
        "independent_physical_groups": 0,
        "physical_gate_eligible": False,
        "per_model_validation": validation,
        "recommended_use": [
            "failure-window mining",
            "weak-label curriculum construction",
            "active acquisition targeting",
        ],
        "forbidden_use": [
            "metric camera calibration",
            "full q/qdot telemetry",
            "3-D force closure",
            "contact-force acceptance",
        ],
    }
