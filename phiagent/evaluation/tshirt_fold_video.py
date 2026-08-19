"""Camera-frame hard gates for generated T-shirt folding videos.

OpenCV is imported only inside the extraction entry point.  Package import and
the pure scoring path stay dependency-light and CPU-only.  These image-space
gates reject visual material violations; they do not establish metric cloth
geometry, force, or robot feasibility.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from phiagent.evaluation.cloth_conservation import (
    Point2D,
    SleeveLengthScore,
    SleeveLengthThresholds,
    SleeveObservation,
    score_sleeve_length_conservation,
)


@dataclass(frozen=True)
class FrameWindow:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("frame window must be non-negative and ordered")

    def contains(self, frame_index: int) -> bool:
        return self.start <= frame_index < self.end


@dataclass(frozen=True)
class TshirtFoldTrackingThresholds:
    sleeve: SleeveLengthThresholds = SleeveLengthThresholds()
    minimum_point_confidence: float = 0.55
    minimum_motion_pixels: float = 2.0
    maximum_material_step_pixels: float = 16.0
    maximum_terminal_bbox_area_ratio: float = 0.62
    minimum_bundle_move_left_pixels: float = 24.0
    maximum_contact_distance_pixels: float = 48.0
    minimum_first_frame_score: float = 0.985
    minimum_background_score: float = 0.94

    def __post_init__(self) -> None:
        for name in (
            "minimum_point_confidence",
            "maximum_terminal_bbox_area_ratio",
            "minimum_first_frame_score",
            "minimum_background_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in (
            "minimum_motion_pixels",
            "maximum_material_step_pixels",
            "minimum_bundle_move_left_pixels",
            "maximum_contact_distance_pixels",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TshirtFoldTrackingContract:
    coordinate_frame: str
    frame_count: int
    viewer_left_sleeve_xy: tuple[Point2D, ...]
    viewer_right_sleeve_xy: tuple[Point2D, ...]
    body_points_xy: tuple[Point2D, ...]
    background_rectangles_xywh: tuple[tuple[int, int, int, int], ...]
    left_fold: FrameWindow
    right_fold: FrameWindow
    body_fold: FrameWindow
    bundle_move: FrameWindow
    lower_left_gripper_xy: tuple[Point2D, ...] = ()
    upper_right_gripper_xy: tuple[Point2D, ...] = ()
    thresholds: TshirtFoldTrackingThresholds = TshirtFoldTrackingThresholds()

    def __post_init__(self) -> None:
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("T-shirt tracking requires a named camera frame")
        if self.frame_count <= 2:
            raise ValueError("T-shirt tracking requires at least three frames")
        for name, points in (
            ("viewer_left_sleeve_xy", self.viewer_left_sleeve_xy),
            ("viewer_right_sleeve_xy", self.viewer_right_sleeve_xy),
            ("body_points_xy", self.body_points_xy),
        ):
            if len(points) < 3:
                raise ValueError(f"{name} requires at least three material points")
            if any(not all(math.isfinite(value) for value in point) for point in points):
                raise ValueError(f"{name} points must be finite")
        for name, points in (
            ("lower_left_gripper_xy", self.lower_left_gripper_xy),
            ("upper_right_gripper_xy", self.upper_right_gripper_xy),
        ):
            if points and len(points) < 2:
                raise ValueError(f"{name} requires at least two tracked points")
            if any(not all(math.isfinite(value) for value in point) for point in points):
                raise ValueError(f"{name} points must be finite")
        if bool(self.lower_left_gripper_xy) != bool(self.upper_right_gripper_xy):
            raise ValueError("both manipulator point sets must be present or absent")
        if not self.background_rectangles_xywh:
            raise ValueError("T-shirt tracking requires frozen background rectangles")
        if any(
            x < 0 or y < 0 or width <= 0 or height <= 0
            for x, y, width, height in self.background_rectangles_xywh
        ):
            raise ValueError("background rectangles must be positive and non-negative")
        windows = (self.left_fold, self.right_fold, self.body_fold, self.bundle_move)
        if any(window.end > self.frame_count for window in windows):
            raise ValueError("a phase window lies outside the video")
        if not all(left.end <= right.start for left, right in zip(windows, windows[1:])):
            raise ValueError("T-shirt phase windows must be ordered and non-overlapping")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TshirtFoldTrackingContract:
        windows = payload.get("phase_windows")
        points = payload.get("material_points")
        thresholds = payload.get("thresholds", {})
        if not isinstance(windows, dict) or not isinstance(points, dict):
            raise ValueError("tracking contract requires phase_windows and material_points")
        if not isinstance(thresholds, dict):
            raise ValueError("tracking contract thresholds must be an object")
        sleeve_thresholds = thresholds.get("sleeve", {})
        if not isinstance(sleeve_thresholds, dict):
            raise ValueError("tracking contract sleeve thresholds must be an object")

        def point_tuple(value: object) -> tuple[Point2D, ...]:
            if not isinstance(value, list):
                raise ValueError("material points must be arrays")
            return tuple((float(item[0]), float(item[1])) for item in value)

        def window(name: str) -> FrameWindow:
            value = windows[name]
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"phase window {name} must be [start, end]")
            return FrameWindow(int(value[0]), int(value[1]))

        rectangles = payload.get("background_rectangles_xywh")
        if not isinstance(rectangles, list):
            raise ValueError("background_rectangles_xywh must be an array")
        manipulators = payload.get("manipulator_points", {})
        if not isinstance(manipulators, dict):
            raise ValueError("manipulator_points must be an object")
        return cls(
            coordinate_frame=str(payload["coordinate_frame"]),
            frame_count=int(payload["frame_count"]),
            viewer_left_sleeve_xy=point_tuple(points["viewer_left_sleeve"]),
            viewer_right_sleeve_xy=point_tuple(points["viewer_right_sleeve"]),
            body_points_xy=point_tuple(points["body"]),
            background_rectangles_xywh=tuple(
                tuple(int(component) for component in item) for item in rectangles
            ),
            left_fold=window("viewer_left_sleeve_fold"),
            right_fold=window("viewer_right_sleeve_fold"),
            body_fold=window("body_fold"),
            bundle_move=window("bundle_move"),
            lower_left_gripper_xy=point_tuple(manipulators["lower_left_gripper"])
            if "lower_left_gripper" in manipulators
            else (),
            upper_right_gripper_xy=point_tuple(manipulators["upper_right_gripper"])
            if "upper_right_gripper" in manipulators
            else (),
            thresholds=TshirtFoldTrackingThresholds(
                sleeve=SleeveLengthThresholds(**sleeve_thresholds),
                **{
                    key: value
                    for key, value in thresholds.items()
                    if key != "sleeve"
                },
            ),
        )


def load_tshirt_fold_tracking_contract(path: Path) -> TshirtFoldTrackingContract:
    payload = json.loads(path.expanduser().resolve().read_text())
    if not isinstance(payload, dict):
        raise ValueError("tracking contract must contain one JSON object")
    return TshirtFoldTrackingContract.from_dict(payload)


@dataclass(frozen=True)
class TrackedMaterialFrame:
    frame_index: int
    viewer_left_sleeve_xy: tuple[Point2D, ...]
    viewer_right_sleeve_xy: tuple[Point2D, ...]
    body_points_xy: tuple[Point2D, ...]
    confidence: float
    lower_left_gripper_xy: tuple[Point2D, ...] = ()
    upper_right_gripper_xy: tuple[Point2D, ...] = ()
    manipulator_confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("tracked material frame index must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("tracked material confidence must be in [0, 1]")
        if not 0.0 <= self.manipulator_confidence <= 1.0:
            raise ValueError("manipulator confidence must be in [0, 1]")


@dataclass(frozen=True)
class TshirtFoldVideoScore:
    sleeve_scores: Mapping[str, SleeveLengthScore]
    gate_results: Mapping[str, bool]
    first_frame_score: float
    background_score: float
    maximum_material_step_pixels: float
    terminal_bbox_area_ratio: float
    terminal_bundle_move_left_pixels: float
    motion_onsets: Mapping[str, int | None]

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, passed in self.gate_results.items() if not passed))

    @property
    def hard_gates_passed(self) -> bool:
        return not self.failed_gates

    def to_dict(self) -> dict[str, object]:
        return {
            "sleeve_scores": {
                name: score.to_dict() for name, score in self.sleeve_scores.items()
            },
            "gate_results": dict(self.gate_results),
            "hard_gates_passed": self.hard_gates_passed,
            "failed_gates": list(self.failed_gates),
            "first_frame_score": self.first_frame_score,
            "background_score": self.background_score,
            "maximum_material_step_pixels": self.maximum_material_step_pixels,
            "terminal_bbox_area_ratio": self.terminal_bbox_area_ratio,
            "terminal_bundle_move_left_pixels": self.terminal_bundle_move_left_pixels,
            "motion_onsets": dict(self.motion_onsets),
        }


def _point_distance(left: Point2D, right: Point2D) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _mean_step(previous: Sequence[Point2D], current: Sequence[Point2D]) -> float:
    if len(previous) != len(current):
        raise ValueError("material tracks require corresponding point counts")
    return fmean(_point_distance(left, right) for left, right in zip(previous, current))


def _motion_onset(values: Sequence[float], minimum_motion_pixels: float) -> int | None:
    return next(
        (index for index, value in enumerate(values, start=1) if value >= minimum_motion_pixels),
        None,
    )


def _bbox_area(points: Sequence[Point2D]) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(max(xs) - min(xs), 1e-6) * max(max(ys) - min(ys), 1e-6)


def _centroid_x(points: Sequence[Point2D]) -> float:
    return fmean(point[0] for point in points)


def _minimum_set_distance(left: Sequence[Point2D], right: Sequence[Point2D]) -> float:
    if not left or not right:
        return math.inf
    return min(_point_distance(a, b) for a in left for b in right)


def _contact_before_onset(
    frames: Sequence[TrackedMaterialFrame],
    *,
    sleeve: str,
    onset: int | None,
    threshold: float,
    minimum_confidence: float,
) -> bool:
    if onset is None:
        return False
    first = max(0, onset - 2)
    for frame in frames[first : onset + 1]:
        if frame.manipulator_confidence < minimum_confidence:
            continue
        sleeve_points = getattr(frame, f"{sleeve}_sleeve_xy")
        if any(
            _minimum_set_distance(gripper, sleeve_points) <= threshold
            for gripper in (
                frame.lower_left_gripper_xy,
                frame.upper_right_gripper_xy,
            )
        ):
            return True
    return False


def score_tshirt_fold_tracks(
    frames: Sequence[TrackedMaterialFrame],
    *,
    contract: TshirtFoldTrackingContract,
    first_frame_score: float,
    background_score: float,
) -> TshirtFoldVideoScore:
    if len(frames) != contract.frame_count:
        raise ValueError("tracked frame count does not match the contract")
    if tuple(frame.frame_index for frame in frames) != tuple(range(contract.frame_count)):
        raise ValueError("tracked frames must be contiguous from frame zero")
    thresholds = contract.thresholds
    left_observations = tuple(
        SleeveObservation(
            frame_index=frame.frame_index,
            polyline_xy=frame.viewer_left_sleeve_xy,
            confidence=frame.confidence,
            visible=frame.confidence >= thresholds.minimum_point_confidence,
        )
        for frame in frames
    )
    right_observations = tuple(
        SleeveObservation(
            frame_index=frame.frame_index,
            polyline_xy=frame.viewer_right_sleeve_xy,
            confidence=frame.confidence,
            visible=frame.confidence >= thresholds.minimum_point_confidence,
        )
        for frame in frames
    )
    sleeve_scores = {
        "viewer_left": score_sleeve_length_conservation(
            "viewer_left",
            left_observations,
            expected_frames=contract.frame_count,
            thresholds=thresholds.sleeve,
        ),
        "viewer_right": score_sleeve_length_conservation(
            "viewer_right",
            right_observations,
            expected_frames=contract.frame_count,
            thresholds=thresholds.sleeve,
        ),
    }
    left_steps = tuple(
        _mean_step(previous.viewer_left_sleeve_xy, current.viewer_left_sleeve_xy)
        for previous, current in zip(frames, frames[1:])
    )
    right_steps = tuple(
        _mean_step(previous.viewer_right_sleeve_xy, current.viewer_right_sleeve_xy)
        for previous, current in zip(frames, frames[1:])
    )
    body_steps = tuple(
        _mean_step(previous.body_points_xy, current.body_points_xy)
        for previous, current in zip(frames, frames[1:])
    )
    left_onset = _motion_onset(left_steps, thresholds.minimum_motion_pixels)
    right_onset = _motion_onset(right_steps, thresholds.minimum_motion_pixels)
    body_onset = _motion_onset(body_steps, thresholds.minimum_motion_pixels)
    left_contact = _contact_before_onset(
        frames,
        sleeve="viewer_left",
        onset=left_onset,
        threshold=thresholds.maximum_contact_distance_pixels,
        minimum_confidence=thresholds.minimum_point_confidence,
    )
    right_contact = _contact_before_onset(
        frames,
        sleeve="viewer_right",
        onset=right_onset,
        threshold=thresholds.maximum_contact_distance_pixels,
        minimum_confidence=thresholds.minimum_point_confidence,
    )
    maximum_step = max((*left_steps, *right_steps, *body_steps))
    initial_points = (
        *frames[0].viewer_left_sleeve_xy,
        *frames[0].viewer_right_sleeve_xy,
        *frames[0].body_points_xy,
    )
    terminal_points = (
        *frames[-1].viewer_left_sleeve_xy,
        *frames[-1].viewer_right_sleeve_xy,
        *frames[-1].body_points_xy,
    )
    terminal_bbox_ratio = _bbox_area(terminal_points) / _bbox_area(initial_points)
    bundle_reference = frames[contract.bundle_move.start].body_points_xy
    bundle_move_left = _centroid_x(bundle_reference) - _centroid_x(frames[-1].body_points_xy)

    left_in_window = left_onset is not None and contract.left_fold.contains(left_onset)
    right_in_window = right_onset is not None and contract.right_fold.contains(right_onset)
    body_after_sleeves = body_onset is not None and body_onset >= contract.body_fold.start
    left_before_right = (
        left_onset is not None
        and right_onset is not None
        and left_onset < right_onset
        and left_in_window
        and right_in_window
    )
    gate_results = {
        "exact_first_frame": first_frame_score >= thresholds.minimum_first_frame_score,
        "viewer_left_sleeve_length_conserved": sleeve_scores["viewer_left"].passed,
        "viewer_right_sleeve_length_conserved": sleeve_scores["viewer_right"].passed,
        "viewer_left_fold_precedes_viewer_right_fold": left_before_right,
        "contact_precedes_cloth_motion": left_contact and right_contact,
        "no_teleportation_or_crossfade": (
            maximum_step <= thresholds.maximum_material_step_pixels
        ),
        "body_fold_after_both_sleeves": body_after_sleeves,
        "bundle_move_after_body_fold": (
            bundle_move_left >= thresholds.minimum_bundle_move_left_pixels
        ),
        "camera_and_background_static": (
            background_score >= thresholds.minimum_background_score
        ),
        "terminal_compact_bundle_stable": (
            terminal_bbox_ratio <= thresholds.maximum_terminal_bbox_area_ratio
        ),
    }
    return TshirtFoldVideoScore(
        sleeve_scores=sleeve_scores,
        gate_results=gate_results,
        first_frame_score=first_frame_score,
        background_score=background_score,
        maximum_material_step_pixels=maximum_step,
        terminal_bbox_area_ratio=terminal_bbox_ratio,
        terminal_bundle_move_left_pixels=bundle_move_left,
        motion_onsets={
            "viewer_left_sleeve": left_onset,
            "viewer_right_sleeve": right_onset,
            "body": body_onset,
        },
    )


def _sample_flow(flow: Any, point: Point2D) -> Point2D:
    height, width = flow.shape[:2]
    x = min(max(int(round(point[0])), 0), width - 1)
    y = min(max(int(round(point[1])), 0), height - 1)
    return float(flow[y, x, 0]), float(flow[y, x, 1])


def _advect_points(points: Sequence[Point2D], forward: Any, backward: Any) -> tuple[tuple[Point2D, ...], float]:
    height, width = forward.shape[:2]
    updated: list[Point2D] = []
    confidences: list[float] = []
    for point in points:
        dx, dy = _sample_flow(forward, point)
        candidate = point[0] + dx, point[1] + dy
        inside = 0 <= candidate[0] < width and 0 <= candidate[1] < height
        bdx, bdy = _sample_flow(backward, candidate)
        residual = math.hypot(dx + bdx, dy + bdy)
        confidence = math.exp(-residual / 3.0) if inside else 0.0
        updated.append(candidate)
        confidences.append(confidence)
    return tuple(updated), min(confidences, default=0.0)


def _normalized_frame_similarity(reference: Any, candidate: Any) -> float:
    import numpy as np

    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return max(0.0, min(1.0, 1.0 - float(difference.mean()) / 255.0))


def extract_and_score_tshirt_fold_video(
    candidate: Path,
    first_frame: Path,
    *,
    contract: TshirtFoldTrackingContract,
) -> TshirtFoldVideoScore:
    """Track the frozen material points and apply all automatic hard gates."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "T-shirt video hard gates require the optional OpenCV and NumPy adapter"
        ) from exc

    capture = cv2.VideoCapture(str(candidate.expanduser().resolve()))
    decoded: list[Any] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    if len(decoded) != contract.frame_count:
        raise ValueError(
            f"candidate decoded {len(decoded)} frames, expected {contract.frame_count}"
        )
    reference = cv2.imread(str(first_frame.expanduser().resolve()), cv2.IMREAD_COLOR)
    if reference is None:
        raise ValueError("failed to decode the supplied first frame")
    if reference.shape != decoded[0].shape:
        raise ValueError("candidate and first frame dimensions differ")
    first_frame_score = _normalized_frame_similarity(reference, decoded[0])
    background_scores: list[float] = []
    for frame in decoded:
        patches = []
        for x, y, width, height in contract.background_rectangles_xywh:
            patch = frame[y : y + height, x : x + width]
            reference_patch = reference[y : y + height, x : x + width]
            if patch.size == 0 or patch.shape != reference_patch.shape:
                raise ValueError("background rectangle lies outside the decoded video")
            patches.append(_normalized_frame_similarity(reference_patch, patch))
        background_scores.append(fmean(patches))
    background_score = min(background_scores)

    left_points = contract.viewer_left_sleeve_xy
    right_points = contract.viewer_right_sleeve_xy
    body_points = contract.body_points_xy
    lower_left_gripper = contract.lower_left_gripper_xy
    upper_right_gripper = contract.upper_right_gripper_xy
    tracked = [
        TrackedMaterialFrame(
            frame_index=0,
            viewer_left_sleeve_xy=left_points,
            viewer_right_sleeve_xy=right_points,
            body_points_xy=body_points,
            confidence=1.0,
            lower_left_gripper_xy=lower_left_gripper,
            upper_right_gripper_xy=upper_right_gripper,
            manipulator_confidence=1.0 if lower_left_gripper else 0.0,
        )
    ]
    previous_gray = cv2.cvtColor(decoded[0], cv2.COLOR_BGR2GRAY)
    for frame_index, frame in enumerate(decoded[1:], start=1):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        forward = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            5,
            25,
            5,
            7,
            1.5,
            0,
        )
        backward = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous_gray,
            None,
            0.5,
            5,
            25,
            5,
            7,
            1.5,
            0,
        )
        left_points, left_confidence = _advect_points(left_points, forward, backward)
        right_points, right_confidence = _advect_points(right_points, forward, backward)
        body_points, body_confidence = _advect_points(body_points, forward, backward)
        lower_left_gripper, lower_confidence = _advect_points(
            lower_left_gripper, forward, backward
        )
        upper_right_gripper, upper_confidence = _advect_points(
            upper_right_gripper, forward, backward
        )
        confidence = min(left_confidence, right_confidence, body_confidence)
        tracked.append(
            TrackedMaterialFrame(
                frame_index=frame_index,
                viewer_left_sleeve_xy=left_points,
                viewer_right_sleeve_xy=right_points,
                body_points_xy=body_points,
                confidence=confidence,
                lower_left_gripper_xy=lower_left_gripper,
                upper_right_gripper_xy=upper_right_gripper,
                manipulator_confidence=min(lower_confidence, upper_confidence),
            )
        )
        previous_gray = current_gray
    return score_tshirt_fold_tracks(
        tracked,
        contract=contract,
        first_frame_score=first_frame_score,
        background_score=background_score,
    )


def write_tshirt_fold_evidence(path: Path, score: TshirtFoldVideoScore) -> None:
    payload = {
        "schema_version": "1.0.0",
        "evidence_boundary": (
            "Camera-frame optical-flow material tracks and image similarity only; no metric 3-D "
            "cloth geometry, force, safety, robot trajectory, or physical success is established."
        ),
        **score.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
