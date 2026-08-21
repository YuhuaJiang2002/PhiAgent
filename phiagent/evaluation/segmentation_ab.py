"""Model-neutral scoring and comparison for video segmentation A/B runs.

This module deliberately imports no model, CUDA, OpenCV, or NumPy dependency at
module import time. Heavy dependencies remain confined to the executable
adapters that need them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from statistics import median
import subprocess
from typing import Any, Mapping, Sequence

from phiagent.evaluation.video_proxy import file_sha256


AB_SCHEMA_VERSION = "1.0.0"
SAM2_MODEL_ID = "sam2"
SAM31_MODEL_ID = "sam3.1"
SUPPORTED_MODEL_IDS = (SAM2_MODEL_ID, SAM31_MODEL_ID)
_OBJECT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


@dataclass(frozen=True)
class MaskGeometryThresholds:
    """Frozen geometry thresholds for one evaluator epoch."""

    baseline_hold_frames: int
    minimum_area_pixels: int
    minimum_component_area_pixels: int
    maximum_connected_components: int
    maximum_major_axis_cv: float
    maximum_major_axis_relative_deviation: float
    maximum_terminal_major_axis_relative_deviation: float
    maximum_area_cv: float
    maximum_area_relative_deviation: float
    maximum_terminal_area_relative_deviation: float
    minimum_component_area_fraction: float = 0.0
    maximum_centroid_step_pixels: float = 90.0

    def __post_init__(self) -> None:
        positive_integers = (
            self.baseline_hold_frames,
            self.minimum_area_pixels,
            self.minimum_component_area_pixels,
            self.maximum_connected_components,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("mask area and component thresholds must be positive")
        for name in (
            "maximum_major_axis_cv",
            "maximum_major_axis_relative_deviation",
            "maximum_terminal_major_axis_relative_deviation",
            "maximum_area_cv",
            "maximum_area_relative_deviation",
            "maximum_terminal_area_relative_deviation",
            "maximum_centroid_step_pixels",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.minimum_component_area_fraction)
            or not 0.0 <= self.minimum_component_area_fraction < 1.0
        ):
            raise ValueError("minimum_component_area_fraction must be finite and in [0, 1)")


@dataclass(frozen=True)
class TrackerSpec:
    """Pinned source and checkpoint identity for one tracking backend."""

    model_id: str
    repository: Path
    repository_commit: str
    checkpoint: Path
    checkpoint_sha256: str
    evaluator_epoch_id: str
    role: str
    model_config: str | None = None
    multiplex_count: int | None = None
    compile_model: bool = False
    offload_video_to_cpu: bool = False
    offload_state_to_cpu: bool = False

    def public_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["repository"] = str(self.repository)
        payload["checkpoint"] = str(self.checkpoint)
        return payload


def load_json_object(path: Path, label: str) -> dict[str, object]:
    """Read a JSON object with a precise type error."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def resolve_existing(root: Path, value: object, label: str) -> Path:
    """Resolve a required repository-relative or absolute path."""

    path = Path(str(value)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def validate_sha256(value: object, label: str) -> str:
    """Validate a lowercase SHA-256 string."""

    digest = str(value)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def validate_revision(value: object, label: str) -> str:
    """Validate a full SHA-1 or SHA-256 source revision."""

    revision = str(value)
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{label} must be a full lowercase Git-style revision")
    return revision


def validate_repository_revision(repository: Path, expected: str, label: str) -> str:
    """Fail closed when an external model checkout is not at its pinned commit."""

    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected:
        raise RuntimeError(f"{label} repository revision {revision} differs from pinned {expected}")
    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        changed_paths = [line[3:] for line in status.splitlines()[:5]]
        raise RuntimeError(
            f"{label} repository has uncommitted content: " + ", ".join(changed_paths)
        )
    return revision


def parse_tracker_spec(
    raw: Mapping[str, object],
    *,
    project_root: Path,
    model_id: str,
) -> TrackerSpec:
    """Parse and validate one model adapter's immutable runtime contract."""

    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"unsupported model_id: {model_id}")
    configured_model_id = str(raw.get("model_id", model_id))
    if configured_model_id != model_id:
        raise ValueError(f"tracker model_id {configured_model_id!r} does not match {model_id!r}")
    role = str(raw.get("role", ""))
    expected_role = "authoritative" if model_id == SAM2_MODEL_ID else "shadow"
    if role != expected_role:
        raise ValueError(f"{model_id} role must be {expected_role!r}")
    repository = resolve_existing(project_root, raw.get("repository"), f"{model_id} repository")
    checkpoint = resolve_existing(project_root, raw.get("checkpoint"), f"{model_id} checkpoint")
    repository_commit = validate_revision(
        raw.get("repository_commit"), f"{model_id} repository_commit"
    )
    checkpoint_sha256 = validate_sha256(
        raw.get("checkpoint_sha256"), f"{model_id} checkpoint_sha256"
    )
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError(f"{model_id} checkpoint SHA-256 differs from the config")
    validate_repository_revision(repository, repository_commit, model_id)
    evaluator_epoch_id = str(raw.get("evaluator_epoch_id", "")).strip()
    if not evaluator_epoch_id:
        raise ValueError(f"{model_id} evaluator_epoch_id is required")

    model_config: str | None = None
    multiplex_count: int | None = None
    if model_id == SAM2_MODEL_ID:
        model_config = str(raw.get("config", "")).strip()
        if not model_config:
            raise ValueError("sam2 config is required")
    else:
        multiplex_count = int(raw.get("multiplex_count", 16))
        if multiplex_count < 2:
            raise ValueError("sam3.1 multiplex_count must be at least two")

    return TrackerSpec(
        model_id=model_id,
        repository=repository,
        repository_commit=repository_commit,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        evaluator_epoch_id=evaluator_epoch_id,
        role=role,
        model_config=model_config,
        multiplex_count=multiplex_count,
        compile_model=raw.get("compile") is True,
        offload_video_to_cpu=raw.get("offload_video_to_cpu") is True,
        offload_state_to_cpu=raw.get("offload_state_to_cpu") is True,
    )


def validate_task_config(config: Mapping[str, object]) -> MaskGeometryThresholds:
    """Validate the shared task and incumbent evaluator contract."""

    if config.get("schema_version") != AB_SCHEMA_VERSION:
        raise ValueError(f"task config must use schema_version {AB_SCHEMA_VERSION}")
    policy = config.get("ab_policy")
    if not isinstance(policy, dict):
        raise ValueError("task config requires ab_policy")
    expected_policy = {
        "authoritative_model": SAM2_MODEL_ID,
        "shadow_model": SAM31_MODEL_ID,
        "decision_mode": "sam2_authoritative_sam31_shadow",
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"ab_policy.{key} must be {expected!r}")
    if policy.get("reuse_incumbent_thresholds_for_shadow") is not False:
        raise ValueError("SAM3.1 must not reuse incumbent SAM2 thresholds for decisions")

    objects = config.get("objects")
    if not isinstance(objects, dict) or len(objects) != 2:
        raise ValueError("task config requires exactly two tracked objects")
    object_ids: set[int] = set()
    for name, raw in objects.items():
        if not _OBJECT_NAME_PATTERN.fullmatch(str(name)):
            raise ValueError(f"invalid object name: {name!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"object {name!r} must be a JSON object")
        object_id = int(raw.get("object_id", 0))
        if object_id < 1 or object_id in object_ids:
            raise ValueError("object IDs must be unique positive integers")
        object_ids.add(object_id)
        polygon = raw.get("initial_polygon_xy")
        if (
            not isinstance(polygon, list)
            or len(polygon) < 3
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(type(value) is not int for value in point)
                for point in polygon
            )
        ):
            raise ValueError(f"object {name!r} requires an integer polygon")

    frame_count = config.get("frame_count")
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    initial_frame_index = config.get("initial_frame_index", 0)
    if type(initial_frame_index) is not int or not 0 <= initial_frame_index < frame_count:
        raise ValueError("initial_frame_index lies outside the video")
    frame_size = config.get("frame_size", [1248, 720])
    if (
        not isinstance(frame_size, list)
        or len(frame_size) != 2
        or any(type(value) is not int or value < 1 for value in frame_size)
    ):
        raise ValueError("frame_size must contain two positive integers")
    fps = config.get("fps", 24.0)
    if type(fps) not in {int, float} or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(config.get("sam2"), dict):
        raise ValueError("task config requires a sam2 tracker block")
    refinement = config.get("initial_mask_refinement")
    if not isinstance(refinement, dict):
        raise ValueError("task config requires initial_mask_refinement")
    maximum_luma = refinement.get("maximum_luma")
    closing_kernel = refinement.get("closing_kernel_pixels")
    minimum_area = refinement.get("minimum_area_pixels")
    if (
        type(maximum_luma) is not int
        or not 0 <= maximum_luma <= 255
        or type(closing_kernel) is not int
        or closing_kernel < 1
        or closing_kernel % 2 == 0
        or type(minimum_area) is not int
        or minimum_area < 1
        or refinement.get("keep_largest_component") is not True
    ):
        raise ValueError("invalid initial-mask refinement settings")
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("task config requires thresholds")
    return MaskGeometryThresholds(**thresholds)


def validate_sam31_config(config: Mapping[str, object]) -> None:
    """Validate fields that do not require access to model assets."""

    if config.get("schema_version") != AB_SCHEMA_VERSION:
        raise ValueError(f"SAM3.1 config must use schema_version {AB_SCHEMA_VERSION}")
    if config.get("model_id") != SAM31_MODEL_ID:
        raise ValueError("SAM3.1 config model_id must be 'sam3.1'")
    if config.get("role") != "shadow":
        raise ValueError("SAM3.1 must remain a shadow model in this evaluator epoch")
    if config.get("thresholds") is not None:
        raise ValueError("the initial SAM3.1 shadow epoch must not contain decision thresholds")
    multiplex_count = config.get("multiplex_count")
    if type(multiplex_count) is not int or multiplex_count < 2:
        raise ValueError("SAM3.1 multiplex_count must be at least two")
    validate_revision(config.get("repository_commit"), "SAM3.1 repository_commit")
    validate_sha256(config.get("checkpoint_sha256"), "SAM3.1 checkpoint_sha256")
    validate_revision(config.get("huggingface_revision"), "SAM3.1 Hugging Face revision")


def score_mask_geometry(
    areas: list[int],
    major_axes: list[float],
    components: list[int],
    *,
    thresholds: MaskGeometryThresholds,
) -> dict[str, object]:
    """Apply the frozen geometry gates to one mask track."""

    if not areas or len(areas) != len(major_axes) or len(areas) != len(components):
        raise ValueError("mask geometry series must be non-empty and aligned")
    if (
        any(area < 0 for area in areas)
        or any(axis < 0 for axis in major_axes)
        or any(component < 0 for component in components)
    ):
        raise ValueError("mask geometry values must be non-negative")
    if thresholds.baseline_hold_frames > len(areas):
        raise ValueError("baseline_hold_frames lies outside the mask series")
    baseline_area = float(median(areas[: thresholds.baseline_hold_frames]))
    baseline_axis = float(median(major_axes[: thresholds.baseline_hold_frames]))
    if baseline_area <= 0 or baseline_axis <= 0:
        raise ValueError("baseline mask geometry must be positive")
    mean_area = sum(areas) / len(areas)
    mean_axis = sum(major_axes) / len(major_axes)
    area_cv = math.sqrt(sum((value - mean_area) ** 2 for value in areas) / len(areas)) / mean_area
    axis_cv = (
        math.sqrt(sum((value - mean_axis) ** 2 for value in major_axes) / len(major_axes))
        / mean_axis
    )
    area_relative = [abs(value / baseline_area - 1.0) for value in areas]
    axis_relative = [abs(value / baseline_axis - 1.0) for value in major_axes]
    gates = {
        "mask_persistent": min(areas) >= thresholds.minimum_area_pixels,
        "mask_connected": min(components) >= 1
        and max(components) <= thresholds.maximum_connected_components,
        "major_axis_cv": axis_cv <= thresholds.maximum_major_axis_cv,
        "major_axis_relative_deviation": max(axis_relative)
        <= thresholds.maximum_major_axis_relative_deviation,
        "terminal_major_axis_relative_deviation": axis_relative[-1]
        <= thresholds.maximum_terminal_major_axis_relative_deviation,
        "area_cv": area_cv <= thresholds.maximum_area_cv,
        "area_relative_deviation": max(area_relative) <= thresholds.maximum_area_relative_deviation,
        "terminal_area_relative_deviation": area_relative[-1]
        <= thresholds.maximum_terminal_area_relative_deviation,
    }
    return {
        "passed": all(gates.values()),
        "gate_results": gates,
        "baseline_area_pixels": baseline_area,
        "baseline_hold_frames": thresholds.baseline_hold_frames,
        "minimum_area_pixels": min(areas),
        "maximum_area_pixels": max(areas),
        "area_cv": area_cv,
        "maximum_area_relative_deviation": max(area_relative),
        "terminal_area_relative_deviation": area_relative[-1],
        "baseline_major_axis_pixels": baseline_axis,
        "minimum_major_axis_pixels": min(major_axes),
        "maximum_major_axis_pixels": max(major_axes),
        "major_axis_cv": axis_cv,
        "maximum_major_axis_relative_deviation": max(axis_relative),
        "terminal_major_axis_relative_deviation": axis_relative[-1],
        "maximum_connected_components": max(components),
    }


def effective_component_area_threshold(
    baseline_area_pixels: float,
    thresholds: MaskGeometryThresholds,
) -> int:
    """Combine the absolute and scale-relative fragment thresholds."""

    if not math.isfinite(baseline_area_pixels) or baseline_area_pixels <= 0:
        raise ValueError("baseline_area_pixels must be finite and positive")
    return max(
        thresholds.minimum_component_area_pixels,
        math.ceil(baseline_area_pixels * thresholds.minimum_component_area_fraction),
    )


def score_centroid_continuity(
    centroids_xy: list[tuple[float, float]],
    *,
    maximum_step_pixels: float,
) -> dict[str, object]:
    """Reject frame-to-frame mask teleportation."""

    if (
        not centroids_xy
        or not math.isfinite(maximum_step_pixels)
        or maximum_step_pixels < 0
        or any(
            len(point) != 2 or any(not math.isfinite(value) for value in point)
            for point in centroids_xy
        )
    ):
        raise ValueError("invalid centroid series or maximum step")
    steps = [
        math.dist(previous, current) for previous, current in zip(centroids_xy, centroids_xy[1:])
    ]
    maximum_observed = max(steps, default=0.0)
    return {
        "passed": maximum_observed <= maximum_step_pixels,
        "maximum_allowed_step_pixels": maximum_step_pixels,
        "maximum_observed_step_pixels": maximum_observed,
        "maximum_step_frame": (1 + steps.index(maximum_observed) if steps else None),
        "centroid_xy": [list(point) for point in centroids_xy],
        "step_pixels": steps,
    }


def score_attachment_distance(
    distances_pixels: list[float],
    *,
    baseline_hold_frames: int,
    maximum_distance_increase_pixels: float,
) -> dict[str, object]:
    """Score whether two masks or one mask and a color target remain attached."""

    if (
        not distances_pixels
        or baseline_hold_frames < 1
        or baseline_hold_frames > len(distances_pixels)
        or not math.isfinite(maximum_distance_increase_pixels)
        or maximum_distance_increase_pixels < 0
        or any(not math.isfinite(distance) or distance < 0 for distance in distances_pixels)
    ):
        raise ValueError("invalid attachment-distance series or thresholds")
    baseline = float(median(distances_pixels[:baseline_hold_frames]))
    maximum_allowed = baseline + maximum_distance_increase_pixels
    maximum_observed = max(distances_pixels)
    return {
        "passed": maximum_observed <= maximum_allowed,
        "baseline_hold_frames": baseline_hold_frames,
        "baseline_distance_pixels": baseline,
        "maximum_distance_increase_pixels": maximum_distance_increase_pixels,
        "maximum_allowed_distance_pixels": maximum_allowed,
        "maximum_observed_distance_pixels": maximum_observed,
        "terminal_distance_pixels": distances_pixels[-1],
        "distance_pixels": distances_pixels,
    }


def save_packed_masks(
    path: Path,
    masks_by_name: Mapping[str, Any],
    *,
    frame_count: int,
    height: int,
    width: int,
) -> dict[str, str]:
    """Save boolean masks without pickle and return object-to-key bindings."""

    import numpy as np

    keys: dict[str, str] = {}
    payload: dict[str, Any] = {}
    for index, (name, masks) in enumerate(sorted(masks_by_name.items())):
        key = f"object_{index:02d}_packed"
        array = np.asarray(masks, dtype=np.uint8)
        if array.shape != (frame_count, height, width):
            raise ValueError(
                f"{name} mask shape {array.shape} differs from {(frame_count, height, width)}"
            )
        keys[name] = key
        payload[key] = np.packbits(array.reshape(frame_count, -1), axis=1, bitorder="little")
    np.savez_compressed(
        path,
        **payload,
        frame_count=frame_count,
        height=height,
        width=width,
        bitorder="little",
    )
    return keys


def load_packed_masks(
    path: Path,
    object_keys: Mapping[str, str],
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    """Load and validate packed masks written by :func:`save_packed_masks`."""

    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        frame_count = int(archive["frame_count"])
        height = int(archive["height"])
        width = int(archive["width"])
        bitorder = str(archive["bitorder"])
        if bitorder != "little":
            raise ValueError(f"unsupported mask bit order: {bitorder}")
        pixel_count = height * width
        masks = {}
        for name, key in object_keys.items():
            if key not in archive:
                raise ValueError(f"missing packed mask key {key!r} for {name!r}")
            packed = archive[key]
            if packed.shape[0] != frame_count:
                raise ValueError(f"packed mask frame count differs for {name!r}")
            unpacked = np.unpackbits(packed, axis=1, bitorder="little")
            masks[name] = unpacked[:, :pixel_count].reshape(frame_count, height, width).astype(bool)
    return masks, (frame_count, height, width)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile requires values and a fraction in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _mask_iou(first: Any, second: Any, np: Any) -> list[float]:
    intersections = np.logical_and(first, second).sum(axis=(1, 2))
    unions = np.logical_or(first, second).sum(axis=(1, 2))
    return [
        1.0 if int(union) == 0 else float(intersection / union)
        for intersection, union in zip(intersections, unions)
    ]


def _centroids(masks: Any, np: Any) -> list[tuple[float, float]]:
    centroids: list[tuple[float, float]] = []
    for mask in masks:
        rows, columns = np.nonzero(mask)
        centroids.append(
            (0.0, 0.0) if not len(columns) else (float(columns.mean()), float(rows.mean()))
        )
    return centroids


def validate_result_mask_artifact(
    result: Mapping[str, object],
    *,
    expected_model_id: str,
) -> tuple[Path, str]:
    """Verify one result's decision-relevant mask artifact binding."""

    if expected_model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"unsupported model_id: {expected_model_id}")
    if result.get("model_id") != expected_model_id:
        raise ValueError(f"result must come from {expected_model_id}")
    path = Path(str(result["masks"]))
    expected_sha256 = validate_sha256(
        result.get("masks_sha256"), f"{expected_model_id} masks_sha256"
    )
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{expected_model_id} mask artifact differs from its bound SHA-256")
    return path, expected_sha256


def compare_tracker_results(
    authoritative: Mapping[str, object],
    shadow: Mapping[str, object],
) -> dict[str, object]:
    """Compare two mask artifacts without treating either model as ground truth."""

    import numpy as np

    if authoritative.get("model_id") != SAM2_MODEL_ID:
        raise ValueError("authoritative result must come from SAM2")
    if shadow.get("model_id") != SAM31_MODEL_ID:
        raise ValueError("shadow result must come from SAM3.1")
    if authoritative.get("prepared_input_sha256") != shadow.get("prepared_input_sha256"):
        raise ValueError("A/B results do not bind the same prepared input")
    authoritative_keys = authoritative.get("mask_keys")
    shadow_keys = shadow.get("mask_keys")
    if not isinstance(authoritative_keys, dict) or not isinstance(shadow_keys, dict):
        raise ValueError("A/B results require mask key mappings")
    if set(authoritative_keys) != set(shadow_keys):
        raise ValueError("A/B results track different object names")
    authoritative_masks_path, authoritative_masks_sha256 = validate_result_mask_artifact(
        authoritative,
        expected_model_id=SAM2_MODEL_ID,
    )
    shadow_masks_path, shadow_masks_sha256 = validate_result_mask_artifact(
        shadow,
        expected_model_id=SAM31_MODEL_ID,
    )
    incumbent_masks, incumbent_shape = load_packed_masks(
        authoritative_masks_path,
        {str(name): str(key) for name, key in authoritative_keys.items()},
    )
    challenger_masks, challenger_shape = load_packed_masks(
        shadow_masks_path,
        {str(name): str(key) for name, key in shadow_keys.items()},
    )
    if incumbent_shape != challenger_shape:
        raise ValueError("A/B mask artifacts have different shapes")
    frame_count, height, width = incumbent_shape
    pixel_count = height * width
    object_names = sorted(incumbent_masks)
    object_results: dict[str, object] = {}
    label_takeover_frames: dict[str, list[int]] = {}
    for name in object_names:
        incumbent = incumbent_masks[name]
        challenger = challenger_masks[name]
        frame_ious = _mask_iou(incumbent, challenger, np)
        disagreements = np.logical_xor(incumbent, challenger).sum(axis=(1, 2)) / pixel_count
        incumbent_areas = incumbent.sum(axis=(1, 2))
        challenger_areas = challenger.sum(axis=(1, 2))
        area_ratios = [
            float(challenger_area / max(int(incumbent_area), 1))
            for incumbent_area, challenger_area in zip(incumbent_areas, challenger_areas)
        ]
        incumbent_centroids = _centroids(incumbent, np)
        challenger_centroids = _centroids(challenger, np)
        centroid_distances = [
            math.dist(first, second)
            for first, second in zip(incumbent_centroids, challenger_centroids)
        ]
        other_names = [candidate for candidate in object_names if candidate != name]
        takeover = []
        if other_names:
            cross_ious = [
                _mask_iou(incumbent_masks[other], challenger, np) for other in other_names
            ]
            for frame_index, same_iou in enumerate(frame_ious):
                if max(values[frame_index] for values in cross_ious) > same_iou:
                    takeover.append(frame_index)
        label_takeover_frames[name] = takeover
        object_results[name] = {
            "mean_frame_iou": sum(frame_ious) / len(frame_ious),
            "minimum_frame_iou": min(frame_ious),
            "p05_frame_iou": _percentile(frame_ious, 0.05),
            "frames_below_iou_0_90": [
                index for index, value in enumerate(frame_ious) if value < 0.90
            ],
            "maximum_disagreement_fraction": float(max(disagreements)),
            "mean_area_ratio_shadow_over_authoritative": (sum(area_ratios) / len(area_ratios)),
            "maximum_centroid_distance_pixels": max(centroid_distances),
            "mean_centroid_distance_pixels": (sum(centroid_distances) / len(centroid_distances)),
            "potential_label_takeover_frames": takeover,
            "frame_iou": frame_ious,
        }

    incumbent_diagnostics = authoritative.get("incumbent_threshold_diagnostics")
    shadow_diagnostics = shadow.get("incumbent_threshold_diagnostics")
    gate_disagreements: dict[str, list[str]] = {}
    if isinstance(incumbent_diagnostics, dict) and isinstance(shadow_diagnostics, dict):
        for name in object_names:
            incumbent_object = incumbent_diagnostics.get(name)
            shadow_object = shadow_diagnostics.get(name)
            if not isinstance(incumbent_object, dict) or not isinstance(shadow_object, dict):
                continue
            incumbent_gates = incumbent_object.get("gate_results")
            shadow_gates = shadow_object.get("gate_results")
            if not isinstance(incumbent_gates, dict) or not isinstance(shadow_gates, dict):
                continue
            gate_disagreements[name] = sorted(
                gate
                for gate in set(incumbent_gates) & set(shadow_gates)
                if bool(incumbent_gates[gate]) != bool(shadow_gates[gate])
            )

    sam2_elapsed = float(authoritative["runtime"]["elapsed_seconds"])
    sam31_elapsed = float(shadow["runtime"]["elapsed_seconds"])
    return {
        "schema_version": AB_SCHEMA_VERSION,
        "comparison_policy": "agreement_only_no_ground_truth",
        "authoritative_model": SAM2_MODEL_ID,
        "shadow_model": SAM31_MODEL_ID,
        "same_prepared_input": True,
        "prepared_input_sha256": authoritative["prepared_input_sha256"],
        "verified_mask_artifacts": {
            SAM2_MODEL_ID: {
                "path": str(authoritative_masks_path),
                "sha256": authoritative_masks_sha256,
            },
            SAM31_MODEL_ID: {
                "path": str(shadow_masks_path),
                "sha256": shadow_masks_sha256,
            },
        },
        "frame_count": frame_count,
        "frame_size": [width, height],
        "objects": object_results,
        "potential_label_takeover_frames": label_takeover_frames,
        "incumbent_threshold_gate_disagreements": gate_disagreements,
        "runtime": {
            "sam2_elapsed_seconds": sam2_elapsed,
            "sam31_elapsed_seconds": sam31_elapsed,
            "sam31_over_sam2_elapsed_ratio": (
                sam31_elapsed / sam2_elapsed if sam2_elapsed > 0 else None
            ),
            "sam2_peak_cuda_memory_mib": authoritative["runtime"]["peak_cuda_memory_mib"],
            "sam31_peak_cuda_memory_mib": shadow["runtime"]["peak_cuda_memory_mib"],
        },
        "authoritative_hard_gates_passed": authoritative["hard_gates_passed"],
        "shadow_hard_gates_passed": None,
        "promotion_eligible": False,
        "promotion_blockers": [
            "SAM3.1 thresholds are not registered for this evaluator epoch",
            "A/B mask agreement is not ground-truth accuracy",
            "native labeled keyframe review has not run",
        ],
    }


def capture_git_state(project_root: Path) -> dict[str, object]:
    """Capture the reproducibility-relevant Git state without mutating it."""

    def run(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }
