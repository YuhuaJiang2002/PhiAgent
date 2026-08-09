"""Instance-level tracking, evaluation, and safe source-object compositing."""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class NormalizedROI:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("object ROI values must be finite")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("object ROI must have non-negative origin and positive size")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("object ROI must fit inside normalized image bounds")

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            max(0, round(self.x * width)),
            max(0, round(self.y * height)),
            min(width, round((self.x + self.width) * width)),
            min(height, round((self.y + self.height) * height)),
        )


@dataclass(frozen=True)
class RGBFrames:
    frames: tuple[bytes, ...]
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or not self.frames:
            raise ValueError("RGB frame sequence requires positive dimensions and frames")
        expected = self.width * self.height * 3
        if any(len(frame) != expected for frame in self.frames):
            raise ValueError("RGB frame has an unexpected byte count")


@dataclass(frozen=True)
class ObjectTrackerConfig:
    initial_roi: NormalizedROI
    initial_color_mode: str = "chromatic"
    minimum_chroma: int = 10
    chroma_tolerance: int = 18
    brightness_tolerance: int = 45
    search_margin: float = 0.12
    minimum_component_pixels: int = 6
    maximum_area_ratio: float = 2.0
    duplicate_cleanup_radius: int = 12
    restoration_margin: int = 8
    restoration_minimum_chroma: int = 2

    def __post_init__(self) -> None:
        if self.initial_color_mode not in {"chromatic", "cyan"}:
            raise ValueError("initial_color_mode must be chromatic or cyan")
        if self.minimum_chroma < 0 or self.chroma_tolerance <= 0:
            raise ValueError("object tracker chroma thresholds are invalid")
        if self.brightness_tolerance <= 0:
            raise ValueError("object tracker brightness tolerance must be positive")
        if not 0 < self.search_margin <= 1:
            raise ValueError("object tracker search margin must be in (0, 1]")
        if self.minimum_component_pixels <= 0:
            raise ValueError("minimum object component size must be positive")
        if not math.isfinite(self.maximum_area_ratio) or self.maximum_area_ratio <= 1:
            raise ValueError("maximum object area ratio must be finite and greater than one")
        if self.duplicate_cleanup_radius < 0:
            raise ValueError("duplicate cleanup radius must be non-negative")
        if self.restoration_margin < 0 or self.restoration_minimum_chroma < 0:
            raise ValueError("restoration mask parameters must be non-negative")


@dataclass(frozen=True)
class ObjectColorModel:
    green_minus_red: float
    blue_minus_red: float
    green_minus_blue: float
    brightness: float


@dataclass(frozen=True)
class ObjectTrack:
    masks: tuple[bytes, ...]
    boxes: tuple[tuple[int, int, int, int] | None, ...]
    mean_colors: tuple[tuple[float, float, float] | None, ...]
    areas: tuple[int, ...]
    model: ObjectColorModel
    width: int
    height: int


@dataclass(frozen=True)
class ObjectInstanceMetrics:
    contour_similarity: float
    color_similarity: float
    temporal_deformation: float
    tracking_coverage: float
    trajectory_similarity: float
    lift_recall: float
    object_consistency: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class MaskCoverage:
    per_frame_fraction: tuple[float, ...]
    mean_fraction: float
    maximum_fraction: float
    covered: bool


@dataclass(frozen=True)
class ObjectConfidenceRoute:
    decision: str
    repair_applied: bool
    source_area_ratio: float
    candidate_area_ratio: float
    source_track_all_frames: bool
    candidate_track_all_frames: bool
    candidate_lift_recall: float
    trajectory_similarity: float
    reason: str


def _track_area_ratio(track: ObjectTrack) -> float:
    areas = [area for area in track.areas if area > 0]
    if not areas:
        raise ValueError("object confidence routing requires a non-empty track")
    return max(areas) / min(areas)


def route_object_preservation(
    source: RGBFrames,
    candidate: RGBFrames,
    config: ObjectTrackerConfig,
    *,
    maximum_candidate_area_ratio: float = 3.0,
) -> ObjectConfidenceRoute:
    """Keep a complete generated object when its measured track is reliable."""

    if maximum_candidate_area_ratio <= 1:
        raise ValueError("maximum candidate area ratio must be greater than one")
    source_track = track_colored_object(source, config)
    candidate_track = track_colored_object(candidate, config, model=source_track.model)
    metrics = evaluate_object_instance(source, candidate, config)
    source_area_ratio = _track_area_ratio(source_track)
    candidate_area_ratio = _track_area_ratio(candidate_track)
    source_all_frames = all(box is not None for box in source_track.boxes)
    all_frames = all(box is not None for box in candidate_track.boxes)
    source_reliable = (
        source_all_frames and source_area_ratio <= maximum_candidate_area_ratio
    )
    candidate_reliable = (
        all_frames
        and candidate_area_ratio <= maximum_candidate_area_ratio
    )
    preserve_raw = candidate_reliable and (
        not source_reliable or metrics.lift_recall == 1.0
    )
    if preserve_raw and not source_reliable:
        decision = "preserve_raw_candidate_source_track_unreliable"
        reason = (
            "source object track is unreliable while the candidate track is present "
            "in all frames with area ratio <= "
            f"{maximum_candidate_area_ratio:g}; destructive source-object repair is disabled"
        )
    elif preserve_raw:
        decision = "preserve_raw_candidate_all_frames"
        reason = (
            "candidate track is present in all frames, area ratio <= "
            f"{maximum_candidate_area_ratio:g}, and lift recall is 1.0; "
            "destructive object repair is disabled"
        )
    else:
        decision = "repair_candidate_object"
        reason = "candidate object confidence gate failed; deterministic repair is required"
    return ObjectConfidenceRoute(
        decision=decision,
        repair_applied=not preserve_raw,
        source_area_ratio=source_area_ratio,
        candidate_area_ratio=candidate_area_ratio,
        source_track_all_frames=source_all_frames,
        candidate_track_all_frames=all_frames,
        candidate_lift_recall=metrics.lift_recall,
        trajectory_similarity=metrics.trajectory_similarity,
        reason=reason,
    )


def _matches_object_color(
    red: int,
    green: int,
    blue: int,
    config: ObjectTrackerConfig,
    model: ObjectColorModel | None,
) -> bool:
    green_red = green - red
    blue_red = blue - red
    green_blue = green - blue
    if model is None:
        if (
            config.initial_color_mode == "chromatic"
            and max(red, green, blue) - min(red, green, blue)
            < config.minimum_chroma
        ):
            return False
        if config.initial_color_mode == "cyan" and (
            green_red < config.minimum_chroma
            or blue_red < config.minimum_chroma
        ):
            return False
        return True
    return not (
        abs(green_red - model.green_minus_red) > config.chroma_tolerance
        or abs(blue_red - model.blue_minus_red) > config.chroma_tolerance
        or abs(green_blue - model.green_minus_blue) > config.chroma_tolerance
        or abs((red + green + blue) / 3 - model.brightness)
        > config.brightness_tolerance
    )


def _component_mask(
    frame: bytes,
    width: int,
    height: int,
    region: tuple[int, int, int, int],
    config: ObjectTrackerConfig,
    model: ObjectColorModel | None,
    previous_center: tuple[float, float] | None,
    maximum_component_pixels: int | None = None,
) -> tuple[bytes, tuple[int, int, int, int] | None, tuple[float, float, float] | None]:
    x0, y0, x1, y1 = region
    eligible = bytearray(width * height)
    for y in range(y0, y1):
        row = y * width
        for x in range(x0, x1):
            pixel = (row + x) * 3
            red, green, blue = frame[pixel : pixel + 3]
            if not _matches_object_color(red, green, blue, config, model):
                continue
            eligible[row + x] = 1

    best: list[int] = []
    best_score = -1.0
    visited = bytearray(width * height)
    diagonal = math.hypot(width, height)
    for y in range(y0, y1):
        for x in range(x0, x1):
            start = y * width + x
            if not eligible[start] or visited[start]:
                continue
            stack = [start]
            visited[start] = 1
            component: list[int] = []
            while stack:
                index = stack.pop()
                component.append(index)
                px = index % width
                py = index // width
                for neighbor in (index - 1, index + 1, index - width, index + width):
                    if neighbor < 0 or neighbor >= width * height or visited[neighbor]:
                        continue
                    nx = neighbor % width
                    ny = neighbor // width
                    if abs(nx - px) + abs(ny - py) != 1 or not eligible[neighbor]:
                        continue
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if len(component) < config.minimum_component_pixels:
                continue
            if (
                maximum_component_pixels is not None
                and len(component) > maximum_component_pixels
            ):
                continue
            center_x = sum(index % width for index in component) / len(component)
            center_y = sum(index // width for index in component) / len(component)
            distance = (
                0.0
                if previous_center is None
                else math.hypot(center_x - previous_center[0], center_y - previous_center[1])
            )
            score = len(component) / (1 + 8 * distance / diagonal)
            if score > best_score:
                best = component
                best_score = score

    mask = bytearray(width * height)
    if not best:
        return bytes(mask), None, None
    red_total = green_total = blue_total = 0
    xs: list[int] = []
    ys: list[int] = []
    for index in best:
        mask[index] = 1
        xs.append(index % width)
        ys.append(index // width)
        pixel = index * 3
        red, green, blue = frame[pixel : pixel + 3]
        red_total += red
        green_total += green
        blue_total += blue
    count = len(best)
    return (
        bytes(mask),
        (min(xs), min(ys), max(xs) + 1, max(ys) + 1),
        (red_total / count, green_total / count, blue_total / count),
    )


def _color_model(color: tuple[float, float, float]) -> ObjectColorModel:
    red, green, blue = color
    return ObjectColorModel(
        green_minus_red=green - red,
        blue_minus_red=blue - red,
        green_minus_blue=green - blue,
        brightness=(red + green + blue) / 3,
    )


def _learn_color_model(
    frame: bytes,
    width: int,
    region: tuple[int, int, int, int],
    config: ObjectTrackerConfig,
) -> ObjectColorModel:
    x0, y0, x1, y1 = region
    bins: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for y in range(y0, y1):
        for x in range(x0, x1):
            pixel = (y * width + x) * 3
            red, green, blue = frame[pixel : pixel + 3]
            if config.initial_color_mode == "chromatic" and (
                max(red, green, blue) - min(red, green, blue)
                < config.minimum_chroma
            ):
                continue
            if config.initial_color_mode == "cyan" and (
                green - red < config.minimum_chroma
                or blue - red < config.minimum_chroma
            ):
                continue
            key = (
                round((green - red) / 12),
                round((blue - red) / 12),
                round(((red + green + blue) / 3) / 24),
            )
            bins[key].append((red, green, blue))
    if not bins:
        raise ValueError("no chromatic object component found in the initial ROI")
    colors = max(bins.values(), key=len)
    count = len(colors)
    return _color_model(
        tuple(sum(color[channel] for color in colors) / count for channel in range(3))
    )


def track_colored_object(
    decoded: RGBFrames,
    config: ObjectTrackerConfig,
    *,
    model: ObjectColorModel | None = None,
) -> ObjectTrack:
    width, height = decoded.width, decoded.height
    initial_region = config.initial_roi.pixels(width, height)
    masks: list[bytes] = []
    boxes: list[tuple[int, int, int, int] | None] = []
    colors: list[tuple[float, float, float] | None] = []
    previous_box: tuple[int, int, int, int] | None = None
    active_model = model
    initial_area: int | None = None
    margin = round(config.search_margin * max(width, height))
    for frame_index, frame in enumerate(decoded.frames):
        if frame_index == 0 or previous_box is None:
            region = initial_region
        else:
            x0, y0, x1, y1 = previous_box
            region = (
                max(0, x0 - margin),
                max(0, y0 - margin),
                min(width, x1 + margin),
                min(height, y1 + margin),
            )
        if active_model is None:
            active_model = _learn_color_model(frame, width, region, config)
        mask, box, color = _component_mask(
            frame,
            width,
            height,
            region,
            config,
            active_model,
            None
            if previous_box is None
            else (
                (previous_box[0] + previous_box[2]) / 2,
                (previous_box[1] + previous_box[3]) / 2,
            ),
            (
                round(initial_area * config.maximum_area_ratio)
                if initial_area is not None
                else None
            ),
        )
        if frame_index == 0 and box is None:
            raise ValueError(
                "no chromatic object component matched the learned initial ROI color"
            )
        if box is not None:
            previous_box = box
            if initial_area is None:
                initial_area = sum(mask)
        masks.append(mask)
        boxes.append(box)
        colors.append(color)
    assert active_model is not None
    return ObjectTrack(
        masks=tuple(masks),
        boxes=tuple(boxes),
        mean_colors=tuple(colors),
        areas=tuple(sum(mask) for mask in masks),
        model=active_model,
        width=width,
        height=height,
    )


def expand_object_restoration_masks(
    decoded: RGBFrames,
    track: ObjectTrack,
    config: ObjectTrackerConfig,
) -> tuple[bytes, ...]:
    """Recover low-saturation object edges connected to each high-confidence mask."""

    if (decoded.width, decoded.height) != (track.width, track.height):
        raise ValueError("restoration-mask dimensions must match the object track")
    width, height = decoded.width, decoded.height
    expanded_masks: list[bytes] = []
    for frame, seed_mask, box in zip(decoded.frames, track.masks, track.boxes):
        if box is None:
            expanded_masks.append(seed_mask)
            continue
        x0 = max(0, box[0] - config.restoration_margin)
        y0 = max(0, box[1] - config.restoration_margin)
        x1 = min(width, box[2] + config.restoration_margin)
        y1 = min(height, box[3] + config.restoration_margin)
        eligible = bytearray(width * height)
        for y in range(y0, y1):
            for x in range(x0, x1):
                index = y * width + x
                pixel = index * 3
                red, green, blue = frame[pixel : pixel + 3]
                brightness = (red + green + blue) / 3
                if (
                    green - red >= config.restoration_minimum_chroma
                    and blue - red >= config.restoration_minimum_chroma
                    and abs((green - blue) - track.model.green_minus_blue)
                    <= 2 * config.chroma_tolerance
                    and abs(brightness - track.model.brightness)
                    <= 2 * config.brightness_tolerance
                ):
                    eligible[index] = 1

        selected = bytearray(seed_mask)
        stack = [index for index, value in enumerate(seed_mask) if value]
        while stack:
            index = stack.pop()
            x, y = index % width, index // width
            for nx, ny in (
                (x - 1, y - 1),
                (x, y - 1),
                (x + 1, y - 1),
                (x - 1, y),
                (x + 1, y),
                (x - 1, y + 1),
                (x, y + 1),
                (x + 1, y + 1),
            ):
                if not (x0 <= nx < x1 and y0 <= ny < y1):
                    continue
                neighbor = ny * width + nx
                if eligible[neighbor] and not selected[neighbor]:
                    selected[neighbor] = 1
                    stack.append(neighbor)

        for _ in range(2):
            previous = bytes(selected)
            for y in range(y0 + 1, y1 - 1):
                for x in range(x0 + 1, x1 - 1):
                    index = y * width + x
                    if previous[index]:
                        continue
                    neighbors = sum(
                        previous[(y + dy) * width + x + dx]
                        for dy in (-1, 0, 1)
                        for dx in (-1, 0, 1)
                        if dx or dy
                    )
                    if neighbors >= 5:
                        selected[index] = 1
        expanded_masks.append(bytes(selected))
    return tuple(expanded_masks)


def _normalized_mask(mask: bytes, box: tuple[int, int, int, int], width: int) -> bytes:
    grid = 24
    x0, y0, x1, y1 = box
    normalized = bytearray(grid * grid)
    for gy in range(grid):
        y = min(y1 - 1, y0 + ((2 * gy + 1) * (y1 - y0)) // (2 * grid))
        for gx in range(grid):
            x = min(x1 - 1, x0 + ((2 * gx + 1) * (x1 - x0)) // (2 * grid))
            normalized[gy * grid + gx] = mask[y * width + x]
    return bytes(normalized)


def _mask_iou(left: bytes, right: bytes) -> float:
    intersection = sum(a and b for a, b in zip(left, right))
    union = sum(a or b for a, b in zip(left, right))
    return intersection / union if union else 0.0


def evaluate_object_instance(
    source: RGBFrames,
    candidate: RGBFrames,
    config: ObjectTrackerConfig,
) -> ObjectInstanceMetrics:
    if (source.width, source.height) != (candidate.width, candidate.height):
        raise ValueError("source and candidate object frames must share dimensions")
    count = min(len(source.frames), len(candidate.frames))
    if count < 3:
        raise ValueError("object evaluation requires at least three frames")
    source_track = track_colored_object(
        RGBFrames(source.frames[:count], source.width, source.height), config
    )
    candidate_track = track_colored_object(
        RGBFrames(candidate.frames[:count], candidate.width, candidate.height),
        config,
        model=source_track.model,
    )

    contour_scores: list[float] = []
    color_scores: list[float] = []
    deformation_errors: list[float] = []
    tracked = 0
    for index in range(count):
        source_box = source_track.boxes[index]
        candidate_box = candidate_track.boxes[index]
        source_color = source_track.mean_colors[index]
        candidate_color = candidate_track.mean_colors[index]
        if (
            source_box is None
            or candidate_box is None
            or source_color is None
            or candidate_color is None
        ):
            contour_scores.append(0.0)
            color_scores.append(0.0)
            deformation_errors.append(1.0)
            continue
        tracked += 1
        source_shape = _normalized_mask(
            source_track.masks[index], source_box, source.width
        )
        candidate_shape = _normalized_mask(
            candidate_track.masks[index], candidate_box, candidate.width
        )
        source_width = source_box[2] - source_box[0]
        source_height = source_box[3] - source_box[1]
        candidate_width = candidate_box[2] - candidate_box[0]
        candidate_height = candidate_box[3] - candidate_box[1]
        area_ratio_error = abs(
            math.log(max(candidate_track.areas[index], 1) / max(source_track.areas[index], 1))
        )
        aspect_ratio_error = abs(
            math.log(
                max(candidate_width / max(candidate_height, 1), 1e-6)
                / max(source_width / max(source_height, 1), 1e-6)
            )
        )
        contour_scores.append(
            min(
                _mask_iou(source_shape, candidate_shape),
                math.exp(-area_ratio_error),
                math.exp(-aspect_ratio_error),
            )
        )
        color_distance = math.sqrt(
            sum((left - right) ** 2 for left, right in zip(source_color, candidate_color))
        )
        color_scores.append(math.exp(-color_distance / 32.0))
        deformation_errors.append(area_ratio_error + aspect_ratio_error)

    temporal_error = sum(
        abs(current - previous)
        for previous, current in zip(deformation_errors, deformation_errors[1:])
    ) / max(1, len(deformation_errors) - 1)
    contour_similarity = sum(contour_scores) / count
    color_similarity = sum(color_scores) / count
    temporal_deformation = math.exp(-2.0 * temporal_error)
    tracking_coverage = tracked / count

    def center(
        box: tuple[int, int, int, int] | None,
    ) -> tuple[float, float] | None:
        if box is None:
            return None
        return (
            (box[0] + box[2]) / (2 * source.width),
            (box[1] + box[3]) / (2 * source.height),
        )

    source_centers = tuple(center(box) for box in source_track.boxes[:count])
    candidate_centers = tuple(center(box) for box in candidate_track.boxes[:count])
    source_origin = source_centers[0]
    candidate_origin = candidate_centers[0]
    trajectory_errors: list[float] = []
    if source_origin is not None and candidate_origin is not None:
        for source_center, candidate_center in zip(source_centers, candidate_centers):
            if source_center is None or candidate_center is None:
                trajectory_errors.append(1.0)
                continue
            source_delta = (
                source_center[0] - source_origin[0],
                source_center[1] - source_origin[1],
            )
            candidate_delta = (
                candidate_center[0] - candidate_origin[0],
                candidate_center[1] - candidate_origin[1],
            )
            trajectory_errors.append(
                math.hypot(
                    source_delta[0] - candidate_delta[0],
                    source_delta[1] - candidate_delta[1],
                )
            )
    else:
        trajectory_errors = [1.0] * count
    trajectory_similarity = math.exp(-max(trajectory_errors) / 0.18)

    lifted = 0
    lift_matches = 0
    if source_origin is not None and candidate_origin is not None:
        for source_center, candidate_center in zip(source_centers, candidate_centers):
            if source_center is None:
                continue
            source_lift = source_origin[1] - source_center[1]
            if source_lift < 0.05:
                continue
            lifted += 1
            if candidate_center is not None:
                candidate_lift = candidate_origin[1] - candidate_center[1]
                if candidate_lift >= max(0.03, 0.5 * source_lift):
                    lift_matches += 1
    lift_recall = lift_matches / lifted if lifted else 1.0

    return ObjectInstanceMetrics(
        contour_similarity=contour_similarity,
        color_similarity=color_similarity,
        temporal_deformation=temporal_deformation,
        tracking_coverage=tracking_coverage,
        trajectory_similarity=trajectory_similarity,
        lift_recall=lift_recall,
        object_consistency=min(
            contour_similarity,
            color_similarity,
            temporal_deformation,
            tracking_coverage,
            trajectory_similarity,
            lift_recall,
        ),
    )


def evaluate_character_mask_coverage(
    object_track: ObjectTrack,
    character_masks: Sequence[bytes],
    *,
    threshold: int = 128,
    covered_fraction: float = 0.05,
) -> MaskCoverage:
    count = min(len(object_track.masks), len(character_masks))
    if count < 1:
        raise ValueError("mask coverage requires object and character masks")
    fractions: list[float] = []
    expected = object_track.width * object_track.height
    for object_mask, character_mask in zip(object_track.masks[:count], character_masks[:count]):
        if len(character_mask) != expected:
            raise ValueError("character mask dimensions do not match object track")
        object_pixels = sum(object_mask)
        overlap = sum(
            bool(is_object) and character >= threshold
            for is_object, character in zip(object_mask, character_mask)
        )
        fractions.append(overlap / object_pixels if object_pixels else 0.0)
    mean_fraction = sum(fractions) / len(fractions)
    maximum_fraction = max(fractions)
    return MaskCoverage(
        per_frame_fraction=tuple(fractions),
        mean_fraction=mean_fraction,
        maximum_fraction=maximum_fraction,
        covered=mean_fraction >= covered_fraction or maximum_fraction >= covered_fraction,
    )


def composite_source_object(
    source: RGBFrames,
    candidate: RGBFrames,
    track: ObjectTrack,
) -> RGBFrames:
    if (source.width, source.height) != (candidate.width, candidate.height):
        raise ValueError("source and candidate compositing dimensions must match")
    count = min(len(source.frames), len(candidate.frames), len(track.masks))
    frames: list[bytes] = []
    width, height = source.width, source.height
    for source_frame, candidate_frame, mask in zip(
        source.frames[:count], candidate.frames[:count], track.masks[:count]
    ):
        output = bytearray(candidate_frame)
        for index, selected in enumerate(mask):
            if not selected:
                continue
            x = index % width
            y = index // width
            neighbors = 0
            selected_neighbors = 0
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors += 1
                    selected_neighbors += bool(mask[ny * width + nx])
            alpha = 1.0 if selected_neighbors == neighbors else 0.5
            pixel = index * 3
            for channel in range(3):
                output[pixel + channel] = round(
                    alpha * source_frame[pixel + channel]
                    + (1 - alpha) * candidate_frame[pixel + channel]
                )
        frames.append(bytes(output))
    return RGBFrames(tuple(frames), width, height)


def remove_duplicate_colored_objects(
    source: RGBFrames,
    candidate: RGBFrames,
    track: ObjectTrack,
    config: ObjectTrackerConfig,
) -> tuple[RGBFrames, tuple[int, ...]]:
    """Erase object-colored candidate pixels before restoring the source instance."""

    if (source.width, source.height) != (candidate.width, candidate.height):
        raise ValueError("duplicate cleanup dimensions must match")
    width, height = source.width, source.height
    candidate_track = track_colored_object(
        candidate,
        config,
        model=track.model,
    )
    source_boxes = [box for box in track.boxes if box is not None]
    if not source_boxes:
        raise ValueError("duplicate cleanup requires a tracked source object")
    initial_source_box = source_boxes[0]
    initial_center = (
        (initial_source_box[0] + initial_source_box[2]) / 2,
        (initial_source_box[1] + initial_source_box[3]) / 2,
    )
    initial_width = initial_source_box[2] - initial_source_box[0]
    initial_height = initial_source_box[3] - initial_source_box[1]
    frames: list[bytes] = []
    removed_counts: list[int] = []
    for source_frame, candidate_frame, source_box, candidate_box, candidate_mask in zip(
        source.frames,
        candidate.frames,
        track.boxes,
        candidate_track.boxes,
        candidate_track.masks,
    ):
        matched = bytearray(candidate_mask)
        blend_box: tuple[int, int, int, int] | None = None
        if source_box is not None and candidate_box is not None:
            source_center = (
                (source_box[0] + source_box[2]) / 2,
                (source_box[1] + source_box[3]) / 2,
            )
            displacement = math.hypot(
                (source_center[0] - initial_center[0]) / width,
                (source_center[1] - initial_center[1]) / height,
            )
            if displacement >= 0.06:
                candidate_center = (
                    (candidate_box[0] + candidate_box[2]) / 2,
                    (candidate_box[1] + candidate_box[3]) / 2,
                )
                half_width = round(1.1 * initial_width)
                half_height = round(0.9 * initial_height)
                x0 = max(0, round(candidate_center[0]) - half_width)
                x1 = min(width, round(candidate_center[0]) + half_width)
                y0 = max(0, round(candidate_center[1]) - half_height)
                y1 = min(height, round(candidate_center[1]) + half_height)
                blend_box = (x0, y0, x1, y1)
        expanded = bytearray(matched)
        for _ in range(config.duplicate_cleanup_radius):
            previous = bytes(expanded)
            for index, selected in enumerate(previous):
                if not selected:
                    continue
                x, y = index % width, index // width
                for nx, ny in (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ):
                    if 0 <= nx < width and 0 <= ny < height:
                        expanded[ny * width + nx] = 1
        output = bytearray(candidate_frame)
        for index, selected in enumerate(expanded):
            if selected:
                pixel = index * 3
                output[pixel : pixel + 3] = source_frame[pixel : pixel + 3]
        if blend_box is not None:
            x0, y0, x1, y1 = blend_box
            feather = max(8, 2 * config.duplicate_cleanup_radius)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    distance = min(x - x0, x1 - 1 - x, y - y0, y1 - 1 - y)
                    alpha = min(1.0, max(0.0, distance / feather))
                    if alpha <= 0:
                        continue
                    pixel = (y * width + x) * 3
                    for channel in range(3):
                        output[pixel + channel] = round(
                            alpha * source_frame[pixel + channel]
                            + (1 - alpha) * output[pixel + channel]
                        )
        frames.append(bytes(output))
        removed_counts.append(
            sum(expanded)
            + (
                (blend_box[2] - blend_box[0]) * (blend_box[3] - blend_box[1])
                if blend_box is not None
                else 0
            )
        )
    return RGBFrames(tuple(frames), width, height), tuple(removed_counts)


def decode_video(
    path: Path,
    ffmpeg: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frame_num: int,
    pixel_format: str,
) -> tuple[bytes, ...]:
    channels = 3 if pixel_format == "rgb24" else 1
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps},scale={width}:{height}:flags=area",
            "-frames:v",
            str(frame_num),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    frame_size = width * height * channels
    if not completed.stdout or len(completed.stdout) % frame_size:
        raise ValueError(f"ffmpeg produced invalid {pixel_format} data for {path}")
    return tuple(
        completed.stdout[index : index + frame_size]
        for index in range(0, len(completed.stdout), frame_size)
    )


def encode_video(
    frames: Sequence[bytes],
    output: Path,
    ffmpeg: Path,
    *,
    width: int,
    height: int,
    fps: int,
    pixel_format: str,
    output_pixel_format: str = "yuv420p",
    crf: int = 18,
) -> None:
    if output_pixel_format not in {"yuv420p", "yuv444p"}:
        raise ValueError("output pixel format must be yuv420p or yuv444p")
    if not 0 <= crf <= 51:
        raise ValueError("H.264 CRF must be in [0, 51]")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            output_pixel_format,
            "-crf",
            str(crf),
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame)
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg encoding failed with exit code {return_code}: {output}")


def preserve_source_object(
    *,
    source_video: Path,
    candidate_video: Path,
    character_mask_video: Path | None,
    output_video: Path,
    object_mask_video: Path,
    report_path: Path,
    ffmpeg: Path,
    width: int,
    height: int,
    fps: int,
    frame_num: int,
    tracker_config: ObjectTrackerConfig,
) -> MaskCoverage | None:
    source = RGBFrames(
        decode_video(
            source_video,
            ffmpeg,
            width=width,
            height=height,
            fps=fps,
            frame_num=frame_num,
            pixel_format="rgb24",
        ),
        width,
        height,
    )
    candidate = RGBFrames(
        decode_video(
            candidate_video,
            ffmpeg,
            width=width,
            height=height,
            fps=fps,
            frame_num=frame_num,
            pixel_format="rgb24",
        ),
        width,
        height,
    )
    track = track_colored_object(source, tracker_config)
    restoration_masks = expand_object_restoration_masks(source, track, tracker_config)
    coverage = None
    if character_mask_video is not None:
        character_masks = decode_video(
            character_mask_video,
            ffmpeg,
            width=width,
            height=height,
            fps=fps,
            frame_num=frame_num,
            pixel_format="gray",
        )
        coverage = evaluate_character_mask_coverage(track, character_masks)
    duplicate_cleaned, duplicate_pixels_removed = remove_duplicate_colored_objects(
        source, candidate, track, tracker_config
    )
    restoration_track = ObjectTrack(
        masks=restoration_masks,
        boxes=track.boxes,
        mean_colors=track.mean_colors,
        areas=tuple(sum(mask) for mask in restoration_masks),
        model=track.model,
        width=track.width,
        height=track.height,
    )
    improved = composite_source_object(source, duplicate_cleaned, restoration_track)
    encode_video(
        improved.frames,
        output_video,
        ffmpeg,
        width=width,
        height=height,
        fps=fps,
        pixel_format="rgb24",
    )
    encode_video(
        tuple(bytes(255 if value else 0 for value in mask) for mask in restoration_masks),
        object_mask_video,
        ffmpeg,
        width=width,
        height=height,
        fps=fps,
        pixel_format="gray",
    )
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source_video": str(source_video.resolve()),
                "candidate_video": str(candidate_video.resolve()),
                "character_mask_video": (
                    str(character_mask_video.resolve())
                    if character_mask_video is not None
                    else None
                ),
                "output_video": str(output_video.resolve()),
                "object_mask_video": str(object_mask_video.resolve()),
                "tracker_config": asdict(tracker_config),
                "character_mask_coverage": (
                    asdict(coverage) if coverage is not None else None
                ),
                "composited": True,
                "duplicate_object_pixels_removed": duplicate_pixels_removed,
                "high_confidence_object_pixels": tuple(sum(mask) for mask in track.masks),
                "restored_object_pixels": tuple(sum(mask) for mask in restoration_masks),
                "safety": (
                    "Duplicate cleanup is limited to color-confirmed pixels in the "
                    "tracked motion corridor. Restoration alpha is non-zero only "
                    "inside source-object pixels."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return coverage
