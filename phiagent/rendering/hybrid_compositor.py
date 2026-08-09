"""Deterministic source-video and robot-layer compositing in image-pixel coordinates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from phiagent.evaluation.object_instance import ObjectTrack, RGBFrames


@dataclass(frozen=True)
class ScreenSpaceOverlayConfig:
    """Place a rendered robot layer relative to a tracked object in the source image."""

    target_width_fraction: float = 0.30
    anchor_offset_x_fraction: float = -0.03
    anchor_offset_y_fraction: float = -0.13
    black_level: int = 20
    edge_softness: int = 24
    quarter_turns_clockwise: int = 0

    def __post_init__(self) -> None:
        values = (
            self.target_width_fraction,
            self.anchor_offset_x_fraction,
            self.anchor_offset_y_fraction,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("screen-space overlay values must be finite")
        if not 0 < self.target_width_fraction <= 1:
            raise ValueError("target_width_fraction must be in (0, 1]")
        if not -1 <= self.anchor_offset_x_fraction <= 1:
            raise ValueError("anchor_offset_x_fraction must be in [-1, 1]")
        if not -1 <= self.anchor_offset_y_fraction <= 1:
            raise ValueError("anchor_offset_y_fraction must be in [-1, 1]")
        if not 0 <= self.black_level <= 254:
            raise ValueError("black_level must be in [0, 254]")
        if self.edge_softness <= 0:
            raise ValueError("edge_softness must be positive")
        if self.quarter_turns_clockwise not in {0, 1, 2, 3}:
            raise ValueError("quarter_turns_clockwise must be 0, 1, 2, or 3")


@dataclass(frozen=True)
class HybridCompositeMetrics:
    frame_count: int
    robot_pixels: int
    restored_object_pixels: int
    source_unchanged_fraction: float
    object_exact_fraction: float

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("hybrid composite requires at least one frame")
        if self.robot_pixels <= 0:
            raise ValueError("hybrid composite contains no robot pixels")
        if self.restored_object_pixels <= 0:
            raise ValueError("hybrid composite restored no object pixels")
        for name in ("source_unchanged_fraction", "object_exact_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _foreground_bounds(frame: bytes, width: int, height: int, black_level: int) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for index in range(width * height):
        pixel = index * 3
        if max(frame[pixel : pixel + 3]) > black_level:
            xs.append(index % width)
            ys.append(index // width)
    if not xs:
        raise ValueError("rendered robot frame has no pixels above black_level")
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _composite_robot_layer_numpy(
    source: RGBFrames,
    robot_layer: RGBFrames,
    object_track: ObjectTrack,
    config: ScreenSpaceOverlayConfig,
    np: object,
) -> tuple[RGBFrames, HybridCompositeMetrics]:
    frame_count = min(len(source.frames), len(object_track.masks))
    output_frames: list[bytes] = []
    robot_pixels = restored_object_pixels = unchanged_pixels = 0
    exact_object_pixels = total_object_pixels = 0
    previous_box: tuple[int, int, int, int] | None = None

    for frame_index in range(frame_count):
        robot_index = (
            0
            if frame_count == 1
            else round(frame_index * (len(robot_layer.frames) - 1) / (frame_count - 1))
        )
        robot = np.frombuffer(robot_layer.frames[robot_index], dtype=np.uint8).reshape(
            robot_layer.height, robot_layer.width, 3
        )
        foreground_y, foreground_x = np.nonzero(robot.max(axis=2) > config.black_level)
        if not len(foreground_x):
            raise ValueError("rendered robot frame has no pixels above black_level")
        crop_x0, crop_x1 = int(foreground_x.min()), int(foreground_x.max()) + 1
        crop_y0, crop_y1 = int(foreground_y.min()), int(foreground_y.max()) + 1
        crop = robot[crop_y0:crop_y1, crop_x0:crop_x1]
        if config.quarter_turns_clockwise:
            crop = np.rot90(crop, -config.quarter_turns_clockwise)
        target_width = max(1, round(config.target_width_fraction * source.width))
        target_height = max(1, round(target_width * crop.shape[0] / crop.shape[1]))
        sample_x = np.linspace(0, crop.shape[1] - 1, target_width).astype(np.intp)
        sample_y = np.linspace(0, crop.shape[0] - 1, target_height).astype(np.intp)
        resized = crop[sample_y[:, None], sample_x[None, :]]

        box = object_track.boxes[frame_index] or previous_box
        if box is None:
            raise ValueError(f"object track has no anchor at frame {frame_index}")
        previous_box = box
        target_x0 = round(
            (box[0] + box[2]) / 2
            + config.anchor_offset_x_fraction * source.width
            - target_width / 2
        )
        target_y0 = round(
            (box[1] + box[3]) / 2
            + config.anchor_offset_y_fraction * source.height
            - target_height / 2
        )
        source_x0 = max(0, -target_x0)
        source_y0 = max(0, -target_y0)
        source_x1 = min(target_width, source.width - target_x0)
        source_y1 = min(target_height, source.height - target_y0)

        source_array = np.frombuffer(source.frames[frame_index], dtype=np.uint8).reshape(
            source.height, source.width, 3
        )
        output = source_array.copy()
        if source_x0 < source_x1 and source_y0 < source_y1:
            overlay = resized[source_y0:source_y1, source_x0:source_x1]
            intensity = overlay.max(axis=2)
            alpha = np.clip(
                (intensity.astype(np.float32) - config.black_level) / config.edge_softness,
                0,
                1,
            )[..., None]
            destination = output[
                target_y0 + source_y0 : target_y0 + source_y1,
                target_x0 + source_x0 : target_x0 + source_x1,
            ]
            destination[:] = np.rint(alpha * overlay + (1 - alpha) * destination).astype(np.uint8)
            robot_pixels += int(np.count_nonzero(alpha[..., 0]))

        object_mask = np.frombuffer(object_track.masks[frame_index], dtype=np.uint8).reshape(
            source.height, source.width
        ).astype(bool)
        output[object_mask] = source_array[object_mask]
        object_pixels = int(np.count_nonzero(object_mask))
        restored_object_pixels += object_pixels
        total_object_pixels += object_pixels
        exact_object_pixels += int(
            np.count_nonzero(np.all(output[object_mask] == source_array[object_mask], axis=1))
        )
        unchanged_pixels += int(np.count_nonzero(np.all(output == source_array, axis=2)))
        output_frames.append(output.tobytes())

    total_pixels = frame_count * source.width * source.height
    return (
        RGBFrames(tuple(output_frames), source.width, source.height),
        HybridCompositeMetrics(
            frame_count=frame_count,
            robot_pixels=robot_pixels,
            restored_object_pixels=restored_object_pixels,
            source_unchanged_fraction=unchanged_pixels / total_pixels,
            object_exact_fraction=exact_object_pixels / total_object_pixels,
        ),
    )


def composite_robot_layer(
    source: RGBFrames,
    robot_layer: RGBFrames,
    object_track: ObjectTrack,
    config: ScreenSpaceOverlayConfig,
) -> tuple[RGBFrames, HybridCompositeMetrics]:
    """Overlay robot pixels and then restore exact source-object pixels.

    The source and object track are in the source image-pixel frame. The robot layer
    is cropped in its own image-pixel frame, scaled, and translated into the source
    frame; no camera, world, or robot-base coordinates are inferred.
    """

    frame_count = min(len(source.frames), len(object_track.masks))
    if frame_count < 1:
        raise ValueError("source and object track must contain aligned frames")
    if (source.width, source.height) != (object_track.width, object_track.height):
        raise ValueError("object track must use the source image-pixel dimensions")
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        return _composite_robot_layer_numpy(source, robot_layer, object_track, config, np)

    output_frames: list[bytes] = []
    robot_pixels = 0
    restored_object_pixels = 0
    unchanged_pixels = 0
    exact_object_pixels = 0
    total_object_pixels = 0
    previous_box: tuple[int, int, int, int] | None = None

    for frame_index in range(frame_count):
        if frame_count == 1:
            robot_index = 0
        else:
            robot_index = round(
                frame_index * (len(robot_layer.frames) - 1) / (frame_count - 1)
            )
        robot_frame = robot_layer.frames[robot_index]
        crop_x0, crop_y0, crop_x1, crop_y1 = _foreground_bounds(
            robot_frame,
            robot_layer.width,
            robot_layer.height,
            config.black_level,
        )
        original_crop_width = crop_x1 - crop_x0
        original_crop_height = crop_y1 - crop_y0
        if config.quarter_turns_clockwise % 2:
            crop_width, crop_height = original_crop_height, original_crop_width
        else:
            crop_width, crop_height = original_crop_width, original_crop_height
        target_width = max(1, round(config.target_width_fraction * source.width))
        target_height = max(1, round(target_width * crop_height / crop_width))

        box = object_track.boxes[frame_index] or previous_box
        if box is None:
            raise ValueError(f"object track has no anchor at frame {frame_index}")
        previous_box = box
        object_center_x = (box[0] + box[2]) / 2
        object_center_y = (box[1] + box[3]) / 2
        target_x0 = round(
            object_center_x
            + config.anchor_offset_x_fraction * source.width
            - target_width / 2
        )
        target_y0 = round(
            object_center_y
            + config.anchor_offset_y_fraction * source.height
            - target_height / 2
        )

        source_frame = source.frames[frame_index]
        output = bytearray(source_frame)
        for target_y in range(target_height):
            destination_y = target_y0 + target_y
            if not 0 <= destination_y < source.height:
                continue
            for target_x in range(target_width):
                destination_x = target_x0 + target_x
                if not 0 <= destination_x < source.width:
                    continue
                rotated_x = min(crop_width - 1, target_x * crop_width // target_width)
                rotated_y = min(crop_height - 1, target_y * crop_height // target_height)
                if config.quarter_turns_clockwise == 0:
                    original_x, original_y = rotated_x, rotated_y
                elif config.quarter_turns_clockwise == 1:
                    original_x = rotated_y
                    original_y = original_crop_height - 1 - rotated_x
                elif config.quarter_turns_clockwise == 2:
                    original_x = original_crop_width - 1 - rotated_x
                    original_y = original_crop_height - 1 - rotated_y
                else:
                    original_x = original_crop_width - 1 - rotated_y
                    original_y = rotated_x
                robot_x = crop_x0 + original_x
                robot_y = crop_y0 + original_y
                robot_pixel = (robot_y * robot_layer.width + robot_x) * 3
                intensity = max(robot_frame[robot_pixel : robot_pixel + 3])
                if intensity <= config.black_level:
                    continue
                alpha = min(
                    1.0,
                    (intensity - config.black_level) / config.edge_softness,
                )
                destination = (destination_y * source.width + destination_x) * 3
                for channel in range(3):
                    output[destination + channel] = round(
                        alpha * robot_frame[robot_pixel + channel]
                        + (1 - alpha) * output[destination + channel]
                    )
                robot_pixels += 1

        object_mask = object_track.masks[frame_index]
        for index, selected in enumerate(object_mask):
            if not selected:
                continue
            pixel = index * 3
            output[pixel : pixel + 3] = source_frame[pixel : pixel + 3]
            restored_object_pixels += 1
            total_object_pixels += 1
            if output[pixel : pixel + 3] == source_frame[pixel : pixel + 3]:
                exact_object_pixels += 1

        unchanged_pixels += sum(
            output[index : index + 3] == source_frame[index : index + 3]
            for index in range(0, len(output), 3)
        )
        output_frames.append(bytes(output))

    total_pixels = frame_count * source.width * source.height
    metrics = HybridCompositeMetrics(
        frame_count=frame_count,
        robot_pixels=robot_pixels,
        restored_object_pixels=restored_object_pixels,
        source_unchanged_fraction=unchanged_pixels / total_pixels,
        object_exact_fraction=exact_object_pixels / total_object_pixels,
    )
    return RGBFrames(tuple(output_frames), source.width, source.height), metrics
