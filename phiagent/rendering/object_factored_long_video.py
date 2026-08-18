"""Geometry-explicit helpers for object-factored long-video compositing.

The functions in this module deliberately accept a NumPy-like module instead
of importing NumPy at package import time.  PhiAgent therefore keeps its light
CPU-only import contract while the command-line compositor can opt into NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceResizeCrop:
    """A named camera-pixel frame obtained by resize then integer crop.

    ``scaled_*`` describes the full source image after resizing.  ``crop_*``
    describes the rectangle retained from that resized image.  Two frames may
    only be remapped when they name the same original camera frame dimensions.
    """

    name: str
    source_width: int
    source_height: int
    scaled_width: int
    scaled_height: int
    crop_left: int
    crop_top: int
    output_width: int
    output_height: int

    def validate(self) -> None:
        values = (
            self.source_width,
            self.source_height,
            self.scaled_width,
            self.scaled_height,
            self.output_width,
            self.output_height,
        )
        if not self.name.strip():
            raise ValueError("coordinate-frame name must be non-empty")
        if any(value <= 0 for value in values):
            raise ValueError("source, scaled, and output dimensions must be positive")
        if self.crop_left < 0 or self.crop_top < 0:
            raise ValueError("crop offsets must be non-negative")
        if self.crop_left + self.output_width > self.scaled_width:
            raise ValueError("horizontal crop lies outside the scaled source frame")
        if self.crop_top + self.output_height > self.scaled_height:
            raise ValueError("vertical crop lies outside the scaled source frame")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "scaled_width": self.scaled_width,
            "scaled_height": self.scaled_height,
            "crop_left": self.crop_left,
            "crop_top": self.crop_top,
            "output_width": self.output_width,
            "output_height": self.output_height,
        }


def remap_boolean_mask(
    np: Any,
    mask: Any,
    *,
    source_frame: SourceResizeCrop,
    target_frame: SourceResizeCrop,
) -> Any:
    """Nearest-neighbour remap between two resize/crop camera frames.

    Pixel centres are mapped through the shared uncropped source image.  Areas
    that were cropped out of ``source_frame`` are returned as false rather than
    inventing mask values.
    """

    source_frame.validate()
    target_frame.validate()
    if (
        source_frame.source_width != target_frame.source_width
        or source_frame.source_height != target_frame.source_height
    ):
        raise ValueError("mask and target frames do not share one camera source")
    array = np.asarray(mask, dtype=bool)
    if array.shape != (source_frame.output_height, source_frame.output_width):
        raise ValueError(
            f"mask shape {array.shape} does not match source coordinate frame "
            f"{(source_frame.output_height, source_frame.output_width)}"
        )

    target_x = np.arange(
        target_frame.crop_left,
        target_frame.crop_left + target_frame.output_width,
        dtype=np.float64,
    )
    target_y = np.arange(
        target_frame.crop_top,
        target_frame.crop_top + target_frame.output_height,
        dtype=np.float64,
    )
    source_x = np.floor(
        (target_x + 0.5)
        * source_frame.scaled_width
        / target_frame.scaled_width
    ).astype(np.int64) - source_frame.crop_left
    source_y = np.floor(
        (target_y + 0.5)
        * source_frame.scaled_height
        / target_frame.scaled_height
    ).astype(np.int64) - source_frame.crop_top

    valid_x = (source_x >= 0) & (source_x < source_frame.output_width)
    valid_y = (source_y >= 0) & (source_y < source_frame.output_height)
    clipped_x = np.clip(source_x, 0, source_frame.output_width - 1)
    clipped_y = np.clip(source_y, 0, source_frame.output_height - 1)
    mapped = array[np.ix_(clipped_y, clipped_x)]
    return np.asarray(mapped & valid_y[:, None] & valid_x[None, :], dtype=bool)


def binary_dilate_square(np: Any, mask: Any, radius: int) -> Any:
    """Dilate a boolean mask with a square kernel using an integral image."""

    if radius < 0:
        raise ValueError("dilation radius must be non-negative")
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("binary dilation expects one 2D mask")
    if radius == 0:
        return array.copy()
    size = radius * 2 + 1
    padded = np.pad(array.astype(np.uint8), radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    window_sum = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return window_sum > 0


def binary_erode_square(np: Any, mask: Any, radius: int) -> Any:
    """Erode a boolean mask with a square kernel."""

    if radius < 0:
        raise ValueError("erosion radius must be non-negative")
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("binary erosion expects one 2D mask")
    if radius == 0:
        return array.copy()
    inverted = np.logical_not(array)
    return np.logical_not(binary_dilate_square(np, inverted, radius))


def rgb_to_opencv_hsv(np: Any, frame_rgb: Any) -> tuple[Any, Any, Any]:
    """Return OpenCV-compatible H, S, and V arrays without importing OpenCV."""

    rgb = np.asarray(frame_rgb, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB-to-HSV conversion expects an HxWx3 frame")
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    delta = maximum - minimum
    hue_degrees = np.zeros_like(maximum)
    nonzero = delta > 1e-7
    red_max = np.logical_and(nonzero, maximum == red)
    green_max = np.logical_and(nonzero, maximum == green)
    blue_max = np.logical_and(nonzero, maximum == blue)
    hue_degrees[red_max] = (
        60.0 * ((green[red_max] - blue[red_max]) / delta[red_max])
    ) % 360.0
    hue_degrees[green_max] = 60.0 * (
        (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    )
    hue_degrees[blue_max] = 60.0 * (
        (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    )
    saturation = np.zeros_like(maximum)
    positive_value = maximum > 1e-7
    saturation[positive_value] = delta[positive_value] / maximum[positive_value]
    return hue_degrees / 2.0, saturation * 255.0, maximum * 255.0


def strict_flower_seed(np: Any, frame_rgb: Any) -> Any:
    """Conservative saturated green, pink, and yellow output-pixel core.

    The target video is only 624x352, where a 3x3 opening erases legitimate
    one- and two-pixel stems.  The tracked object mask supplies the envelope;
    this color test is only a scale-aware fallback for tracker gaps.
    """

    hue, saturation, value = rgb_to_opencv_hsv(np, frame_rgb)
    green = np.logical_and.reduce((hue >= 28, hue <= 91, saturation >= 67, value >= 28))
    pink = np.logical_and.reduce((hue >= 145, hue <= 179, saturation >= 105, value >= 55))
    yellow = np.logical_and.reduce((hue >= 12, hue <= 35, saturation >= 105, value >= 70))
    return np.logical_or.reduce((green, pink, yellow))


def source_skin_like(np: Any, frame_rgb: Any) -> Any:
    """Conservative source-skin negative used to preserve occlusion order."""

    values = np.asarray(frame_rgb, dtype=np.float32)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return np.logical_and.reduce(
        (
            cr >= 132,
            cr <= 180,
            cb >= 75,
            cb <= 135,
            red > green * 1.03,
            green > blue * 0.90,
        )
    )


def resolve_flower_visibility(
    np: Any,
    *,
    candidates: Any,
    edit_support: Any,
    source_person: Any,
    source_skin_negative: Any,
    person_core_erosion: int,
) -> Any:
    """Resolve flower/person conflicts with an explicit camera-frame z-order.

    Flower candidates win in the edit-support boundary, but the generated
    subject wins in the eroded source-person core.  Skin negatives always win.
    This fail-closed rule prevents a contaminated object-union track from
    copying a source face or shirt onto the generated robot.
    """

    candidate = np.asarray(candidates, dtype=bool)
    support = np.asarray(edit_support, dtype=bool)
    person = np.asarray(source_person, dtype=bool)
    skin = np.asarray(source_skin_negative, dtype=bool)
    if not (candidate.shape == support.shape == person.shape == skin.shape):
        raise ValueError("visibility masks must use one common camera-pixel frame")
    person_core = binary_erode_square(np, person, person_core_erosion)
    return np.logical_and.reduce(
        (support, candidate, np.logical_not(skin), np.logical_not(person_core))
    )


def validate_visibility_partition(
    np: Any,
    *,
    edit_support: Any,
    flower_restore: Any,
    source_person_core: Any,
    source_skin_negative: Any,
) -> None:
    """Fail closed when an object layer violates its declared z-order."""

    support = np.asarray(edit_support, dtype=np.bool_)
    flower = np.asarray(flower_restore, dtype=np.bool_)
    person_core = np.asarray(source_person_core, dtype=np.bool_)
    skin = np.asarray(source_skin_negative, dtype=np.bool_)
    if not (support.shape == flower.shape == person_core.shape == skin.shape):
        raise ValueError("visibility-partition masks must have one common shape")
    if np.any(np.logical_and(flower, np.logical_not(support))):
        raise ValueError("flower restore escapes the declared edit support")
    if np.any(np.logical_and(flower, person_core)):
        raise ValueError("flower restore overlaps the protected source-person core")
    if np.any(np.logical_and(flower, skin)):
        raise ValueError("flower restore overlaps the protected source-skin negative")


def compose_object_factored_frame(
    np: Any,
    *,
    source_rgb: Any,
    generated_rgb: Any,
    edit_support: Any,
    flower_restore: Any,
) -> Any:
    """Compose source base, generated subject, then visible source flowers."""

    source = np.asarray(source_rgb, dtype=np.uint8)
    generated = np.asarray(generated_rgb, dtype=np.uint8)
    support = np.asarray(edit_support, dtype=bool)
    flowers = np.asarray(flower_restore, dtype=bool)
    if source.shape != generated.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source and generated frames must be matching HxWx3 RGB arrays")
    if support.shape != source.shape[:2] or flowers.shape != support.shape:
        raise ValueError("edit and flower masks must match the frame dimensions")
    output = source.copy()
    output[support] = generated[support]
    output[flowers] = source[flowers]
    return output
