from __future__ import annotations

import pytest

from phiagent.rendering.object_factored_long_video import (
    SourceResizeCrop,
    binary_dilate_square,
    compose_object_factored_frame,
    remap_boolean_mask,
    resolve_flower_visibility,
    source_skin_like,
    strict_flower_seed,
)


def _frame(name: str, **overrides: int) -> SourceResizeCrop:
    values = {
        "source_width": 8,
        "source_height": 4,
        "scaled_width": 8,
        "scaled_height": 4,
        "crop_left": 0,
        "crop_top": 0,
        "output_width": 8,
        "output_height": 4,
    }
    values.update(overrides)
    return SourceResizeCrop(name=name, **values)


def test_coordinate_frame_rejects_crop_outside_scaled_source() -> None:
    with pytest.raises(ValueError, match="horizontal crop"):
        _frame("camera:bad", scaled_width=7).validate()


def test_mask_remap_accounts_for_source_crop_before_downscale() -> None:
    np = pytest.importorskip("numpy")
    source = _frame(
        "camera:source_aligned",
        scaled_width=10,
        crop_left=1,
        output_width=8,
    )
    target = _frame(
        "camera:target",
        scaled_width=5,
        scaled_height=2,
        output_width=5,
        output_height=2,
    )
    mask = np.zeros((4, 8), dtype=bool)
    mask[:, 1:5] = True

    mapped = remap_boolean_mask(np, mask, source_frame=source, target_frame=target)

    assert mapped.shape == (2, 5)
    assert mapped[:, 1:3].all()
    assert not mapped[:, 0].any()
    assert not mapped[:, 3:].any()


def test_square_dilation_expands_one_pixel_by_requested_radius() -> None:
    np = pytest.importorskip("numpy")
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True

    dilated = binary_dilate_square(np, mask, 2)

    assert int(dilated.sum()) == 25
    assert dilated[1:6, 1:6].all()


def test_source_flower_layer_overrides_generated_subject_layer() -> None:
    np = pytest.importorskip("numpy")
    source = np.full((3, 4, 3), 20, dtype=np.uint8)
    generated = np.full((3, 4, 3), 200, dtype=np.uint8)
    support = np.zeros((3, 4), dtype=bool)
    support[:, 1:3] = True
    flower = np.zeros((3, 4), dtype=bool)
    flower[1, 2] = True

    output = compose_object_factored_frame(
        np,
        source_rgb=source,
        generated_rgb=generated,
        edit_support=support,
        flower_restore=flower,
    )

    assert np.all(output[:, 0] == 20)
    assert np.all(output[:, 1] == 200)
    assert np.all(output[1, 2] == 20)


def test_skin_negative_detects_warm_source_hand_color() -> None:
    np = pytest.importorskip("numpy")
    frame = np.zeros((1, 2, 3), dtype=np.uint8)
    frame[0, 0] = (190, 140, 115)
    frame[0, 1] = (25, 160, 40)

    skin = source_skin_like(np, frame)

    assert bool(skin[0, 0]) is True
    assert bool(skin[0, 1]) is False


def test_low_resolution_strict_flower_seed_keeps_single_green_pixel() -> None:
    np = pytest.importorskip("numpy")
    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    frame[1, 1] = (25, 160, 40)

    flower = strict_flower_seed(np, frame)

    assert bool(flower[1, 1]) is True


def test_flower_visibility_excludes_person_core_but_keeps_boundary() -> None:
    np = pytest.importorskip("numpy")
    candidates = np.ones((7, 7), dtype=bool)
    support = np.ones((7, 7), dtype=bool)
    person = np.zeros((7, 7), dtype=bool)
    person[1:6, 1:6] = True
    skin = np.zeros((7, 7), dtype=bool)

    visible = resolve_flower_visibility(
        np,
        candidates=candidates,
        edit_support=support,
        source_person=person,
        source_skin_negative=skin,
        person_core_erosion=1,
    )

    assert bool(visible[3, 3]) is False
    assert bool(visible[1, 1]) is True
    assert bool(visible[0, 0]) is True
