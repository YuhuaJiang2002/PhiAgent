from __future__ import annotations

import pytest

from phiagent.evaluation.embodiment import (
    EmbodimentEvaluationConfig,
    EmbodimentFrame,
    EmbodimentSequence,
    ImageLandmark,
    ImageMask,
    KinematicLink,
    evaluate_embodiment,
)


FRAME = "camera:demo_pixels"
NAMES = ("wrist", "knuckle", "tip")


def _mask(*, shift: int = 0, fragmented: bool = False) -> ImageMask:
    pixels = {(x + shift, y) for x in range(2, 6) for y in range(2, 6)}
    if fragmented:
        pixels.add((12, 12))
    return ImageMask(FRAME, 20, 20, frozenset(pixels))


def _frame(
    index: int,
    points: dict[str, tuple[float, float]],
    *,
    components: tuple[ImageMask, ...] | None = None,
    target_id: str | None = "robot-1",
) -> EmbodimentFrame:
    return EmbodimentFrame(
        index,
        FRAME,
        (_mask(shift=index),) if components is None else components,
        tuple(ImageLandmark(name, FRAME, *xy) for name, xy in points.items()),
        target_id,
    )


def _sequence(frames: tuple[EmbodimentFrame, ...]) -> EmbodimentSequence:
    return EmbodimentSequence(
        FRAME,
        frames,
        NAMES,
        (KinematicLink("palm", "wrist", "knuckle", expected_length=2.0),),
    )


def _config(*, articulation: bool = True) -> EmbodimentEvaluationConfig:
    return EmbodimentEvaluationConfig(
        required_landmarks=NAMES,
        expect_articulation=articulation,
        minimum_articulation_displacement=0.08,
        maximum_link_relative_drift=0.1,
    )


def test_coherent_articulated_motion_passes_all_structural_metrics() -> None:
    scorecard = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(1, {"wrist": (4, 2), "knuckle": (6, 2), "tip": (7, 4)}),
                _frame(2, {"wrist": (6, 2), "knuckle": (8, 2), "tip": (8, 5)}),
            )
        ),
        _config(),
    )

    assert all(value == 1.0 for value in scorecard.scores().values())
    assert scorecard.diagnostics.articulation_displacements


def test_rigid_pasted_layer_translation_fails_articulation() -> None:
    scorecard = evaluate_embodiment(
        _sequence(
            tuple(
                _frame(
                    index,
                    {
                        "wrist": (2 + 3 * index, 2),
                        "knuckle": (4 + 3 * index, 2),
                        "tip": (6 + 3 * index, 2),
                    },
                )
                for index in range(3)
            )
        ),
        _config(),
    )

    assert scorecard.articulation == 0.0
    assert scorecard.essential == 0.0


def test_rigid_pasted_layer_rotation_fails_articulation() -> None:
    scorecard = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(1, {"wrist": (4, 0), "knuckle": (4, 2), "tip": (4, 4)}),
            )
        ),
        _config(),
    )

    assert scorecard.articulation == pytest.approx(0.0, abs=1e-12)
    assert scorecard.essential == pytest.approx(0.0, abs=1e-12)


def test_two_frame_articulated_motion_uses_the_two_frame_window() -> None:
    scorecard = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(1, {"wrist": (4, 2), "knuckle": (6, 2), "tip": (7, 4)}),
            )
        ),
        _config(),
    )

    assert scorecard.articulation == 1.0
    assert len(scorecard.diagnostics.sustained_articulation_displacements) == 1


def test_late_one_frame_fragmentation_is_not_hidden_by_average() -> None:
    scorecard = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(1, {"wrist": (3, 2), "knuckle": (5, 2), "tip": (6, 4)}),
                _frame(
                    2,
                    {"wrist": (4, 2), "knuckle": (6, 2), "tip": (6, 5)},
                    components=(_mask(shift=2, fragmented=True),),
                ),
            )
        ),
        _config(),
    )

    assert scorecard.topology == 0.0
    assert scorecard.diagnostics.connected_component_counts[-1] == 2


def test_duplicate_component_fails_topology() -> None:
    scorecard = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(
                    1,
                    {"wrist": (3, 2), "knuckle": (5, 2), "tip": (6, 4)},
                    components=(_mask(shift=1), _mask(shift=10)),
                ),
            )
        ),
        _config(),
    )

    assert scorecard.topology == 0.0
    assert scorecard.diagnostics.component_counts == (1, 2)


def test_missing_landmarks_and_link_geometry_drift_fail() -> None:
    missing = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(1, {"wrist": (3, 2), "knuckle": (5, 2)}),
            )
        ),
        _config(),
    )
    drift = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(1, {"wrist": (3, 2), "knuckle": (7, 2), "tip": (8, 5)}),
            )
        ),
        _config(),
    )

    assert missing.landmark_tracking == 0.0
    assert drift.geometry == 0.0


def test_stable_hold_can_explicitly_not_require_articulation() -> None:
    scorecard = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(1, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
            )
        ),
        _config(articulation=False),
    )

    assert scorecard.articulation == 1.0
    assert scorecard.essential == 1.0


def test_area_instability_and_target_identity_drift_are_exposed() -> None:
    large = ImageMask(
        FRAME,
        20,
        20,
        frozenset((x, y) for x in range(2, 10) for y in range(2, 10)),
    )
    scorecard = evaluate_embodiment(
        _sequence(
            (
                _frame(0, {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}),
                _frame(
                    1,
                    {"wrist": (3, 2), "knuckle": (5, 2), "tip": (6, 4)},
                    components=(large,),
                    target_id="robot-2",
                ),
            )
        ),
        _config(),
    )

    assert scorecard.area_stability == 0.0
    assert scorecard.target_identity == 0.0
    assert scorecard.diagnostics.target_ids == ("robot-1", "robot-2")


def test_required_target_identity_fails_for_all_or_partial_missing_evidence() -> None:
    frames = (
        _frame(
            0,
            {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)},
            target_id=None,
        ),
        _frame(
            1,
            {"wrist": (3, 2), "knuckle": (5, 2), "tip": (6, 4)},
            target_id=None,
        ),
    )
    all_missing = evaluate_embodiment(_sequence(frames), _config())
    partial_missing = evaluate_embodiment(
        _sequence(
            (
                frames[0],
                _frame(1, {"wrist": (3, 2), "knuckle": (5, 2), "tip": (6, 4)}),
            )
        ),
        _config(),
    )

    assert all_missing.target_identity == 0.0
    assert partial_missing.target_identity == 0.0


def test_identity_requirement_can_be_explicitly_opted_out() -> None:
    scorecard = evaluate_embodiment(
        _sequence(
            (
                _frame(
                    0,
                    {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)},
                    target_id=None,
                ),
                _frame(
                    1,
                    {"wrist": (3, 2), "knuckle": (5, 2), "tip": (6, 4)},
                    target_id=None,
                ),
            )
        ),
        EmbodimentEvaluationConfig(
            required_landmarks=NAMES,
            require_target_identity=False,
            expect_articulation=False,
        ),
    )

    assert scorecard.target_identity == 1.0


def test_lone_jitter_frame_cannot_pass_sustained_articulation() -> None:
    stationary = {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 2)}
    frames = tuple(
        _frame(
            index,
            {"wrist": (2, 2), "knuckle": (4, 2), "tip": (6, 7)} if index == 3 else stationary,
        )
        for index in range(6)
    )
    scorecard = evaluate_embodiment(_sequence(frames), _config())

    assert max(scorecard.diagnostics.articulation_displacements) > 0.08
    assert sorted(scorecard.diagnostics.sustained_articulation_displacements)[:2] == [0.0, 0.0]
    assert scorecard.articulation == 0.0


def test_invalid_inputs_and_coordinate_frame_mismatch_raise() -> None:
    with pytest.raises(ValueError, match="finite"):
        ImageLandmark("tip", FRAME, float("nan"), 1.0)
    with pytest.raises(ValueError, match="outside"):
        ImageMask(FRAME, 2, 2, frozenset({(2, 0)}))
    with pytest.raises(ValueError, match="distinct"):
        KinematicLink("bad", "wrist", "wrist")
    with pytest.raises(ValueError, match="does not match"):
        EmbodimentFrame(
            0,
            FRAME,
            (_mask(),),
            (ImageLandmark("wrist", "camera:other_pixels", 1, 1),),
        )
