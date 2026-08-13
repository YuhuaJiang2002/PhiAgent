from __future__ import annotations

import pytest

from phiagent.rendering.scene_replacement import (
    EntityRole,
    NormalizedBox,
    ReplacementGranularity,
    ReplacementSpec,
    SceneReplacementPlan,
    Shot,
    TrackKeyframe,
    TrackSegment,
)


def _keyframe(frame: int, x: float, confidence: float = 1.0) -> TrackKeyframe:
    return TrackKeyframe(frame, NormalizedBox(x, 0.2, 0.2, 0.4), confidence)


def _plan() -> SceneReplacementPlan:
    return SceneReplacementPlan(
        shots=(Shot("wide", 0, 9), Shot("close", 10, 19)),
        tracks=(
            TrackSegment(
                "person-a",
                "wide",
                EntityRole.SUBJECT,
                (_keyframe(0, 0.1), _keyframe(9, 0.2)),
                side="left",
            ),
            TrackSegment(
                "person-b",
                "wide",
                EntityRole.SUBJECT,
                (_keyframe(0, 0.65), _keyframe(9, 0.55)),
                side="right",
            ),
            TrackSegment(
                "flower",
                "wide",
                EntityRole.OBJECT,
                (
                    TrackKeyframe(0, NormalizedBox(0.20, 0.3, 0.15, 0.2)),
                    TrackKeyframe(9, NormalizedBox(0.45, 0.3, 0.15, 0.2)),
                ),
            ),
            TrackSegment(
                "vase",
                "wide",
                EntityRole.OBJECT,
                (
                    TrackKeyframe(0, NormalizedBox(0.45, 0.6, 0.1, 0.3)),
                    TrackKeyframe(9, NormalizedBox(0.45, 0.6, 0.1, 0.3)),
                ),
            ),
            TrackSegment(
                "person-a",
                "close",
                EntityRole.SUBJECT,
                (_keyframe(10, 0.35), _keyframe(19, 0.45)),
                side="left",
            ),
            TrackSegment(
                "flower",
                "close",
                EntityRole.OBJECT,
                (
                    TrackKeyframe(10, NormalizedBox(0.48, 0.3, 0.15, 0.2)),
                    TrackKeyframe(19, NormalizedBox(0.58, 0.3, 0.15, 0.2)),
                ),
            ),
            TrackSegment(
                "vase",
                "close",
                EntityRole.OBJECT,
                (
                    TrackKeyframe(10, NormalizedBox(0.58, 0.6, 0.1, 0.3)),
                    TrackKeyframe(19, NormalizedBox(0.58, 0.6, 0.1, 0.3)),
                ),
            ),
        ),
        replacements=(
            ReplacementSpec(
                "person-a", "Sharpa-left", ReplacementGranularity.HAND_FOREARM
            ),
            ReplacementSpec("person-b", "Allegro-right", ReplacementGranularity.HAND),
        ),
        protected_object_ids=("flower", "vase"),
        maximum_carry_frames=1,
    )


def test_routes_multiple_subjects_and_protected_objects() -> None:
    route = _plan().route_frame(5)

    assert [item.source_entity_id for item in route.replacements] == [
        "person-a",
        "person-b",
    ]
    assert [item.side for item in route.replacements] == ["left", "right"]
    assert {item.entity_id for item in route.protected_objects} == {"flower", "vase"}
    assert any(item.overlaps_replacement for item in route.protected_objects)
    assert not route.diagnostics


def test_camera_cut_uses_new_shot_tracks_without_cross_shot_carry() -> None:
    route = _plan().route_frame(15)

    assert route.shot_id == "close"
    assert [item.source_entity_id for item in route.replacements] == ["person-a"]
    assert route.replacements[0].box.x > 0.35
    assert [(item.code, item.entity_id) for item in route.diagnostics] == [
        ("subject_not_in_shot", "person-b")
    ]


def test_missing_and_low_confidence_tracks_are_explicit_diagnostics() -> None:
    plan = SceneReplacementPlan(
        shots=(Shot("shot", 0, 10),),
        tracks=(
            TrackSegment(
                "person",
                "shot",
                EntityRole.SUBJECT,
                (_keyframe(0, 0.1), _keyframe(1, 0.2)),
            ),
            TrackSegment(
                "object",
                "shot",
                EntityRole.OBJECT,
                (_keyframe(5, 0.4, confidence=0.2),),
            ),
        ),
        replacements=(
            ReplacementSpec("person", "robot", ReplacementGranularity.FULL_BODY),
        ),
        protected_object_ids=("object",),
        maximum_carry_frames=1,
        minimum_confidence=0.5,
    )

    route = plan.route_frame(5)

    assert not route.replacements
    assert not route.protected_objects
    assert {item.code for item in route.diagnostics} == {
        "subject_track_missing",
        "protected_object_confidence_low",
    }


def test_rejects_object_as_replacement_source() -> None:
    with pytest.raises(ValueError, match="not a tracked subject"):
        SceneReplacementPlan(
            shots=(Shot("shot", 0, 0),),
            tracks=(
                TrackSegment(
                    "flower",
                    "shot",
                    EntityRole.OBJECT,
                    (TrackKeyframe(0, NormalizedBox(0.2, 0.2, 0.2, 0.2)),),
                ),
            ),
            replacements=(
                ReplacementSpec(
                    "flower", "robot", ReplacementGranularity.FULL_BODY
                ),
            ),
        )


def test_rejects_overlapping_shots() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        SceneReplacementPlan(
            shots=(Shot("first", 0, 10), Shot("second", 10, 20)),
            tracks=(),
            replacements=(),
        )
