from __future__ import annotations

import math
from pathlib import Path

import pytest

from phiagent.physical_language.schema import (
    ContactState,
    EPLChunk,
    EPLSequence,
    FrameKind,
    FrameRef,
    ManipulationPhase,
    MotionSE3,
    Point3D,
    PoseSE3,
    Relation,
)
from phiagent.physical_language.tokenizer import EPLTokenizer


def test_transform_composition_inverse_and_point_frames() -> None:
    camera = FrameRef(FrameKind.CAMERA, "front")
    world = FrameRef(FrameKind.WORLD)
    hand = FrameRef(FrameKind.HUMAN_WRIST, "right")
    angle = math.pi / 2
    world_t_camera = PoseSE3(
        camera,
        world,
        (1.0, 0.0, 0.0),
        (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)),
    )
    camera_t_hand = PoseSE3(hand, camera, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    world_t_hand = world_t_camera.compose(camera_t_hand)
    assert world_t_hand.translation_m == pytest.approx((1.0, 1.0, 0.0))
    identity = world_t_hand.compose(world_t_hand.inverse())
    assert identity.translation_m == pytest.approx((0.0, 0.0, 0.0))
    point = Point3D(hand, (0.0, 0.0, 0.0))
    assert world_t_hand.transform_point(point).xyz_m == pytest.approx((1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="point is in"):
        world_t_hand.transform_point(Point3D(camera, (0.0, 0.0, 0.0)))


def _chunk() -> EPLChunk:
    camera = FrameRef(FrameKind.CAMERA, "front")
    wrist = FrameRef(FrameKind.HUMAN_WRIST, "right")
    object_frame = FrameRef(FrameKind.OBJECT, "cup")
    fingertips = tuple(Point3D(camera, (0.01 * index, 0.0, 0.4)) for index in range(5))
    return EPLChunk(
        start_s=0.0,
        end_s=0.1,
        phase=ManipulationPhase.GRASP,
        eef_delta=MotionSE3(camera, (0.01, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        wrist_pose=PoseSE3(
            wrist, camera, (0.1, 0.2, 0.4), (0.0, 0.0, 0.0, 1.0), 0.9
        ),
        fingertips=fingertips,  # type: ignore[arg-type]
        hand_aperture_m=0.04,
        hand_articulation=(0.1, 0.2),
        contact_state=ContactState.STABLE,
        contact_points=(fingertips[0],),
        object_pose=PoseSE3(
            object_frame, camera, (0.1, 0.2, 0.5), (0.0, 0.0, 0.0, 1.0), 0.8
        ),
        object_delta=MotionSE3(camera, (0.0, 0.0, 0.01), (0.0, 0.0, 0.0, 1.0)),
        object_state_changes=("lifted",),
        relations=(Relation("right_hand", "grasping", "cup", 0.8),),
        confidence=0.8,
    )


def test_epl_json_round_trip(tmp_path: Path) -> None:
    sequence = EPLSequence("0.1.0", "human.mp4", (_chunk(),))
    output = tmp_path / "epl.json"
    sequence.to_json(output)
    restored = EPLSequence.from_json(output)
    assert restored == sequence
    assert "target_T_source" in restored.conventions


def test_epl_rejects_object_delta_without_pose() -> None:
    payload = _chunk().to_dict()
    payload["object_pose"] = None
    with pytest.raises(ValueError, match="without object_pose"):
        EPLChunk.from_dict(payload)


def test_tokenizer_is_deterministic_and_clips() -> None:
    tokenizer = EPLTokenizer()
    tokens = tokenizer.encode_chunk(_chunk())
    assert tokens == tokenizer.encode_chunk(_chunk())
    assert "<PHASE:grasp>" in tokens
    assert "<CONTACT:stable>" in tokens
    assert "<OBJECT:PRESENT>" in tokens
