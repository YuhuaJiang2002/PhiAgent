from __future__ import annotations

from phiagent.perception.camera import PinholeIntrinsics
from phiagent.perception.extractor import PhysicalStateExtractor
from phiagent.perception.schema import (
    HandObservation,
    ObjectObservation,
    PerceptionSequence,
)
from phiagent.physical_language.schema import (
    ContactState,
    FrameKind,
    FrameRef,
    ManipulationPhase,
    Point3D,
    PoseSE3,
)
from phiagent.physical_language.visualization import build_overlay_primitives


def _hand(timestamp_s: float, wrist_x: float, fingertip_x: float) -> HandObservation:
    camera = FrameRef(FrameKind.CAMERA, "front")
    wrist = FrameRef(FrameKind.HUMAN_WRIST, "right")
    points = [Point3D(camera, (0.5, 0.5, 1.0)) for _ in range(21)]
    for index in (4, 8, 12, 16, 20):
        points[index] = Point3D(camera, (fingertip_x, 0.0, 1.0))
    points[8] = Point3D(camera, (fingertip_x + 0.02, 0.0, 1.0))
    return HandObservation(
        timestamp_s,
        PoseSE3(wrist, camera, (wrist_x, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        tuple(points),
        (),
        0.9,
    )


def _object(timestamp_s: float, x: float) -> ObjectObservation:
    camera = FrameRef(FrameKind.CAMERA, "front")
    object_frame = FrameRef(FrameKind.OBJECT, "cube")
    return ObjectObservation(
        timestamp_s,
        "cube",
        PoseSE3(object_frame, camera, (x, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        "free",
        0.8,
    )


def test_extractor_finds_contact_and_object_manipulation() -> None:
    observations = PerceptionSequence(
        "0.1.0",
        (
            _hand(0.0, 0.0, 0.20),
            _hand(0.1, 0.1, 0.01),
            _hand(0.2, 0.2, 0.02),
        ),
        (_object(0.0, 0.0), _object(0.1, 0.01), _object(0.2, 0.02)),
    )
    epl = PhysicalStateExtractor().extract(observations, "human.mp4")
    assert len(epl.chunks) == 2
    assert epl.chunks[0].phase is ManipulationPhase.APPROACH
    assert epl.chunks[1].contact_state is ContactState.STABLE
    assert epl.chunks[1].phase is ManipulationPhase.MANIPULATE
    assert epl.chunks[1].relations[0].object == "cube"


def test_camera_projection_rejects_non_camera_frame() -> None:
    camera = FrameRef(FrameKind.CAMERA, "front")
    intrinsics = PinholeIntrinsics(100.0, 100.0, 50.0, 50.0, 100, 100)
    assert intrinsics.project(Point3D(camera, (0.5, 0.0, 1.0))) == (100.0, 50.0)
    world = FrameRef(FrameKind.WORLD)
    try:
        intrinsics.project(Point3D(world, (0.0, 0.0, 1.0)))
    except ValueError as exc:
        assert "camera-frame" in str(exc)
    else:
        raise AssertionError("projection accepted a world-frame point")


def test_overlay_projects_hand_object_contact_eef_and_phase() -> None:
    observations = PerceptionSequence(
        "0.1.0",
        (_hand(0.0, 0.0, 0.01), _hand(0.1, 0.1, 0.02)),
        (_object(0.0, 0.01), _object(0.1, 0.02)),
    )
    chunk = PhysicalStateExtractor().extract(observations, "human.mp4").chunks[0]
    intrinsics = PinholeIntrinsics(100.0, 100.0, 50.0, 50.0, 128, 128)
    primitive = build_overlay_primitives(
        observations.hands[0],
        observations.objects[0],
        chunk,
        (Point3D(FrameRef(FrameKind.CAMERA, "front"), (0.0, 0.0, 1.0)),),
        intrinsics,
    )
    assert len(primitive.hand_points_px) == 21
    assert len(primitive.hand_edges_px) == 20
    assert len(primitive.object_axes_px) == 3
    assert primitive.eef_trajectory_px == ((50, 50),)
    assert primitive.phase_label in {"grasp", "manipulate"}
