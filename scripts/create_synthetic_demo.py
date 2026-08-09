#!/usr/bin/env python3
"""Create a clearly labelled, deterministic fixture for end-to-end smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.schema import EmbodimentDescriptor  # noqa: E402
from phiagent.perception.camera import PinholeIntrinsics  # noqa: E402
from phiagent.perception.schema import (  # noqa: E402
    HandObservation,
    ObjectObservation,
    PerceptionSequence,
)
from phiagent.physical_language.schema import (  # noqa: E402
    FrameKind,
    FrameRef,
    Point3D,
    PoseSE3,
)
from phiagent.retargeting.base import LinearRetargetingConfig  # noqa: E402


def _keypoints(camera: FrameRef, wrist_x: float, object_x: float) -> tuple[Point3D, ...]:
    offsets = (
        (0.00, 0.00),
        (0.01, -0.02),
        (0.03, -0.035),
        (0.05, -0.045),
        (0.07, -0.05),
        (0.02, -0.01),
        (0.05, -0.015),
        (0.08, -0.015),
        (0.11, -0.015),
        (0.02, 0.00),
        (0.055, 0.00),
        (0.09, 0.00),
        (0.12, 0.00),
        (0.02, 0.012),
        (0.05, 0.018),
        (0.08, 0.02),
        (0.11, 0.022),
        (0.015, 0.025),
        (0.04, 0.035),
        (0.065, 0.042),
        (0.09, 0.048),
    )
    points = [Point3D(camera, (wrist_x + dx, dy, 1.0), 0.99) for dx, dy in offsets]
    # Once the wrist reaches the object, keep the index tip on the measured
    # object centre so the EPL contact transition is deterministic.
    if wrist_x >= 0.05:
        points[8] = Point3D(camera, (object_x, 0.0, 1.0), 0.99)
    return tuple(points)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty fixture: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit("synthetic fixture generation requires OpenCV and NumPy") from exc

    camera = FrameRef(FrameKind.CAMERA, "synthetic_front")
    wrist_frame = FrameRef(FrameKind.HUMAN_WRIST, "right")
    object_frame = FrameRef(FrameKind.OBJECT, "cube")
    timestamps = (0.0, 0.5, 1.0, 1.5, 2.0)
    wrist_positions = (-0.2, -0.1, 0.0, 0.1, 0.2)
    object_positions = (0.22, 0.22, 0.22, 0.24, 0.28)
    hands = []
    objects = []
    for timestamp, wrist_x, object_x in zip(
        timestamps, wrist_positions, object_positions
    ):
        hands.append(
            HandObservation(
                timestamp_s=timestamp,
                wrist_pose=PoseSE3(
                    wrist_frame,
                    camera,
                    (wrist_x, 0.0, 1.0),
                    (0.0, 0.0, 0.0, 1.0),
                    0.99,
                ),
                keypoints_3d=_keypoints(camera, wrist_x, object_x),
                articulation=(),
                confidence=0.99,
            )
        )
        objects.append(
            ObjectObservation(
                timestamp_s=timestamp,
                object_id="cube",
                pose=PoseSE3(
                    object_frame,
                    camera,
                    (object_x, 0.0, 1.0),
                    (0.0, 0.0, 0.0, 1.0),
                    0.99,
                ),
                state="free" if timestamp < 1.5 else "moving",
                confidence=0.99,
            )
        )
    sequence = PerceptionSequence("0.1.0", tuple(hands), tuple(objects))
    observation_path = args.output_dir / "observations.json"
    sequence.to_json(observation_path)
    intrinsics = PinholeIntrinsics(400.0, 400.0, 320.0, 240.0, 640, 480)
    intrinsics_path = args.output_dir / "camera_intrinsics.json"
    intrinsics_path.write_text(json.dumps(intrinsics.to_dict(), indent=2) + "\n")

    embodiment = EmbodimentDescriptor(
        name="tabletop_hinge_pusher",
        joint_names=("joint1",),
        lower_limits_rad=(-1.5707963267948966,),
        upper_limits_rad=(1.5707963267948966,),
        end_effector_frame="arm_tip",
    )
    retarget_config = LinearRetargetingConfig(
        embodiment=embodiment,
        initial_joint_positions_rad=(-0.7,),
        eef_twist_to_joint_delta=((3.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
    )
    retarget_path = args.output_dir / "retarget_config.json"
    retarget_path.write_text(
        json.dumps(retarget_config.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    video_path = args.output_dir / "human.mp4"
    fps = 30
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480)
    )
    if not writer.isOpened():
        raise SystemExit(f"could not create fixture video: {video_path}")
    for frame_index in range(60):
        timestamp = frame_index / fps
        sample = min(len(timestamps) - 1, int(timestamp / 0.5))
        frame = np.full((480, 640, 3), (32, 36, 42), dtype=np.uint8)
        hand = hands[sample]
        points = [
            tuple(round(value) for value in intrinsics.project(point))
            for point in hand.keypoints_3d
        ]
        for point in points:
            cv2.circle(frame, point, 4, (80, 230, 120), -1, cv2.LINE_AA)
        object_point = intrinsics.project(
            Point3D(camera, objects[sample].pose.translation_m)
        )
        centre = tuple(round(value) for value in object_point)
        cv2.rectangle(
            frame,
            (centre[0] - 16, centre[1] - 16),
            (centre[0] + 16, centre[1] + 16),
            (40, 90, 240),
            3,
        )
        cv2.putText(
            frame,
            "SYNTHETIC TEST FIXTURE - NOT A MODEL RESULT",
            (30, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    print(
        json.dumps(
            {
                "video": str(video_path),
                "observations": str(observation_path),
                "camera_intrinsics": str(intrinsics_path),
                "retarget_config": str(retarget_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
