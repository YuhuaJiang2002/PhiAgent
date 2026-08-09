"""Frame-explicit EPL overlays for source-video inspection."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from phiagent.perception.camera import PinholeIntrinsics
from phiagent.perception.schema import HandObservation, ObjectObservation, PerceptionSequence
from phiagent.physical_language.schema import EPLChunk, EPLSequence, Point3D

HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True)
class OverlayPrimitives:
    hand_points_px: tuple[tuple[int, int], ...]
    hand_edges_px: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    contact_points_px: tuple[tuple[int, int], ...]
    eef_trajectory_px: tuple[tuple[int, int], ...]
    object_axes_px: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    phase_label: str


def _pixel(intrinsics: PinholeIntrinsics, point: Point3D) -> tuple[int, int]:
    x, y = intrinsics.project(point)
    return (round(x), round(y))


def _object_axes(
    observation: ObjectObservation | None,
    intrinsics: PinholeIntrinsics,
    axis_length_m: float,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    if observation is None:
        return ()
    pose = observation.pose
    origin = pose.transform_point(Point3D(pose.source_frame, (0.0, 0.0, 0.0)))
    origin_px = _pixel(intrinsics, origin)
    axes = []
    for endpoint in (
        (axis_length_m, 0.0, 0.0),
        (0.0, axis_length_m, 0.0),
        (0.0, 0.0, axis_length_m),
    ):
        projected = pose.transform_point(Point3D(pose.source_frame, endpoint))
        axes.append((origin_px, _pixel(intrinsics, projected)))
    return tuple(axes)


def build_overlay_primitives(
    hand: HandObservation,
    object_observation: ObjectObservation | None,
    chunk: EPLChunk,
    eef_history: tuple[Point3D, ...],
    intrinsics: PinholeIntrinsics,
    axis_length_m: float = 0.06,
) -> OverlayPrimitives:
    """Project one aligned observation without silently changing coordinates."""

    points = tuple(_pixel(intrinsics, point) for point in hand.keypoints_3d)
    edges = tuple((points[start], points[end]) for start, end in HAND_EDGES)
    return OverlayPrimitives(
        hand_points_px=points,
        hand_edges_px=edges,
        contact_points_px=tuple(
            _pixel(intrinsics, point) for point in chunk.contact_points
        ),
        eef_trajectory_px=tuple(_pixel(intrinsics, point) for point in eef_history),
        object_axes_px=_object_axes(object_observation, intrinsics, axis_length_m),
        phase_label=chunk.phase.value,
    )


class EPLVisualizer:
    """Render hand, object, contact, EEF, and phase overlays into an MP4."""

    def render(
        self,
        video_path: Path,
        observations: PerceptionSequence,
        epl: EPLSequence,
        intrinsics: PinholeIntrinsics,
        output_path: Path,
    ) -> dict[str, int | float | str]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment-specific.
            raise RuntimeError("EPL visualization requires opencv-python") from exc
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open source video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise ValueError("source video has invalid FPS or dimensions")
        if (width, height) != (intrinsics.width, intrinsics.height):
            capture.release()
            raise ValueError(
                "camera intrinsics dimensions do not match source video: "
                f"{intrinsics.width}x{intrinsics.height} != {width}x{height}"
            )
        if not epl.chunks:
            capture.release()
            raise ValueError("cannot visualize an empty EPL sequence")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"could not create output video: {output_path}")

        observation_times = [item.timestamp_s for item in observations.hands]
        chunk_starts = [item.start_s for item in epl.chunks]
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_s = frame_index / fps
                observation_index = min(
                    len(observation_times) - 1,
                    max(0, bisect.bisect_right(observation_times, timestamp_s) - 1),
                )
                chunk_index = min(
                    len(chunk_starts) - 1,
                    max(0, bisect.bisect_right(chunk_starts, timestamp_s) - 1),
                )
                chunk = epl.chunks[chunk_index]
                history = tuple(
                    Point3D(item.wrist_pose.target_frame, item.wrist_pose.translation_m)
                    for item in epl.chunks[: chunk_index + 1]
                )
                primitives = build_overlay_primitives(
                    observations.hands[observation_index],
                    observations.objects[observation_index],
                    chunk,
                    history,
                    intrinsics,
                )
                for start, end in primitives.hand_edges_px:
                    cv2.line(frame, start, end, (80, 220, 80), 2, cv2.LINE_AA)
                for point in primitives.hand_points_px:
                    cv2.circle(frame, point, 3, (40, 255, 255), -1, cv2.LINE_AA)
                for point in primitives.contact_points_px:
                    cv2.circle(frame, point, 7, (0, 0, 255), 2, cv2.LINE_AA)
                for start, end in zip(
                    primitives.eef_trajectory_px,
                    primitives.eef_trajectory_px[1:],
                ):
                    cv2.line(frame, start, end, (255, 100, 40), 2, cv2.LINE_AA)
                for (start, end), color in zip(
                    primitives.object_axes_px,
                    ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
                ):
                    cv2.line(frame, start, end, color, 3, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"EPL phase: {primitives.phase_label}",
                    (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                frame_index += 1
        finally:
            capture.release()
            writer.release()
        return {
            "frames": frame_index,
            "fps": fps,
            "width": width,
            "height": height,
            "output": str(output_path),
        }
