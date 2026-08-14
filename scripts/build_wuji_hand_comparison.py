#!/usr/bin/env python3
"""Build an auditable human-hand to official Wuji Hand comparison.

Heavy dependencies stay optional: this module imports MediaPipe, Pinocchio,
MuJoCo, OpenCV, and the Wuji retargeter only after CLI validation. The Wuji
repositories and checkpoints/assets are external inputs and are never vendored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def ffprobe_video(path: Path) -> dict[str, object]:
    payload = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,duration,nb_read_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(payload)


def direct_detection_mask(frames: list[object | None]) -> list[bool]:
    """Recover direct detections from Wuji's identity-preserving hold behavior.

    ``VideoMediaPipe`` allocates a new landmark array for a detected frame and
    reuses the previous object when detection is absent. Object identity thus
    distinguishes detections from held observations without modifying upstream.
    """

    result: list[bool] = []
    previous: object | None = None
    for frame in frames:
        direct = frame is not None and frame is not previous
        result.append(direct)
        if frame is not None:
            previous = frame
    return result


def urdf_velocity_limits(path: Path, joint_names: list[str]) -> list[float]:
    root = ET.parse(path).getroot()
    by_name: dict[str, float] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "velocity" in limit.attrib:
            by_name[joint.attrib["name"]] = float(limit.attrib["velocity"])
    missing = [name for name in joint_names if name not in by_name]
    if missing:
        raise ValueError(f"URDF velocity limits missing for joints: {missing}")
    return [by_name[name] for name in joint_names]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--wuji-retargeting-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--camera-azimuth", type=float, default=180.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance", type=float, default=0.34)
    return parser.parse_args()


def _check_paths(args: argparse.Namespace) -> dict[str, Path]:
    source = args.source_video.expanduser().resolve()
    root = args.wuji_retargeting_root.expanduser().resolve()
    config = root / "example" / "config" / "adaptive_analytical_video.yaml"
    model_root = root / "wuji_retargeting" / "wuji-description" / "hand" / "body"
    paths = {
        "source": source,
        "root": root,
        "config": config,
        "mjcf": model_root / "mjcf" / f"{args.hand_side}.xml",
        "urdf": model_root / "urdf" / f"{args.hand_side}.urdf",
    }
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required input: " + ", ".join(missing))
    return paths


def _gradient_background(np, cv2, width: int, height: int):
    top = np.array([18, 29, 52], dtype=np.float32)
    bottom = np.array([5, 10, 23], dtype=np.float32)
    mix = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    canvas = top[None, None, :] * (1.0 - mix) + bottom[None, None, :] * mix
    canvas = np.repeat(canvas, width, axis=1)
    for x in range(0, width, 80):
        cv2.line(canvas, (x, 0), (x, height), (38, 50, 74), 1, cv2.LINE_AA)
    for y in range(0, height, 80):
        cv2.line(canvas, (0, y), (width, y), (38, 50, 74), 1, cv2.LINE_AA)
    return canvas.astype(np.uint8)


def _composite_render(np, cv2, rendered, background):
    bright = np.max(rendered, axis=2).astype(np.float32)
    alpha = np.clip(bright / 10.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)[:, :, None]
    hand = np.clip((rendered.astype(np.float32) / 255.0) ** 0.82 * 255.0, 0, 255)
    return np.clip(hand * alpha + background.astype(np.float32) * (1.0 - alpha), 0, 255).astype(
        np.uint8
    )


def _label_frame(cv2, frame, frame_index: int, total_frames: int):
    header = 48
    cv2.rectangle(frame, (0, 0), (frame.shape[1], header), (7, 12, 25), -1)
    cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (91, 108, 147), 2)
    cv2.putText(frame, "HUMAN SOURCE", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 242, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "WUJI HAND  |  OFFICIAL MODEL SIMULATION",
        (frame.shape[1] // 2 + 24, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (126, 231, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"MediaPipe-21 -> official Wuji retargeter -> 20-DOF q   |   {frame_index + 1:03d}/{total_frames}",
        (frame.shape[1] // 2 + 24, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (181, 193, 219),
        1,
        cv2.LINE_AA,
    )
    return frame


def main() -> int:
    args = _parse_args()
    paths = _check_paths(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        import cv2
        import mediapipe
        import mujoco
        import numpy as np
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Wuji comparison dependencies are optional. Install the pinned Wuji "
            "retargeting video stack, MuJoCo, MediaPipe, OpenCV, NumPy, and PyYAML. "
            f"Missing import: {exc}"
        ) from exc

    root = paths["root"]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "example"))
    from input_devices.video_mediapipe import VideoMediaPipe
    from wuji_retargeting import Retargeter

    full_config = yaml.safe_load(paths["config"].read_text())
    started = time.perf_counter()
    device = VideoMediaPipe(
        str(paths["source"]),
        hand_side=args.hand_side,
        loop=False,
        video_config=full_config.get("video_input", {}),
        show_video=False,
    )
    landmarks = device.get_landmarks()
    detection_mask = direct_detection_mask(landmarks)
    if not landmarks or landmarks[0] is None:
        raise RuntimeError("no hand was detected in the first frame")

    retargeter = Retargeter.from_yaml(str(paths["config"]), args.hand_side)
    q_values = []
    costs = []
    for index, landmark in enumerate(landmarks):
        if landmark is None:
            raise RuntimeError(f"landmark hold has no prior observation at frame {index}")
        q, verbose = retargeter.retarget_verbose(landmark, apply_filter=True)
        q_values.append(q)
        costs.append(float(verbose["cost"]))
    q = np.asarray(q_values, dtype=np.float32)
    fps = float(device.fps)
    qdot = np.gradient(q, 1.0 / fps, axis=0).astype(np.float32)

    model = mujoco.MjModel.from_xml_path(str(paths["mjcf"]))
    data = mujoco.MjData(model)
    if model.nq != q.shape[1]:
        raise RuntimeError(f"trajectory/model mismatch: q={q.shape[1]}, model.nq={model.nq}")
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    optimizer_joint_names = list(retargeter.optimizer.robot.dof_joint_names)
    if model_joint_names != optimizer_joint_names:
        raise RuntimeError("official URDF and MJCF joint orders differ; refusing an identity remap")

    lower = model.jnt_range[:, 0]
    upper = model.jnt_range[:, 1]
    limit_violations = int(np.count_nonzero((q < lower[None, :] - 1e-5) | (q > upper[None, :] + 1e-5)))
    if limit_violations:
        raise RuntimeError(f"trajectory has {limit_violations} joint-limit violations")
    velocity_limits = np.asarray(
        urdf_velocity_limits(paths["urdf"], model_joint_names), dtype=np.float32
    )
    velocity_violations = int(
        np.count_nonzero(np.abs(qdot) > velocity_limits[None, :] + 1e-5)
    )
    if velocity_violations:
        raise RuntimeError(f"trajectory has {velocity_violations} URDF velocity-limit violations")
    q_delta_norm = np.linalg.norm(np.diff(q, axis=0), axis=1)
    qddot = np.gradient(qdot, 1.0 / fps, axis=0)

    trajectory_path = output_dir / "wuji-hand-official-model-retarget-q.npz"
    np.savez_compressed(
        trajectory_path,
        q=q,
        qdot=qdot,
        timestamps=np.arange(q.shape[0], dtype=np.float64) / fps,
        landmarks=np.asarray(landmarks, dtype=np.float32),
        direct_detection=np.asarray(detection_mask, dtype=np.bool_),
        joint_names=np.asarray(model_joint_names),
        optimizer_cost=np.asarray(costs, dtype=np.float32),
    )

    width, height, header = 1280, 720, 48
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance
    camera.lookat[:] = [0.0, 0.0, 0.055]
    render_options = mujoco.MjvOption()
    render_options.geomgroup[:] = 0
    render_options.geomgroup[1] = 1
    background = _gradient_background(np, cv2, width, height)

    video_path = output_dir / "human-to-wuji-hand-official-model-comparison-20p7s.mp4"
    command = [
        args.ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width * 2}x{height + header}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    source_capture = cv2.VideoCapture(str(paths["source"]))
    poster_frame = None
    render_started = time.perf_counter()
    try:
        for index in range(q.shape[0]):
            ok, source_bgr = source_capture.read()
            if not ok:
                raise RuntimeError(f"source decode stopped at frame {index}")
            source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            source_rgb = cv2.resize(source_rgb, (width, height), interpolation=cv2.INTER_AREA)
            data.qpos[:] = q[index]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=render_options)
            robot = _composite_render(np, cv2, renderer.render(), background)
            body = np.hstack([source_rgb, robot])
            frame = np.zeros((height + header, width * 2, 3), dtype=np.uint8)
            frame[header:] = body
            _label_frame(cv2, frame, index, q.shape[0])
            if index == q.shape[0] // 2:
                poster_frame = frame.copy()
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
    finally:
        source_capture.release()
        renderer.close()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg encoder failed with exit code {return_code}")
    render_seconds = time.perf_counter() - render_started

    poster_path = output_dir / "human-to-wuji-hand-official-model-comparison-poster.jpg"
    if poster_frame is None:
        raise RuntimeError("poster frame was not captured")
    cv2.imwrite(str(poster_path), cv2.cvtColor(poster_frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    probe = ffprobe_video(video_path)
    decoded_frames = int(probe["streams"][0]["nb_read_frames"])
    if decoded_frames != q.shape[0]:
        raise RuntimeError(f"decoded frame mismatch: {decoded_frames} != {q.shape[0]}")

    total_seconds = time.perf_counter() - started
    manifest = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "claim": "Vision-derived human motion retargeted to an official Wuji Hand simulation model.",
        "non_claims": [
            "This is not footage of physical Wuji hardware.",
            "Monocular MediaPipe landmarks do not establish metric depth, contact force, or real execution.",
        ],
        "source": {
            "path": paths["source"].name,
            "sha256": sha256_file(paths["source"]),
            "frames": int(q.shape[0]),
            "fps": fps,
        },
        "upstream": {
            "repository": "https://github.com/wuji-technology/wuji-retargeting",
            "commit": git_head(root),
            "description_commit": git_head(root / "wuji_retargeting" / "wuji-description"),
            "config_sha256": sha256_file(paths["config"]),
            "mjcf_sha256": sha256_file(paths["mjcf"]),
            "urdf_sha256": sha256_file(paths["urdf"]),
            "model": "Wuji Hand (right/left selected by hand_side; not Wuji Hand 2 Beta)",
        },
        "trajectory": {
            "artifact": trajectory_path.name,
            "artifact_sha256": sha256_file(trajectory_path),
            "q_shape": list(q.shape),
            "qdot_shape": list(qdot.shape),
            "direct_detection_frames": int(sum(detection_mask)),
            "held_observation_frames": int(len(detection_mask) - sum(detection_mask)),
            "joint_limit_violations": limit_violations,
            "velocity_limit_violations": velocity_violations,
            "max_velocity_limit_ratio": float(
                np.max(np.abs(qdot) / velocity_limits[None, :])
            ),
            "qdot_abs_max_rad_s": float(np.max(np.abs(qdot))),
            "qddot_abs_p95_rad_s2": float(np.percentile(np.abs(qddot), 95)),
            "qddot_abs_max_rad_s2": float(np.max(np.abs(qddot))),
            "frozen_transition_count_at_1e_4_rad": int(
                np.count_nonzero(q_delta_norm <= 1e-4)
            ),
            "q_delta_norm_mean_rad": float(np.mean(q_delta_norm)),
            "q_delta_norm_p95_rad": float(np.percentile(q_delta_norm, 95)),
            "optimizer_cost_mean": float(np.mean(costs)),
            "optimizer_cost_p95": float(np.percentile(costs, 95)),
            "optimizer_cost_max": float(np.max(costs)),
        },
        "output": {
            "video": video_path.name,
            "video_sha256": sha256_file(video_path),
            "poster": poster_path.name,
            "poster_sha256": sha256_file(poster_path),
            "probe": probe,
            "decoded_frames": decoded_frames,
            "render_seconds": render_seconds,
            "render_fps": float(q.shape[0] / render_seconds),
            "total_seconds": total_seconds,
            "end_to_end_fps": float(q.shape[0] / total_seconds),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "mediapipe": mediapipe.__version__,
            "opencv": cv2.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
    }
    manifest_path = output_dir / "human-to-wuji-hand-official-model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest["output"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
