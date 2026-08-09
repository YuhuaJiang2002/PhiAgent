#!/usr/bin/env python3
"""Render a lightweight articulated USDZ motion preview without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd


def rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "X":
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=float)
    if axis == "Y":
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=float)
    if axis == "Z":
        return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=float)
    raise ValueError(f"unsupported revolute axis: {axis!r}")


def project(points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            0.88 * points[:, 0] - 0.88 * points[:, 1],
            1.25 * points[:, 2] - 0.34 * (points[:, 0] + points[:, 1]),
        )
    )


def transformed_points(
    points: np.ndarray,
    part_matrix: np.ndarray,
    joint: dict[str, object] | None,
    angle: float,
) -> np.ndarray:
    if joint is None:
        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (homogeneous @ part_matrix)[:, :3]
    origin = np.asarray(joint["origin"], dtype=float)
    return points @ rotation(str(joint["axis"]), angle).T + origin


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("usdz", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.fps <= 0 or args.seconds <= 0 or min(args.width, args.height) <= 0:
        raise SystemExit("FPS, duration, width, and height must be positive")
    usdz = args.usdz.expanduser().resolve()
    if not usdz.is_file() or usdz.stat().st_size == 0:
        raise SystemExit(f"USDZ does not exist or is empty: {usdz}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise SystemExit("ffmpeg or imageio-ffmpeg is required") from exc

    output = args.output.expanduser().resolve()
    frames_dir = (
        args.frames_dir.expanduser().resolve()
        if args.frames_dir
        else output.with_name(output.stem + "-frames")
    )
    if output.exists() or frames_dir.exists():
        raise SystemExit("output or frame directory already exists; choose a new demo path")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True)

    stage = Usd.Stage.Open(str(usdz))
    if stage is None:
        raise SystemExit(f"could not open USDZ: {usdz}")
    parts_prim = next(
        (prim for prim in stage.Traverse() if prim.GetPath().name == "parts"),
        None,
    )
    if parts_prim is None:
        raise SystemExit("USDZ has no parts scope")

    joints_by_child: dict[str, dict[str, object]] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        child = str(prim.GetAttribute("mini_articraft:child").Get())
        joints_by_child[child] = {
            "axis": str(prim.GetAttribute("physics:axis").Get()),
            "origin": tuple(prim.GetAttribute("mini_articraft:origin:xyz").Get()),
            "lower": math.radians(float(prim.GetAttribute("physics:lowerLimit").Get())),
            "upper": math.radians(float(prim.GetAttribute("physics:upperLimit").Get())),
        }
    if len(joints_by_child) != 1:
        raise SystemExit("first motion renderer requires exactly one revolute child joint")

    meshes: list[dict[str, object]] = []
    palette = ((35, 92, 166), (229, 132, 34), (35, 38, 45), (96, 170, 112))
    for part_index, part in enumerate(parts_prim.GetChildren()):
        matrix = np.array(part.GetAttribute("xformOp:transform").Get(), dtype=float)
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Mesh" or not prim.GetPath().HasPrefix(part.GetPath()):
                continue
            counts = list(prim.GetAttribute("faceVertexCounts").Get())
            indices = list(prim.GetAttribute("faceVertexIndices").Get())
            faces: list[list[int]] = []
            offset = 0
            for count in counts:
                faces.append(indices[offset : offset + count])
                offset += count
            meshes.append(
                {
                    "part": part.GetName(),
                    "points": np.array(prim.GetAttribute("points").Get(), dtype=float),
                    "faces": faces,
                    "matrix": matrix,
                    "color": palette[part_index % len(palette)],
                }
            )
    if not meshes:
        raise SystemExit("USDZ contains no renderable meshes")

    joint = next(iter(joints_by_child.values()))
    sample_points: list[np.ndarray] = []
    for angle in (float(joint["lower"]), float(joint["upper"])):
        for mesh in meshes:
            sample_points.append(
                transformed_points(
                    mesh["points"],
                    mesh["matrix"],
                    joints_by_child.get(str(mesh["part"])),
                    angle,
                )
            )
    projected = project(np.concatenate(sample_points))
    extent = np.maximum(projected.max(axis=0) - projected.min(axis=0), 1e-6)
    scale = min(args.width * 0.62 / extent[0], args.height * 0.56 / extent[1])
    center = (projected.max(axis=0) + projected.min(axis=0)) / 2
    frame_count = max(2, round(args.seconds * args.fps))
    font = ImageFont.load_default(size=24)
    small_font = ImageFont.load_default(size=17)

    for frame_index in range(frame_count):
        phase = frame_index / (frame_count - 1)
        blend = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
        angle = float(joint["lower"]) + blend * (
            float(joint["upper"]) - float(joint["lower"])
        )
        image = Image.new("RGB", (args.width, args.height), (238, 241, 246))
        draw = ImageDraw.Draw(image)
        draw.text((42, 34), "ArtiCraft handover-case asset demo", fill=(22, 28, 38), font=font)
        draw.text(
            (42, 68),
            "Articulated asset preview - not a physics-verified robot handover",
            fill=(155, 55, 45),
            font=small_font,
        )
        draw.text(
            (42, args.height - 54),
            f"lid_hinge: {math.degrees(angle):05.1f} deg",
            fill=(40, 48, 60),
            font=small_font,
        )

        view_vector = np.array((1.0, 1.0, 0.55))
        faces_to_draw: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []
        for mesh in meshes:
            world = transformed_points(
                mesh["points"],
                mesh["matrix"],
                joints_by_child.get(str(mesh["part"])),
                angle,
            )
            screen = project(world)
            screen = (screen - center) * scale
            screen[:, 0] += args.width / 2
            screen[:, 1] = args.height / 2 - screen[:, 1]
            for face in mesh["faces"]:
                if len(face) >= 3:
                    normal = np.cross(
                        world[face[1]] - world[face[0]],
                        world[face[2]] - world[face[0]],
                    )
                    if float(normal @ view_vector) <= 0:
                        continue
                polygon = [(float(screen[i, 0]), float(screen[i, 1])) for i in face]
                depth = float(np.mean(world[face] @ view_vector))
                faces_to_draw.append((depth, polygon, mesh["color"]))
        for _, polygon, color in sorted(faces_to_draw):
            draw.polygon(polygon, fill=color, outline=(20, 25, 32), width=2)
        image.save(frames_dir / f"{frame_index:06d}.png")

    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    metadata = {
        "source_usdz": str(usdz),
        "output": str(output),
        "frame_count": frame_count,
        "fps": args.fps,
        "duration_s": frame_count / args.fps,
        "resolution": [args.width, args.height],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "limitations": [
            "This is a geometric articulation preview, not a robot handover.",
            "No grasp, contact, collision, or object-transfer physics are evaluated.",
        ],
        "command": command,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"VIDEO={output}")
    print(f"METADATA={output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
