#!/usr/bin/env python3
"""Re-render robot replacement with per-frame person masks and motion warping."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.agent.epl_video_evolution import ReplacementParameters
from scripts.build_epl_agentic_robot_replacement import (  # noqa: E402
    _object_mask,
    _odd,
    _read_frame,
    _source_info,
    _writer,
)


def _person_mask_frame(
    cv2: object,
    mp: object,
    np: object,
    frame: object,
    roi_mask: object,
) -> object:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    score = mp.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=1
    ).process(rgb).segmentation_mask
    person = (score >= 0.22).astype(np.uint8) * 255
    person = cv2.bitwise_and(person, roi_mask)
    person = cv2.morphologyEx(
        person,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    person = cv2.dilate(
        person,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    return person


def _render(
    *,
    cv2: object,
    mp: object,
    np: object,
    ffmpeg: str,
    source: Path,
    output: Path,
    robot_anchor: object,
    anchor_mask: object,
    subject_roi: object,
    parameters: ReplacementParameters,
    anchor_index: int,
    flow_width: int,
) -> None:
    source_info = _source_info(cv2, source)
    width = int(source_info["width"])
    height = int(source_info["height"])
    fps = float(source_info["fps"])
    frame_count = int(source_info["frames"])
    flow_height = max(2, round(height * flow_width / width))
    source_anchor = _read_frame(cv2, source, anchor_index)
    anchor_small = cv2.resize(source_anchor, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
    anchor_gray = cv2.cvtColor(anchor_small, cv2.COLOR_BGR2GRAY)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    writer = _writer(ffmpeg, output, width, height, fps)
    capture = cv2.VideoCapture(str(source))
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_small = cv2.resize(frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
            current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            person_score = segmenter.process(rgb).segmentation_mask
            person_mask = (person_score >= 0.22).astype(np.uint8) * 255
            person_mask = cv2.bitwise_and(person_mask, subject_roi)
            person_mask = cv2.morphologyEx(
                person_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            )
            person_mask = cv2.dilate(
                person_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            )

            if parameters.flow_strength > 0:
                flow = cv2.calcOpticalFlowFarneback(
                    current_gray,
                    anchor_gray,
                    None,
                    0.5,
                    4,
                    25,
                    4,
                    7,
                    1.5,
                    cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
                )
                if parameters.flow_blur_pixels > 1:
                    flow = cv2.GaussianBlur(
                        flow,
                        (parameters.flow_blur_pixels, parameters.flow_blur_pixels),
                        0,
                    )
                clip_at_flow_scale = parameters.flow_clip_pixels * flow_width / width
                magnitude = np.linalg.norm(flow, axis=2)
                scale = np.minimum(1.0, clip_at_flow_scale / np.maximum(magnitude, 1e-6))
                flow *= scale[..., None]
                full_flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
                full_flow[..., 0] *= width / flow_width
                full_flow[..., 1] *= height / flow_height
                motion_gate = cv2.absdiff(current_gray, anchor_gray)
                motion_gate = (motion_gate >= 7).astype(np.uint8) * 255
                motion_gate = cv2.bitwise_and(
                    motion_gate,
                    cv2.resize(
                        anchor_mask,
                        (flow_width, flow_height),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                )
                motion_gate = cv2.dilate(
                    motion_gate,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                )
                motion_gate = cv2.GaussianBlur(motion_gate, (15, 15), 0)
                full_gate = cv2.resize(
                    motion_gate, (width, height), interpolation=cv2.INTER_LINEAR
                ).astype(np.float32) / 255.0
                map_x = grid_x + full_flow[..., 0] * parameters.flow_strength * full_gate
                map_y = grid_y + full_flow[..., 1] * parameters.flow_strength * full_gate
                warped_robot = cv2.remap(
                    robot_anchor,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101,
                )
                warped_mask = cv2.remap(
                    anchor_mask,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            else:
                warped_robot = robot_anchor
                warped_mask = anchor_mask

            binary_mask = cv2.bitwise_or(
                cv2.bitwise_or(
                    (warped_mask >= 96).astype(np.uint8) * 255,
                    person_mask,
                ),
                anchor_mask,
            )
            if parameters.mask_dilation_pixels:
                size = _odd(parameters.mask_dilation_pixels * 2 + 1)
                binary_mask = cv2.dilate(
                    binary_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                )
            if parameters.mask_feather_pixels:
                alpha = cv2.GaussianBlur(
                    binary_mask,
                    (0, 0),
                    parameters.mask_feather_pixels,
                ).astype(np.float32) / 255.0
            else:
                alpha = binary_mask.astype(np.float32) / 255.0
            candidate = np.rint(
                frame.astype(np.float32) * (1.0 - alpha[..., None])
                + warped_robot.astype(np.float32) * alpha[..., None]
            ).astype(np.uint8)
            base_object_mask = _object_mask(cv2, np, frame, 0)
            protected_object_mask = base_object_mask
            if parameters.object_dilation_pixels:
                size = _odd(parameters.object_dilation_pixels * 2 + 1)
                protected_object_mask = cv2.dilate(
                    base_object_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                )
            if parameters.protect_objects:
                candidate[protected_object_mask > 0] = frame[protected_object_mask > 0]

            assert writer.stdin is not None
            writer.stdin.write(candidate.tobytes())
            decoded += 1
    finally:
        capture.release()
        segmenter.close()
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg writer failed with code {return_code}")
    if decoded != frame_count:
        raise RuntimeError(f"decoded {decoded}/{frame_count} frames")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--robot-anchor", type=Path, required=True)
    parser.add_argument("--anchor-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-seconds", type=float, default=11.5)
    parser.add_argument("--flow-strength", type=float, default=0.48)
    parser.add_argument("--flow-width", type=int, default=256)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    args = parser.parse_args()

    import cv2
    import mediapipe as mp
    import numpy as np

    source = args.source.expanduser().resolve()
    robot_anchor = cv2.imread(str(args.robot_anchor.expanduser().resolve()))
    anchor_mask = cv2.imread(str(args.anchor_mask.expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    if robot_anchor is None or anchor_mask is None:
        raise RuntimeError("cannot decode robot anchor or mask")
    source_info = _source_info(cv2, source)
    width, height = int(source_info["width"]), int(source_info["height"])
    robot_anchor = cv2.resize(robot_anchor, (width, height), interpolation=cv2.INTER_LANCZOS4)
    anchor_mask = cv2.resize(anchor_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    subject_roi = np.zeros((height, width), dtype=np.uint8)
    subject_roi[
        round(height * 0.035) : round(height * 0.93),
        round(width * 0.47) : round(width * 0.88),
    ] = 255
    anchor_index = min(
        int(source_info["frames"]) - 1,
        round(args.anchor_seconds * float(source_info["fps"])),
    )
    parameters = ReplacementParameters(
        flow_strength=args.flow_strength,
        flow_blur_pixels=3,
        flow_clip_pixels=24.0,
        mask_dilation_pixels=2,
        mask_feather_pixels=1.5,
        protect_objects=True,
        object_dilation_pixels=2,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _render(
        cv2=cv2,
        mp=mp,
        np=np,
        ffmpeg=str(args.ffmpeg),
        source=source,
        output=output,
        robot_anchor=robot_anchor,
        anchor_mask=anchor_mask,
        subject_roi=subject_roi,
        parameters=parameters,
        anchor_index=anchor_index,
        flow_width=args.flow_width,
    )
    subprocess.run(
        [str(args.ffmpeg), "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
