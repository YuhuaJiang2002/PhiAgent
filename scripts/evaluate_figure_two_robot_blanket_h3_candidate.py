#!/usr/bin/env python3
"""Fail-closed RGB audit for a photorealistic two-robot quilt-fold proposal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence


AUTO_GATE_IDS = {
    "artifact_hash_binding",
    "gpu_and_runtime_preflight",
    "pinned_model_revision",
    "media_contract_1024x768_24fps_192frames",
    "exact_first_frame_ssim_at_least_0_985",
    "single_uninterrupted_shot_no_cut_crossfade_or_teleport",
    "fixed_camera_background_and_lighting",
    "terminal_motion_below_1_pixel_rms_for_last_12_frames",
}
CONCLUSIVE_AUTO_GATE_IDS = {
    "artifact_hash_binding",
    "gpu_and_runtime_preflight",
    "pinned_model_revision",
    "media_contract_1024x768_24fps_192frames",
    "exact_first_frame_ssim_at_least_0_985",
    "terminal_motion_below_1_pixel_rms_for_last_12_frames",
}
EXPECTED_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(command: Sequence[str], timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stderr[-4000:]
        )
    return completed


def _probe(video: Path) -> dict[str, Any]:
    completed = _run(
        (
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,"
            "nb_frames,nb_read_frames,duration:format=duration",
            "-of",
            "json",
            str(video),
        )
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError("candidate must contain exactly one video stream")
    return {"stream": streams[0], "format": payload.get("format", {})}


def _ssim_to_still(
    video: Path,
    still: Path,
    *,
    start_frame: int,
    frame_count: int,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[float, str]:
    crop_filter = ""
    if crop is not None:
        width, height, x_value, y_value = crop
        crop_filter = f",crop={width}:{height}:{x_value}:{y_value}"
    graph = (
        f"[0:v]trim=start_frame={start_frame},setpts=PTS-STARTPTS"
        f"{crop_filter},format=yuv420p[a];"
        f"[1:v]setpts=PTS-STARTPTS{crop_filter},format=yuv420p[b];"
        "[a][b]ssim"
    )
    completed = _run(
        (
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(video),
            "-loop",
            "1",
            "-framerate",
            "24",
            "-i",
            str(still),
            "-filter_complex",
            graph,
            "-frames:v",
            str(frame_count),
            "-an",
            "-f",
            "null",
            "-",
        )
    )
    matches = re.findall(r"All:([0-9.]+)", completed.stderr)
    if not matches:
        raise ValueError("ffmpeg did not report SSIM")
    return float(matches[-1]), completed.stderr[-4000:]


def _scene_cuts(video: Path) -> tuple[list[float], str]:
    completed = _run(
        (
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(video),
            "-vf",
            "select='gt(scene,0.30)',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        )
    )
    times = [
        float(value)
        for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", completed.stderr)
    ]
    return times, completed.stderr[-8000:]


def _terminal_flow_rms(
    video: Path,
    *,
    start_frame: int,
    frame_count: int,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "terminal pixel-motion evaluation requires OpenCV and NumPy"
        ) from error

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open candidate video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    gray_frames = []
    try:
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    if len(gray_frames) != frame_count:
        raise ValueError(
            f"expected {frame_count} terminal frames at {start_frame}, "
            f"decoded {len(gray_frames)}"
        )

    pairwise_rms = []
    for previous, current in zip(gray_frames, gray_frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            4,
            21,
            5,
            7,
            1.5,
            0,
        )
        if not np.isfinite(flow).all():
            raise ValueError("terminal optical flow contains non-finite values")
        pairwise_rms.append(
            float(np.sqrt(np.mean(np.sum(np.square(flow), axis=2))))
        )
    return {
        "algorithm": "OpenCV Farneback dense optical flow",
        "start_frame": start_frame,
        "frame_count": frame_count,
        "transition_count": len(pairwise_rms),
        "pairwise_vector_magnitude_rms_pixels": pairwise_rms,
        "aggregate_vector_magnitude_rms_pixels": float(
            np.sqrt(np.mean(np.square(pairwise_rms)))
        ),
        "maximum_pairwise_vector_magnitude_rms_pixels": max(pairwise_rms),
    }


def _gate(gate_id: str, state: str, **evidence: Any) -> dict[str, Any]:
    if state not in {"PASS", "FAIL", "UNAVAILABLE"}:
        raise ValueError(f"invalid gate state: {state}")
    return {"gate_id": gate_id, "state": state, "evidence": evidence}


def _find_candidate_video(candidate_dir: Path) -> Path:
    candidates = sorted(candidate_dir.glob("candidates/*/seed-*/candidate.mp4"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one generated video beneath {candidate_dir}, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def _review_packet(
    output: Path,
    video: Path,
    candidate_sha256: str,
    manual_gate_ids: list[str],
) -> dict[str, Any]:
    contact_sheet = output / "native-review-contact-sheet.jpg"
    command = (
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video),
        "-vf",
        "select='not(mod(n,12))',scale=512:384:flags=lanczos,"
        "tile=4x4:padding=6:margin=6",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(contact_sheet),
    )
    _run(command)
    frames_dir = output / "native-review-frames"
    frames_dir.mkdir()
    frame_indices = (0, 12, 34, 48, 82, 96, 115, 140, 158, 175, 191)
    for index in frame_indices:
        _run(
            (
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"select=eq(n\\,{index})",
                "-vsync",
                "0",
                "-frames:v",
                "1",
                str(frames_dir / f"frame-{index:03d}.png"),
            )
        )
    template = {
        "schema_version": "1.0.0",
        "candidate_sha256": candidate_sha256,
        "review_scope": "native_resolution_full_video",
        "reviewer": "UNASSIGNED",
        "reviewed_at": None,
        "instructions": (
            "Watch the original candidate at native 1024x768 resolution from start to "
            "finish. Set every gate to PASS or FAIL and cite time ranges. Do not infer "
            "hidden contact or material state. Any FAIL or missing evidence rejects."
        ),
        "gates": [
            {"gate_id": gate_id, "state": "UNAVAILABLE", "time_ranges": [], "notes": ""}
            for gate_id in manual_gate_ids
        ],
        "overall_visual_acceptance": False,
        "claim_boundary": (
            "This review can accept visual-generation quality only; it cannot establish "
            "real-robot execution or any metric/force/safety gate."
        ),
    }
    _write_json(output / "native-review-template.json", template)
    return {
        "contact_sheet": str(contact_sheet),
        "contact_sheet_sha256": _sha256(contact_sheet),
        "frame_indices": list(frame_indices),
        "template": str(output / "native-review-template.json"),
    }


def main() -> int:
    args = _parser().parse_args()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse evaluation directory: {output}")
    output.mkdir(parents=True)

    config = _read_json(campaign_dir / "inputs/campaign-config.json")
    spec = _read_json(campaign_dir / "inputs/generation-spec.json")
    manifest = _read_json(campaign_dir / "manifest.json")
    metadata = _read_json(candidate_dir / "metadata.json")
    gpu_selection = _read_json(candidate_dir / "gpu_selection.json")
    runtime_probe = _read_json(candidate_dir / "runtime_probe.json")
    checkpoint = _read_json(candidate_dir / "checkpoint_evidence.json")
    initial = campaign_dir / "inputs/initial-frame.png"
    video = _find_candidate_video(candidate_dir)
    candidate_record = _read_json(video.parent / "tri-evolve-record.json")
    candidate_sha256 = _sha256(video)
    completed_candidates = metadata.get("completed_candidates", [])

    binding_ok = (
        _sha256(initial) == config["initial_frame"]["sha256"]
        and _sha256(campaign_dir / "inputs/generation-spec.json")
        == manifest["hashes"]["generation_spec_file"]
        and metadata.get("spec_sha256") == manifest["hashes"]["generation_spec_file"]
        and metadata.get("challenge_frame_sha256") == _sha256(initial)
        and metadata.get("challenge_sha256") == spec["challenge"]["challenge_sha256"]
        and metadata.get("harness_version") == spec["cases"][0]["harness_version"]
        and metadata.get("decision") == "GENERATION_COMPLETE_PENDING_EVALUATION"
        and len(completed_candidates) == 1
        and completed_candidates[0].get("video_sha256") == candidate_sha256
        and candidate_record.get("video_sha256") == candidate_sha256
        and candidate_record.get("challenge_sha256")
        == spec["challenge"]["challenge_sha256"]
        and candidate_record.get("task_plan_sha256")
        == spec["cases"][0]["task_plan_sha256"]
        and candidate_record.get("harness_sha256")
        == spec["cases"][0]["harness_sha256"]
        and candidate_record.get("prompt_sha256")
        == spec["cases"][0]["prompt_sha256"]
    )
    selected = gpu_selection.get("selected", [])
    expected_gpu_count = int(spec["compute"]["num_gpus"])
    expected_gpu_indices = [
        int(value) for value in spec["compute"]["physical_gpu_indices"]
    ]
    gpu_ok = (
        len(selected) == expected_gpu_count
        and [int(item.get("physical_index", -1)) for item in selected]
        == expected_gpu_indices
        and all(item.get("name") == "NVIDIA H200" for item in selected)
        and all(
            int(item.get("free_mib", -1))
            >= int(spec["compute"]["minimum_free_mib"])
            for item in selected
        )
        and all(not item.get("compute_processes") for item in selected)
        and all(str(item.get("uuid", "")).startswith("GPU-") for item in selected)
        and runtime_probe.get("cuda_available") is True
        and runtime_probe.get("device_count") == expected_gpu_count
        and runtime_probe.get("torch") == "2.11.0+cu128"
        and runtime_probe.get("cuda") == "12.8"
        and spec["model"]["runtime"].get("approximate_attention") is False
    )
    revision_ok = (
        checkpoint.get("revision") == EXPECTED_REVISION
        and checkpoint.get("partition") == "Ref2VA"
        and checkpoint.get("dtype") == "bfloat16"
        and checkpoint.get("quantization") is None
        and checkpoint.get("checkpoint_root") == spec["model"]["checkpoint_root"]
        and len(checkpoint.get("transformer_shards", [])) == 13
        and len(checkpoint.get("text_encoder_shards", [])) == 14
        and all(
            int(item.get("bytes", 0)) > 0
            for item in checkpoint.get("transformer_shards", [])
            + checkpoint.get("text_encoder_shards", [])
        )
    )

    probe = _probe(video)
    stream = probe["stream"]
    frame_count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or -1)
    fps = float(Fraction(stream.get("avg_frame_rate") or stream["r_frame_rate"]))
    duration = float(stream.get("duration") or probe["format"].get("duration") or -1)
    media_ok = (
        int(stream["width"]) == 1024
        and int(stream["height"]) == 768
        and frame_count == 192
        and abs(fps - 24.0) < 1e-9
        and abs(duration - 8.0) <= 1.0 / 24.0
    )
    first_ssim, first_log = _ssim_to_still(
        video, initial, start_frame=0, frame_count=1
    )
    cut_times, cut_log = _scene_cuts(video)
    background_patches = {
        "upper_center_room": (480, 250, 272, 0),
        "bed_and_window": (250, 180, 370, 180),
        "table_front": (760, 120, 132, 630),
    }
    background_results = {}
    for patch_id, crop in background_patches.items():
        value, _ = _ssim_to_still(
            video,
            initial,
            start_frame=0,
            frame_count=192,
            crop=crop,
        )
        background_results[patch_id] = value
    minimum_background_ssim = min(background_results.values())
    final_frame = output / "final-frame.png"
    _run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video),
            "-vf",
            "select=eq(n\\,191)",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(final_frame),
        )
    )
    terminal_flow = _terminal_flow_rms(
        video,
        start_frame=180,
        frame_count=12,
    )
    terminal_flow_ok = (
        terminal_flow["maximum_pairwise_vector_magnitude_rms_pixels"] < 1.0
    )

    auto_gates = {
        "artifact_hash_binding": _gate(
            "artifact_hash_binding", "PASS" if binding_ok else "FAIL"
        ),
        "gpu_and_runtime_preflight": _gate(
            "gpu_and_runtime_preflight",
            "PASS" if gpu_ok else "FAIL",
            selected=selected,
            runtime_probe=runtime_probe,
        ),
        "pinned_model_revision": _gate(
            "pinned_model_revision",
            "PASS" if revision_ok else "FAIL",
            revision=checkpoint.get("revision"),
            transformer_shards=len(checkpoint.get("transformer_shards", [])),
            text_encoder_shards=len(checkpoint.get("text_encoder_shards", [])),
        ),
        "media_contract_1024x768_24fps_192frames": _gate(
            "media_contract_1024x768_24fps_192frames",
            "PASS" if media_ok else "FAIL",
            probe=probe,
        ),
        "exact_first_frame_ssim_at_least_0_985": _gate(
            "exact_first_frame_ssim_at_least_0_985",
            "PASS" if first_ssim >= 0.985 else "FAIL",
            ssim=first_ssim,
            threshold=0.985,
        ),
        "single_uninterrupted_shot_no_cut_crossfade_or_teleport": _gate(
            "single_uninterrupted_shot_no_cut_crossfade_or_teleport",
            "FAIL" if cut_times else "UNAVAILABLE",
            scene_threshold=0.30,
            detected_cut_times_seconds=cut_times,
            reason=(
                "Hard-cut pre-screen passed, but crossfade, teleport, and local material "
                "discontinuity require candidate-SHA-bound native review."
                if not cut_times
                else "automatic hard-cut detector found a discontinuity"
            ),
        ),
        "fixed_camera_background_and_lighting": _gate(
            "fixed_camera_background_and_lighting",
            "UNAVAILABLE" if minimum_background_ssim >= 0.90 else "FAIL",
            patch_ssim=background_results,
            minimum_ssim=minimum_background_ssim,
            threshold=0.90,
            reason=(
                "Background-patch pre-screen passed, but full-frame camera and lighting "
                "stability require candidate-SHA-bound native review."
                if minimum_background_ssim >= 0.90
                else "one or more frozen-background patches changed beyond threshold"
            ),
        ),
        "terminal_motion_below_1_pixel_rms_for_last_12_frames": _gate(
            "terminal_motion_below_1_pixel_rms_for_last_12_frames",
            "PASS" if terminal_flow_ok else "FAIL",
            threshold_pixels=1.0,
            comparison="maximum pairwise vector-magnitude RMS must be below threshold",
            optical_flow=terminal_flow,
        ),
    }
    if set(auto_gates) != AUTO_GATE_IDS:
        raise AssertionError("automatic hard-gate implementation drift")

    visual_gate_ids = list(config["visual_hard_gates"])
    if len(visual_gate_ids) != len(set(visual_gate_ids)):
        raise ValueError("visual gate identifiers must be unique")
    gates = []
    manual_gate_ids = []
    for gate_id in visual_gate_ids:
        if gate_id in auto_gates:
            gates.append(auto_gates[gate_id])
        else:
            gates.append(
                _gate(
                    gate_id,
                    "UNAVAILABLE",
                    reason="candidate-SHA-bound native-resolution full-video review required",
                )
            )
            manual_gate_ids.append(gate_id)
    failed = [item["gate_id"] for item in gates if item["state"] == "FAIL"]
    unavailable = [item["gate_id"] for item in gates if item["state"] == "UNAVAILABLE"]
    automatic_pass = not any(
        item["state"] == "FAIL" for item in gates if item["gate_id"] in AUTO_GATE_IDS
    )
    review_packet = _review_packet(
        output, video, candidate_sha256, manual_gate_ids
    )
    physical_gates = [
        _gate(
            gate_id,
            "UNAVAILABLE",
            reason="generated RGB cannot establish this physical-execution gate",
        )
        for gate_id in config["physical_promotion_gates"]
    ]
    result = {
        "schema_version": "1.0.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {"path": str(video), "sha256": candidate_sha256},
        "decision": (
            "REJECTED_AUTOMATIC_HARD_GATE_FAILURE"
            if failed
            else "PENDING_CANDIDATE_SHA_BOUND_NATIVE_REVIEW"
        ),
        "honest_status": "BLOCKED" if failed else "PARTIAL",
        "automatic_hard_gates_passed": automatic_pass,
        "visual_acceptance": False,
        "physical_promotion": False,
        "failed_visual_gate_ids": failed,
        "unavailable_visual_gate_ids": unavailable,
        "visual_gates": gates,
        "physical_promotion_gates": physical_gates,
        "metrics": {
            "first_frame_ssim": first_ssim,
            "detected_cut_times_seconds": cut_times,
            "background_patch_ssim": background_results,
            "minimum_background_patch_ssim": minimum_background_ssim,
            "terminal_optical_flow": terminal_flow,
        },
        "review_packet": review_packet,
        "diagnostic_logs": {
            "first_frame_ssim": first_log,
            "scene_cut": cut_log,
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output / "evaluation.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
