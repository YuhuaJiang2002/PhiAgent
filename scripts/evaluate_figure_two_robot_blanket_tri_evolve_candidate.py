#!/usr/bin/env python3
"""Fail-closed evaluation for one E2 blanket Tri-Evolve candidate."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_figure_two_robot_blanket_h3_candidate import (  # noqa: E402
    _gate,
    _probe,
    _run,
    _scene_cuts,
    _sha256,
    _ssim_to_still,
    _terminal_flow_rms,
    _write_json,
)
from phiagent.harness.blanket_tri_evolve import canonical_sha256  # noqa: E402


HARNESS_VERSION = "figure-two-robot-blanket-tri-evolve-e2-v1"
EXPECTED_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
AUTOMATIC_GATE_IDS = {
    "artifact_hash_binding",
    "gpu_and_runtime_preflight",
    "pinned_model_revision",
    "media_contract_1024x768_24fps_192frames",
    "predeclared_exact_boundary_binding",
    "exact_first_frame_ssim_at_least_0_985",
    "single_uninterrupted_shot_no_cut_crossfade_or_teleport",
    "fixed_camera_background_and_lighting",
    "terminal_motion_below_1_pixel_rms_for_last_12_frames",
}
CONCLUSIVE_AUTOMATIC_GATE_IDS = AUTOMATIC_GATE_IDS - {
    "single_uninterrupted_shot_no_cut_crossfade_or_teleport"
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _candidate_video(candidate_dir: Path) -> Path:
    videos = sorted(candidate_dir.glob("candidates/*/seed-*/candidate.mp4"))
    if len(videos) != 1:
        raise ValueError(f"expected exactly one bound candidate, found {len(videos)}")
    return videos[0]


def _case_for_record(spec: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item for item in spec["cases"] if item.get("label") == record.get("case_id")
    ]
    if len(matches) != 1:
        raise ValueError("candidate record does not bind exactly one generation case")
    return matches[0]


def _review_packet_e2(
    output: Path,
    video: Path,
    candidate_sha256: str,
    manual_gate_ids: list[str],
    frame_indices: list[int],
) -> dict[str, Any]:
    contact_sheet = output / "native-review-contact-sheet.jpg"
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
            "select='not(mod(n,12))',scale=512:384:flags=lanczos,"
            "tile=4x4:padding=6:margin=6",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(contact_sheet),
        )
    )
    frames_dir = output / "native-review-frames"
    frames_dir.mkdir()
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
        "attestation": {
            "watched_full_video_from_start_to_finish": False,
            "watched_at_native_1024x768_resolution": False,
            "reviewed_original_mp4_not_contact_sheet_only": False,
        },
        "instructions": (
            "Watch the original candidate at native resolution. Set every gate to PASS "
            "or FAIL with time ranges and notes. Missing or hidden evidence rejects."
        ),
        "gates": [
            {"gate_id": gate_id, "state": "UNAVAILABLE", "time_ranges": [], "notes": ""}
            for gate_id in manual_gate_ids
        ],
        "overall_visual_acceptance": False,
        "claim_boundary": (
            "Native review can accept visual-generation quality only; it cannot "
            "establish metric, force, safety, or real-hardware evidence."
        ),
    }
    _write_json(output / "native-review-template.json", template)
    return {
        "contact_sheet": str(contact_sheet),
        "contact_sheet_sha256": _sha256(contact_sheet),
        "frame_indices": frame_indices,
        "template": str(output / "native-review-template.json"),
    }


def main() -> int:
    args = _parser().parse_args()
    campaign = args.campaign_dir.expanduser().resolve()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse evaluation directory: {output}")
    output.mkdir(parents=True)

    config = _read_json(campaign / "inputs/campaign-config.json")
    spec = _read_json(campaign / "inputs/generation-spec.json")
    manifest = _read_json(campaign / "manifest.json")
    evaluation_contract_path = campaign / "inputs/evaluation-contract.json"
    evaluation_contract = _read_json(evaluation_contract_path)
    evaluation_contract_sha256 = (
        campaign / "inputs/evaluation-contract.sha256"
    ).read_text(encoding="utf-8").strip()
    metadata = _read_json(candidate_dir / "metadata.json")
    gpu_selection = _read_json(candidate_dir / "gpu_selection.json")
    runtime_probe = _read_json(candidate_dir / "runtime_probe.json")
    checkpoint = _read_json(candidate_dir / "checkpoint_evidence.json")
    video = _candidate_video(candidate_dir)
    candidate_record = _read_json(video.parent / "tri-evolve-record.json")
    case = _case_for_record(spec, candidate_record)
    initial = campaign / "inputs/initial-frame.png"
    raw_video = video.with_name("candidate.raw.mp4")
    candidate_sha256 = _sha256(video)
    raw_sha256 = _sha256(raw_video) if raw_video.is_file() else None
    binding = candidate_record.get("boundary_binding", {})
    completed = metadata.get("completed_candidates", [])

    binding_ok = (
        config.get("campaign_id") == HARNESS_VERSION
        and metadata.get("harness_version") == HARNESS_VERSION
        and metadata.get("boundary_binding_enabled") is True
        and _sha256(initial) == config["initial_frame"]["sha256"]
        and _sha256(campaign / "inputs/generation-spec.json")
        == manifest["hashes"]["generation_spec_file"]
        and metadata.get("spec_sha256") == manifest["hashes"]["generation_spec_file"]
        and metadata.get("challenge_frame_sha256") == _sha256(initial)
        and metadata.get("challenge_sha256") == spec["challenge"]["challenge_sha256"]
        and candidate_record.get("challenge_sha256")
        == spec["challenge"]["challenge_sha256"]
        and candidate_record.get("task_plan_sha256") == case["task_plan_sha256"]
        and candidate_record.get("harness_sha256") == case["harness_sha256"]
        and candidate_record.get("prompt_sha256") == case["prompt_sha256"]
        and len(completed) == 1
        and completed[0].get("video_sha256") == candidate_sha256
        and candidate_record.get("video_sha256") == candidate_sha256
        and evaluation_contract.get("candidate_independent") is True
        and evaluation_contract.get("initial_frame_sha256") == _sha256(initial)
        and evaluation_contract.get("task_plan_sha256")
        == manifest["hashes"]["task_plan"]
        and evaluation_contract_sha256 == canonical_sha256(evaluation_contract)
    )
    boundary_ok = (
        spec["generation"].get("boundary_binding", {}).get("enabled") is True
        and binding.get("method")
        == "prepend_exact_source_frame_then_first_191_generated_frames"
        and binding.get("source_frame_count") == 1
        and binding.get("generated_frame_count") == 191
        and binding.get("thresholds_unchanged") is True
        and binding.get("source_frame_sha256") == _sha256(initial)
        and binding.get("bound_video_sha256") == candidate_sha256
        and binding.get("raw_video_sha256") == raw_sha256
    )

    selected = gpu_selection.get("selected", [])
    expected_gpu_count = int(spec["compute"]["num_gpus"])
    gpu_ok = (
        len(selected) == expected_gpu_count
        and all(item.get("name") == "NVIDIA H200" for item in selected)
        and all(str(item.get("uuid", "")).startswith("GPU-") for item in selected)
        and all(
            int(item.get("free_mib", -1)) >= int(spec["compute"]["minimum_free_mib"])
            for item in selected
        )
        and all(not item.get("compute_processes") for item in selected)
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

    thresholds = evaluation_contract["automatic_thresholds"]
    first_ssim, first_log = _ssim_to_still(
        video, initial, start_frame=0, frame_count=1
    )
    cut_times, cut_log = _scene_cuts(video)
    background_patches = {
        patch_id: (int(xywh[2]), int(xywh[3]), int(xywh[0]), int(xywh[1]))
        for patch_id, xywh in evaluation_contract["background_patches_xywh"].items()
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
    terminal_window = evaluation_contract["terminal_flow_window"]
    terminal_flow = _terminal_flow_rms(
        video,
        start_frame=int(terminal_window["start_frame"]),
        frame_count=int(terminal_window["frame_count"]),
    )
    terminal_flow_max = terminal_flow[
        "maximum_pairwise_vector_magnitude_rms_pixels"
    ]

    automatic = {
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
        ),
        "media_contract_1024x768_24fps_192frames": _gate(
            "media_contract_1024x768_24fps_192frames",
            "PASS" if media_ok else "FAIL",
            probe=probe,
        ),
        "predeclared_exact_boundary_binding": _gate(
            "predeclared_exact_boundary_binding",
            "PASS" if boundary_ok else "FAIL",
            binding=binding,
            raw_candidate_preserved=raw_video.is_file(),
        ),
        "exact_first_frame_ssim_at_least_0_985": _gate(
            "exact_first_frame_ssim_at_least_0_985",
            "PASS"
            if first_ssim >= float(thresholds["exact_first_frame_ssim_min"])
            else "FAIL",
            ssim=first_ssim,
            threshold=thresholds["exact_first_frame_ssim_min"],
        ),
        "single_uninterrupted_shot_no_cut_crossfade_or_teleport": _gate(
            "single_uninterrupted_shot_no_cut_crossfade_or_teleport",
            "FAIL" if cut_times else "UNAVAILABLE",
            detected_hard_cut_times_seconds=cut_times,
            reason=(
                "Hard-cut pre-screen passed; crossfade, teleport, and local material "
                "continuity still require native-resolution review."
                if not cut_times
                else "One or more hard cuts were detected."
            ),
        ),
        "fixed_camera_background_and_lighting": _gate(
            "fixed_camera_background_and_lighting",
            "PASS"
            if minimum_background_ssim
            >= float(thresholds["background_patch_ssim_min"])
            else "FAIL",
            patch_ssim=background_results,
            minimum_ssim=minimum_background_ssim,
            threshold=thresholds["background_patch_ssim_min"],
        ),
        "terminal_motion_below_1_pixel_rms_for_last_12_frames": _gate(
            "terminal_motion_below_1_pixel_rms_for_last_12_frames",
            "PASS"
            if terminal_flow_max
            < float(thresholds["terminal_flow_rms_pixels_max"])
            else "FAIL",
            terminal_flow=terminal_flow,
            threshold=thresholds["terminal_flow_rms_pixels_max"],
        ),
    }
    manual_gate_ids = [
        gate_id
        for gate_id in config["visual_hard_gates"]
        if gate_id not in AUTOMATIC_GATE_IDS
    ]
    review = _review_packet_e2(
        output,
        video,
        candidate_sha256,
        manual_gate_ids,
        [int(item) for item in evaluation_contract["native_review_frame_indices"]],
    )
    native_review = {
        gate_id: _gate(
            gate_id,
            "UNAVAILABLE",
            reason="candidate-SHA-bound native-resolution human review not completed",
        )
        for gate_id in manual_gate_ids
    }
    failed_automatic = sorted(
        gate_id
        for gate_id, result in automatic.items()
        if gate_id in CONCLUSIVE_AUTOMATIC_GATE_IDS and result["state"] != "PASS"
    )
    physical = {
        gate_id: _gate(
            gate_id,
            "UNAVAILABLE",
            reason="generated RGB cannot satisfy physical promotion evidence",
        )
        for gate_id in config["physical_promotion_gates"]
    }
    decision = "REJECT" if failed_automatic else "PENDING_NATIVE_REVIEW"
    report = {
        "schema_version": "1.0.0",
        "campaign_id": HARNESS_VERSION,
        "candidate": {
            "case_id": candidate_record["case_id"],
            "strategy": candidate_record["strategy"],
            "seed": candidate_record["seed"],
            "path": str(video),
            "sha256": candidate_sha256,
            "raw_path": str(raw_video),
            "raw_sha256": raw_sha256,
        },
        "evaluation_contract": {
            "path": str(evaluation_contract_path),
            "canonical_sha256": evaluation_contract_sha256,
            "file_sha256": _sha256(evaluation_contract_path),
        },
        "automatic_gate_results": automatic,
        "native_review_gate_results": native_review,
        "physical_promotion_gate_results": physical,
        "automatic_metrics": {
            "first_frame_ssim": first_ssim,
            "background_patch_ssim": background_results,
            "minimum_background_patch_ssim": minimum_background_ssim,
            "detected_hard_cut_times_seconds": cut_times,
            "terminal_flow": terminal_flow,
        },
        "failed_automatic_hard_gate_ids": failed_automatic,
        "automatic_decision": decision,
        "overall_visual_acceptance": False,
        "physical_promotion": {
            "promote": False,
            "reason": "proposal_not_physical_calibration",
        },
        "review_packet": review,
        "logs": {
            "first_frame_ssim_tail": first_log,
            "scene_cut_tail": cut_log,
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output / "evaluation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 2 if decision == "REJECT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
