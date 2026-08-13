#!/usr/bin/env python3
"""Build a full flower-video delivery with shadow, EPL, and temporal gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.flower_repair_policy import FlowerRepairPolicy  # noqa: E402
from scripts.repair_video_transition_spikes import motion_bridge  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, object]]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[Any] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video has no decoded frames: {path}")
    height, width = frames[0].shape[:2]
    return frames, {
        "frames": len(frames),
        "width": width,
        "height": height,
        "fps": fps,
    }


def _load_packed(np: Any, path: Path, key: str = "packed") -> Any:
    payload = np.load(path, allow_pickle=False)
    height = int(payload["height"])
    width = int(payload["width"])
    packed = payload[key]
    unpacked = np.unpackbits(
        packed,
        axis=-1,
        bitorder=str(payload["bitorder"]),
    )
    return unpacked[..., : height * width].reshape(
        *packed.shape[:-1], height, width
    ).astype(bool)


def masked_transition_energy(np: Any, frames: list[Any], mask: Any) -> Any:
    values = []
    for index in range(1, len(frames)):
        delta = np.abs(
            frames[index].astype(np.float32) - frames[index - 1].astype(np.float32)
        ).mean(axis=2)
        values.append(float(delta[mask].mean()))
    return np.asarray(values, dtype=np.float64)


def detect_unsupported_transitions(
    np: Any,
    candidate_energy: Any,
    source_energy: Any,
    *,
    minimum_local_ratio: float,
    minimum_source_excess: float,
    local_radius: int = 12,
    minimum_separation: int = 4,
) -> list[int]:
    if candidate_energy.shape != source_energy.shape:
        raise ValueError("candidate and source transition arrays must match")
    proposals: list[tuple[float, int]] = []
    for transition in range(1, len(candidate_energy) + 1):
        row = transition - 1
        start = max(0, row - local_radius)
        end = min(len(candidate_energy), row + local_radius + 1)
        local = float(np.median(candidate_energy[start:end]))
        ratio = float(candidate_energy[row] / max(local, 1e-6))
        excess = float(candidate_energy[row] - source_energy[row])
        if ratio >= minimum_local_ratio and excess >= minimum_source_excess:
            proposals.append((ratio * excess, transition))
    selected: list[int] = []
    for _score, transition in sorted(proposals, reverse=True):
        if all(abs(transition - prior) >= minimum_separation for prior in selected):
            selected.append(transition)
    return sorted(selected)


def repair_transition_neighborhoods(
    cv2: Any,
    np: Any,
    frames: list[Any],
    transitions: list[int],
    robot_masks: Any,
    flower_masks: Any,
    static_safety: Any,
    phase04a_protection: dict[int, Any],
    *,
    radius: int,
    dilation_pixels: int,
    feather_sigma: float,
    mode: str,
) -> tuple[list[Any], list[dict[str, object]], Any]:
    repaired = [frame.copy() for frame in frames]
    touched = np.zeros((len(frames), *static_safety.shape), dtype=bool)
    kernel_size = dilation_pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    for transition in transitions:
        left = transition - radius
        right = transition + radius
        if left < 0 or right >= len(frames):
            raise ValueError(f"transition {transition} cannot use radius {radius}")
        bridges = motion_bridge(
            cv2,
            np,
            frames[left],
            frames[right],
            right - left - 1,
            mode=mode,
        )
        dynamic = np.any(robot_masks[left : right + 1], axis=0)
        dynamic = cv2.dilate(dynamic.astype(np.uint8), kernel) > 0
        allowed = dynamic & static_safety
        alpha = cv2.GaussianBlur(
            allowed.astype(np.float32),
            (0, 0),
            feather_sigma,
        )
        alpha[~static_safety] = 0.0
        alpha = np.clip(alpha, 0.0, 1.0)
        for offset, frame_index in enumerate(range(left + 1, right)):
            protected = flower_masks[frame_index].copy()
            if frame_index in phase04a_protection:
                protected |= phase04a_protection[frame_index]
            frame_alpha = alpha.copy()
            frame_alpha[protected] = 0.0
            composite = (
                frames[frame_index].astype(np.float32) * (1.0 - frame_alpha[..., None])
                + bridges[offset].astype(np.float32) * frame_alpha[..., None]
            )
            repaired[frame_index] = np.clip(np.rint(composite), 0, 255).astype(np.uint8)
            repaired[frame_index][protected] = frames[frame_index][protected]
            touched[frame_index] |= frame_alpha > 0
    records = [
        {
            "transition_frame": transition,
            "left_endpoint_frame": transition - radius,
            "right_endpoint_frame": transition + radius,
            "replaced_frame_range": [transition - radius + 1, transition + radius, 1],
        }
        for transition in transitions
    ]
    return repaired, records, touched


def apply_phase04a_relighting_residual(
    np: Any,
    frames: list[Any],
    geometry: list[Any],
    relighted: list[Any],
    safe_masks: list[Any],
    phase04a_protection: dict[int, Any],
    *,
    start_frame: int,
    taper_frames: int,
    maximum_delta: float,
) -> tuple[list[Any], dict[str, object], Any]:
    if not (len(geometry) == len(relighted) == len(safe_masks)):
        raise ValueError("phase-04A relighting inputs must have equal lengths")
    result = [frame.copy() for frame in frames]
    touched = np.zeros((len(frames), *frames[0].shape[:2]), dtype=bool)
    delta_rows = []
    for local, (base_geometry, routed, safe_frame) in enumerate(
        zip(geometry, relighted, safe_masks)
    ):
        target = start_frame + local
        safe = safe_frame.mean(axis=2) >= 127
        safe &= ~phase04a_protection.get(target, np.zeros_like(safe))
        edge = min(local + 1, len(geometry) - local, taper_frames)
        weight = min(1.0, edge / max(taper_frames, 1))
        delta = routed.astype(np.float32) - base_geometry.astype(np.float32)
        delta = np.clip(delta, -maximum_delta, maximum_delta) * weight
        updated = result[target].astype(np.float32)
        updated[safe] += delta[safe]
        result[target] = np.clip(np.rint(updated), 0, 255).astype(np.uint8)
        result[target][phase04a_protection.get(target, np.zeros_like(safe))] = frames[
            target
        ][phase04a_protection.get(target, np.zeros_like(safe))]
        touched[target] |= safe
        delta_rows.append(float(np.mean(np.abs(delta[safe]))) if safe.any() else 0.0)
    return result, {
        "source_frame_range": [start_frame, start_frame + len(geometry), 1],
        "taper_frames": taper_frames,
        "maximum_channel_delta": maximum_delta,
        "mean_applied_abs_delta": float(np.mean(delta_rows)),
        "maximum_frame_mean_abs_delta": float(np.max(delta_rows)),
    }, touched


def _encode(
    ffmpeg: Path,
    frames: list[Any],
    output: Path,
    fps: float,
    *,
    lossless: bool,
) -> None:
    height, width = frames[0].shape[:2]
    codec = ["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"] if lossless else [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
    ]
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            *codec,
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    if process.wait():
        raise RuntimeError(f"FFmpeg failed to encode {output}")


def _annotate(cv2: Any, frame: Any, text: str) -> Any:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 25), (20, 20, 20), -1)
    cv2.putText(
        result,
        text,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def _review_storyboard(
    cv2: Any,
    source: list[Any],
    before: list[Any],
    after: list[Any],
    transitions: list[int],
    output: Path,
) -> None:
    rows = []
    for transition in transitions:
        indices = range(transition - 2, transition + 3)
        for label, frames in (("source", source), ("before", before), ("after", after)):
            tiles = []
            for index in indices:
                tile = cv2.resize(frames[index], (249, 144), interpolation=cv2.INTER_AREA)
                tiles.append(_annotate(cv2, tile, f"{label} frame={index}"))
            rows.append(cv2.hconcat(tiles))
    cv2.imwrite(str(output), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])


def _dense_storyboard(
    cv2: Any,
    np: Any,
    source: list[Any],
    before: list[Any],
    after: list[Any],
    output: Path,
) -> None:
    indices = list(range(0, len(after), 24))
    rows = []
    for start in range(0, len(indices), 7):
        batch = indices[start : start + 7]
        for label, frames in (("source", source), ("before", before), ("after", after)):
            tiles = []
            for index in batch:
                tile = cv2.resize(frames[index], (192, 111), interpolation=cv2.INTER_AREA)
                tiles.append(_annotate(cv2, tile, f"{label} {index}"))
            while len(tiles) < 7:
                tiles.append(np.zeros_like(tiles[0]))
            rows.append(cv2.hconcat(tiles))
    cv2.imwrite(str(output), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--shadow-manifest", type=Path, required=True)
    parser.add_argument("--epl-manifest", type=Path, required=True)
    parser.add_argument("--epl-contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--rejected-trained-lora-assessment", type=Path, required=True)
    parser.add_argument("--safety-mask", type=Path, required=True)
    parser.add_argument("--robot-body-masks", type=Path, required=True)
    parser.add_argument("--robot-wrist-masks", type=Path, required=True)
    parser.add_argument("--robot-limb-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--phase04a-stem-masks", type=Path, required=True)
    parser.add_argument("--phase04a-hand-masks", type=Path, required=True)
    parser.add_argument("--phase04a-geometry", type=Path, required=True)
    parser.add_argument("--phase04a-relighted", type=Path, required=True)
    parser.add_argument("--phase04a-relight-mask", type=Path, required=True)
    parser.add_argument("--phase04a-start-frame", type=int, default=272)
    parser.add_argument("--minimum-local-ratio", type=float, default=2.0)
    parser.add_argument("--minimum-source-excess", type=float, default=4.5)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--mask-dilation", type=int, default=2)
    parser.add_argument("--mask-feather-sigma", type=float, default=2.0)
    parser.add_argument("--relight-taper-frames", type=int, default=8)
    parser.add_argument("--relight-maximum-delta", type=float, default=8.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "review").mkdir()
    (output_dir / "provenance" / "execution-sources").mkdir(parents=True)

    paths = {
        key: value.expanduser().resolve()
        for key, value in vars(args).items()
        if isinstance(value, Path)
    }
    for label, path in paths.items():
        if label == "output_dir":
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")

    import cv2
    import numpy as np

    base, base_info = _decode(cv2, paths["base_video"])
    source, source_info = _decode(cv2, paths["source_video"])
    geometry, geometry_info = _decode(cv2, paths["phase04a_geometry"])
    relighted, relighted_info = _decode(cv2, paths["phase04a_relighted"])
    relight_masks, relight_mask_info = _decode(cv2, paths["phase04a_relight_mask"])
    if len(base) != 660 or len(source) != 660:
        raise RuntimeError("expected base and source to decode 660 frames")
    if base_info["width"] != source_info["width"] or base_info["height"] != source_info["height"]:
        raise RuntimeError("base and source dimensions differ")
    if not (len(geometry) == len(relighted) == len(relight_masks) == 106):
        raise RuntimeError("expected 106 phase-04A relighting frames")

    height = int(base_info["height"])
    width = int(base_info["width"])
    static_safety = cv2.imread(str(paths["safety_mask"]), cv2.IMREAD_GRAYSCALE)
    if static_safety is None:
        raise RuntimeError("failed to load safety mask")
    static_safety = cv2.resize(
        static_safety,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    static_safety = cv2.dilate(
        (static_safety > 0).astype(np.uint8),
        np.ones((31, 31), dtype=np.uint8),
    ) > 0

    body = _load_packed(np, paths["robot_body_masks"])
    wrists = _load_packed(np, paths["robot_wrist_masks"])
    limbs = _load_packed(np, paths["robot_limb_masks"])
    flowers = _load_packed(np, paths["flower_masks"])
    robot = body | wrists | limbs
    if robot.shape != flowers.shape or robot.shape != (660, height, width):
        raise RuntimeError("full-film masks do not match the video")

    stem_payload = np.load(paths["phase04a_stem_masks"], allow_pickle=False)
    hand_payload = np.load(paths["phase04a_hand_masks"], allow_pickle=False)
    stems = _load_packed(np, paths["phase04a_stem_masks"], "masks_packed")[0]
    hands = _load_packed(np, paths["phase04a_hand_masks"], "masks_packed")
    stem_indices = [int(value) for value in stem_payload["source_frame_indices"]]
    hand_indices = [int(value) for value in hand_payload["source_frame_indices"]]
    if stem_indices != hand_indices:
        raise RuntimeError("phase-04 stem and hand tracks use different source frames")
    phase04a_protection = {
        frame: stems[local] | hands[0, local] | hands[1, local] | flowers[frame]
        for local, frame in enumerate(stem_indices)
        if args.phase04a_start_frame <= frame < args.phase04a_start_frame + 106
    }

    candidate_energy = masked_transition_energy(np, base, static_safety)
    source_energy = masked_transition_energy(np, source, static_safety)
    transitions = detect_unsupported_transitions(
        np,
        candidate_energy,
        source_energy,
        minimum_local_ratio=args.minimum_local_ratio,
        minimum_source_excess=args.minimum_source_excess,
    )

    scorecard = json.loads(paths["epl_manifest"].read_text())["selected_scorecard"]
    policy = FlowerRepairPolicy.load(paths["policy"])
    recipes = [
        {
            "name": "no-temporal-repair",
            "hard_background_lock": 0,
            "restore_source_flowers": 0,
            "exclude_source_face_from_flower_restore": 0,
            "mask_dilation_pixels": 0,
            "flower_dilation_pixels": 0,
            "face_box_margin_pixels": 0,
        },
        {
            "name": "global-noise-smoothing",
            "hard_background_lock": 0,
            "restore_source_flowers": 0,
            "exclude_source_face_from_flower_restore": 0,
            "mask_dilation_pixels": 3,
            "flower_dilation_pixels": 0,
            "face_box_margin_pixels": 0,
        },
        {
            "name": "epl-local-crossfade",
            "hard_background_lock": 1,
            "restore_source_flowers": 1,
            "exclude_source_face_from_flower_restore": 1,
            "mask_dilation_pixels": args.mask_dilation,
            "flower_dilation_pixels": 2,
            "face_box_margin_pixels": 12,
        },
        {
            "name": "epl-local-flow",
            "hard_background_lock": 1,
            "restore_source_flowers": 1,
            "exclude_source_face_from_flower_restore": 1,
            "mask_dilation_pixels": 3,
            "flower_dilation_pixels": 2,
            "face_box_margin_pixels": 12,
        },
    ]
    ranking = [
        {"recipe": dict(recipe), "predicted_constrained_utility": score}
        for recipe, score in policy.rank(scorecard, recipes)
    ]
    selected_recipe = str(ranking[0]["recipe"]["name"])
    if selected_recipe != "epl-local-crossfade":
        raise RuntimeError(f"trained policy selected unsupported route: {selected_recipe}")

    temporal, repairs, temporal_touched = repair_transition_neighborhoods(
        cv2,
        np,
        base,
        transitions,
        robot,
        flowers,
        static_safety,
        phase04a_protection,
        radius=args.temporal_radius,
        dilation_pixels=args.mask_dilation,
        feather_sigma=args.mask_feather_sigma,
        mode="crossfade",
    )
    final, relight_record, relight_touched = apply_phase04a_relighting_residual(
        np,
        temporal,
        geometry,
        relighted,
        relight_masks,
        phase04a_protection,
        start_frame=args.phase04a_start_frame,
        taper_frames=args.relight_taper_frames,
        maximum_delta=args.relight_maximum_delta,
    )
    touched = temporal_touched | relight_touched

    after_energy = masked_transition_energy(np, final, static_safety)
    selected_before = {str(row): float(candidate_energy[row - 1]) for row in transitions}
    selected_after = {str(row): float(after_energy[row - 1]) for row in transitions}
    selected_improved = {
        str(row): selected_after[str(row)] < selected_before[str(row)]
        for row in transitions
    }
    outside_exact = all(
        np.array_equal(base[index][~static_safety], final[index][~static_safety])
        for index in range(660)
    )
    flower_exact = min(
        float(np.mean(np.all(base[index][flowers[index]] == final[index][flowers[index]], axis=1)))
        if flowers[index].any()
        else 1.0
        for index in range(660)
    )
    phase04a_exact = min(
        float(np.mean(np.all(base[index][mask] == final[index][mask], axis=1)))
        if mask.any()
        else 1.0
        for index, mask in phase04a_protection.items()
    )

    lossless_output = output_dir / "flower-full-shadow-epl-temporal-lossless.mp4"
    compatibility_output = output_dir / "flower-full-shadow-epl-temporal.mp4"
    _encode(
        paths["ffmpeg"],
        final,
        lossless_output,
        float(base_info["fps"]),
        lossless=True,
    )
    _encode(
        paths["ffmpeg"],
        final,
        compatibility_output,
        float(base_info["fps"]),
        lossless=False,
    )
    decoded_output, output_info = _decode(cv2, compatibility_output)
    if len(decoded_output) != 660:
        raise RuntimeError("encoded full delivery changed the frame count")

    _review_storyboard(
        cv2,
        source,
        base,
        final,
        transitions,
        output_dir / "review" / "all-repaired-transitions-source-before-after.jpg",
    )
    _dense_storyboard(
        cv2,
        np,
        source,
        base,
        final,
        output_dir / "review" / "full-timeline-1fps-source-before-after.jpg",
    )

    shadow_manifest = json.loads(paths["shadow_manifest"].read_text())
    lora_assessment = json.loads(paths["rejected_trained_lora_assessment"].read_text())
    automatic_gates = {
        "all_660_frames_decoded": len(decoded_output) == 660,
        "shadow_background_parent_accepted": all(
            shadow_manifest["acceptance_gates"].values()
        ),
        "trained_policy_selected_epl_local_crossfade": selected_recipe
        == "epl-local-crossfade",
        "rejected_topology_lora_pixels_excluded": not bool(
            lora_assessment["assessment"]["passed"]
        ),
        "outside_safety_exact_preencode": outside_exact,
        "flowers_exact_preencode": flower_exact == 1.0,
        "phase04a_contact_protected_exact_preencode": phase04a_exact == 1.0,
        "all_selected_transitions_improved": all(selected_improved.values()),
        "edit_scope_bounded": float(touched.mean()) <= 0.08,
    }
    script_copy = output_dir / "provenance" / "execution-sources" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), script_copy)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "accepted_shadow_epl_parent_plus_learned_policy_local_temporal_routing",
        "status": "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": {
            "used": False,
            "reason": "CPU-only routing over previously generated GPU artifacts",
        },
        "packages": {
            package: (
                importlib.metadata.version(package)
                if package in {dist.metadata["Name"] for dist in importlib.metadata.distributions()}
                else None
            )
            for package in ("numpy", "opencv-python", "opencv-python-headless")
        },
        "coordinate_frame": "camera:H3_output_pixels",
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
            if label not in {"output_dir", "ffmpeg"}
        },
        "execution_source": {"path": str(script_copy), "sha256": _sha256(script_copy)},
        "base_info": base_info,
        "source_info": source_info,
        "phase04a_input_info": {
            "geometry": geometry_info,
            "relighted": relighted_info,
            "safe_mask": relight_mask_info,
        },
        "trained_policy": {
            "ranking": ranking,
            "selected_recipe": selected_recipe,
            "role": "select the non-regressing background/object-protected repair contract; it is not a pixel denoiser",
        },
        "rejected_trained_lora": {
            "assessment_passed": lora_assessment["assessment"]["passed"],
            "failed_gates": lora_assessment["assessment"]["failed_gates"],
            "role": "excluded from pixel generation after failed topology and motion gates",
        },
        "unsupported_transition_detection": {
            "minimum_local_ratio": args.minimum_local_ratio,
            "minimum_source_excess": args.minimum_source_excess,
            "selected_transition_frames": transitions,
        },
        "temporal_repairs": repairs,
        "phase04a_relighting": relight_record,
        "metrics": {
            "selected_transition_energy_before": selected_before,
            "selected_transition_energy_after": selected_after,
            "selected_transition_improved": selected_improved,
            "maximum_roi_transition_before": float(candidate_energy.max()),
            "maximum_roi_transition_after": float(after_energy.max()),
            "outside_safety_exact_preencode": outside_exact,
            "flower_exact_fraction_min_preencode": flower_exact,
            "phase04a_protected_exact_fraction_min_preencode": phase04a_exact,
            "modified_fraction_mean": float(touched.mean()),
            "modified_fraction_max": float(touched.reshape(660, -1).mean(axis=1).max()),
        },
        "automatic_gates": automatic_gates,
        "all_automatic_gates_pass": all(automatic_gates.values()),
        "outputs": {
            "lossless": {
                "path": str(lossless_output),
                "sha256": _sha256(lossless_output),
            },
            "compatibility": {
                "path": str(compatibility_output),
                "sha256": _sha256(compatibility_output),
                "info": output_info,
            },
            "transition_review": "review/all-repaired-transitions-source-before-after.jpg",
            "dense_review": "review/full-timeline-1fps-source-before-after.jpg",
        },
        "limitations": [
            "Full-film shadow/background and temporal delivery is a reviewed 2-D video claim, not physical robot execution.",
            "Only source-unsupported isolated transitions are bridged; real action transitions are retained.",
            "The trained repair policy selects the protection/routing recipe but does not synthesize pixels.",
            "The newly trained topology LoRA is excluded because its held-out structure and motion gates failed.",
            "Persistent single-flower/contact identity remains independently accepted only on source frames [272,378).",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "transitions": transitions,
                "automatic_gates": automatic_gates,
                "metrics": manifest["metrics"],
            },
            indent=2,
        )
    )
    return 0 if manifest["all_automatic_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
