#!/usr/bin/env python3
"""Run Stage-3-BIR + Adaptive VDA with frozen V1/V2 routing.

Generation never reads H2O 3D ground truth.  ``evaluate`` is a separate phase
that first verifies the generated manifest and all candidate hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.perception.adaptive_vda import (  # noqa: E402
    HANDS,
    VERSION,
    AdaptiveVDAConfig,
    apply_root_depth_correction,
    arrays_equal,
    camera_z_to_world,
    estimate_wrist_patch_residual,
    interaction_protection_weight,
    is_hard_sequence,
    map_relative_depth,
    project_camera_points,
    robust_temporal_correction,
)
from phiagent.rendering.wan_animate import (  # noqa: E402
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


METRICS = (
    "pa_mpjpe_mm",
    "w_mpjpe_mm",
    "wa_mpjpe_mm",
    "rte_percent",
    "accel_m_s2",
)
ALLOWED_V2_CHANGES = {
    "transl",
    "joints_3d_world",
    "vertices_world",
    "joints_3d_camera",
    "vertices_camera",
    "joints_2d",
    "joints_in_frame",
}
V2_FINITE_FIELDS = ALLOWED_V2_CHANGES | {
    "adaptive_vda_depth_correction_camera_z_m",
    "adaptive_vda_world_shift_m",
    "adaptive_vda_penetration_energy",
    "adaptive_vda_max_depth_proxy_m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "generate", "evaluate", "audit", "all"))
    parser.add_argument(
        "--stage3-root", type=Path, required=True, help="Stage-3-BIR runs directory"
    )
    parser.add_argument(
        "--vda-depth-root", type=Path, required=True, help="Sparse numeric VDA NPZ directory"
    )
    parser.add_argument(
        "--scale-summary", type=Path, required=True, help="Frozen VDA/metric calibration JSON"
    )
    parser.add_argument(
        "--v1-hard-root", type=Path, required=True, help="Frozen V1 dense-recovery run tree"
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="A new experiment directory"
    )
    parser.add_argument(
        "--h3mr-code-root",
        type=Path,
        required=True,
        help="External code root containing hand_harness/",
    )
    parser.add_argument(
        "--h2o-root",
        type=Path,
        help="H2O raw root; otherwise H2O_ROOT or copied input/h2o_gt",
    )
    parser.add_argument(
        "--evaluation-dir", type=Path, help="Directory containing the official evaluator"
    )
    parser.add_argument(
        "--hawor-root", type=Path, help="External HaWoR repository used by the evaluator"
    )
    parser.add_argument(
        "--gpu", type=int, help="Physical GPU index; default selects the freest GPU"
    )
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=4096)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("scale summary must contain a non-empty records list")
    sequences = [str(record["sequence"]) for record in records]
    if len(sequences) != len(set(sequences)):
        raise RuntimeError("scale summary contains duplicate sequences")
    return records


def find_hand_npz(root: Path, sequence: str, hand: str) -> Path:
    directory = root / sequence / "05_bimanual_refine" / "bir" / hand
    files = sorted(directory.glob("*.npz"))
    if len(files) != 1:
        raise RuntimeError(
            f"expected one {hand} NPZ for {sequence} in {directory}, found {len(files)}"
        )
    return files[0]


def candidate_path(output_root: Path, sequence: str, hand: str) -> Path:
    return (
        output_root
        / "runs"
        / sequence
        / "05_bimanual_refine"
        / "bir"
        / hand
        / f"mano_{hand}_stage3_bir_adaptive_vda_hawor_30fps.npz"
    )


def resolve_h2o_root(args: argparse.Namespace) -> Path:
    candidates = []
    if args.h2o_root is not None:
        candidates.append(args.h2o_root)
    if os.environ.get("H2O_ROOT"):
        candidates.append(Path(os.environ["H2O_ROOT"]))
    # A self-contained experiment commonly stores the copied GT beside stage3_bir/.
    candidates.append(args.stage3_root.resolve().parents[1] / "h2o_gt")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    raise RuntimeError("H2O root was not found; pass --h2o-root or set H2O_ROOT")


def git_state(repository: Path) -> dict[str, str]:
    result = {}
    for name, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command, cwd=repository, check=False, capture_output=True, text=True
        )
        result[name] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    return result


def preflight(args: argparse.Namespace, *, require_gt: bool) -> dict[str, Any]:
    stage3_root = args.stage3_root.expanduser().resolve()
    vda_depth_root = args.vda_depth_root.expanduser().resolve()
    scale_summary = args.scale_summary.expanduser().resolve()
    v1_hard_root = args.v1_hard_root.expanduser().resolve()
    h3mr_code_root = args.h3mr_code_root.expanduser().resolve()
    for path, label in (
        (stage3_root, "Stage-3-BIR runs"),
        (vda_depth_root, "VDA numeric depth"),
        (v1_hard_root, "V1 hard-recovery runs"),
        (h3mr_code_root, "H3MR code"),
    ):
        if not path.is_dir():
            raise RuntimeError(f"missing {label}: {path}")
    if not scale_summary.is_file():
        raise RuntimeError(f"missing scale summary: {scale_summary}")
    detector = h3mr_code_root / "hand_harness/stages/bimanual_penetration_detection.py"
    if not detector.is_file():
        raise RuntimeError(f"missing Stage-3 penetration detector: {detector}")
    output_root = args.output_root.expanduser().resolve()
    candidate_manifest = output_root / "candidate_manifest.json"
    if args.command in ("generate", "all") and candidate_manifest.exists():
        raise FileExistsError(f"candidate manifest already exists: {output_root}")
    if args.command in ("evaluate", "audit") and not candidate_manifest.is_file():
        raise FileNotFoundError(f"candidate manifest is missing: {candidate_manifest}")
    if args.command == "evaluate":
        evaluation_outputs = (
            output_root / "evaluation" / "per_frame_metrics.npz",
            output_root / "evaluation" / "summary.json",
        )
        existing = [str(path) for path in evaluation_outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing evaluation outputs: " + ", ".join(existing)
            )
    if args.command == "audit" and (output_root / "audit.json").exists():
        raise FileExistsError(
            f"refusing to overwrite existing audit: {output_root / 'audit.json'}"
        )

    config = AdaptiveVDAConfig()
    records = load_records(scale_summary)
    branches = {"v1_hard": [], "v2_ordinary": []}
    for record in records:
        sequence = str(record["sequence"])
        hard = is_hard_sequence(
            int(record["valid_keyframes"]), float(record["scale_relative_mad"]), config
        )
        branch = "v1_hard" if hard else "v2_ordinary"
        branches[branch].append(sequence)
        for hand in HANDS:
            find_hand_npz(v1_hard_root if hard else stage3_root, sequence, hand)
        if not hard:
            depth_path = vda_depth_root / f"{sequence}_depths.npz"
            if not depth_path.is_file():
                raise RuntimeError(f"missing numeric VDA depth: {depth_path}")

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    result: dict[str, Any] = {
        "status": "passed",
        "version": VERSION,
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "branches": branches,
        "selected_gpu": {
            "physical_index": selected.physical_index,
            "name": selected.name,
            "free_mib": selected.free_mib,
            "total_mib": selected.total_mib,
        },
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "inputs": {
            "stage3_root": str(stage3_root),
            "vda_depth_root": str(vda_depth_root),
            "scale_summary": str(scale_summary),
            "scale_summary_sha256": sha256(scale_summary),
            "v1_hard_root": str(v1_hard_root),
            "h3mr_code_root": str(h3mr_code_root),
            "h3mr_git": git_state(h3mr_code_root),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
    }
    if require_gt:
        h2o_root = resolve_h2o_root(args)
        evaluation_dir = (args.evaluation_dir or h3mr_code_root / "evaluation").resolve()
        hawor_root = (args.hawor_root or h3mr_code_root / "hawor").resolve()
        evaluator = evaluation_dir / "evaluate_hawor_h2o_mano_baseline.py"
        if not evaluator.is_file() or not hawor_root.is_dir():
            raise RuntimeError("official evaluator or HaWoR runtime is missing")
        result["evaluation"] = {
            "h2o_root": str(h2o_root),
            "evaluation_dir": str(evaluation_dir),
            "hawor_root": str(hawor_root),
            "evaluator_sha256": sha256(evaluator),
        }
    return result


def run_detector(
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    vertices: Mapping[str, np.ndarray],
    h3mr_code_root: Path,
) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(h3mr_code_root))
    from hand_harness.stages.bimanual_penetration_detection import (  # noqa: PLC0415
        PenetrationDetectionSettings,
        detect_penetration_arrays,
    )

    return detect_penetration_arrays(
        left_vertices=vertices["left"],
        right_vertices=vertices["right"],
        left_faces=payloads["left"]["faces"],
        right_faces=payloads["right"]["faces"],
        left_valid=payloads["left"].get("stage2_valid", payloads["left"]["valid"]),
        right_valid=payloads["right"].get("stage2_valid", payloads["right"]["valid"]),
        settings=PenetrationDetectionSettings(batch_size=4),
        device="cuda:0",
    )


def generate_v2_sequence(
    record: Mapping[str, Any],
    args: argparse.Namespace,
    config: AdaptiveVDAConfig,
) -> dict[str, Any]:
    sequence = str(record["sequence"])
    stage3_root = args.stage3_root.resolve()
    payloads = {
        hand: load_npz(find_hand_npz(stage3_root, sequence, hand)) for hand in HANDS
    }
    frames = len(payloads["left"]["frame_index"])
    if not np.array_equal(payloads["left"]["frame_index"], payloads["right"]["frame_index"]):
        raise RuntimeError(f"left/right timelines differ for {sequence}")
    depth_path = args.vda_depth_root.resolve() / f"{sequence}_depths.npz"
    with np.load(depth_path, allow_pickle=False) as archive:
        relative_depths = np.asarray(archive["depths"], dtype=np.float32)
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    if len(relative_depths) != len(frame_indices):
        raise RuntimeError(f"VDA depth/index length mismatch for {sequence}")

    corrections: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    mapping = record["mapping"]
    for hand in HANDS:
        if not bool(mapping.get("accepted", False)):
            corrections[hand] = np.zeros(frames, dtype=np.float64)
            diagnostics[hand] = {
                "accepted": False,
                "reason": "vda_metric_mapping_rejected",
                "mapping_reason": mapping.get("reason"),
            }
            continue
        estimates = []
        for offset, frame in enumerate(frame_indices):
            if frame < 0 or frame >= frames:
                raise RuntimeError(f"VDA frame {frame} is outside {sequence}")
            if not bool(payloads[hand]["valid"][frame]):
                continue
            metric_depth = map_relative_depth(relative_depths[offset], mapping)
            estimate = estimate_wrist_patch_residual(
                metric_depth,
                payloads[hand]["joints_3d_camera"][frame, 0],
                payloads[hand]["camera_intrinsics"],
                config,
            )
            if estimate is not None:
                value, detail = estimate
                estimates.append((int(frame), float(value), detail))
        corrections[hand], diagnostics[hand] = robust_temporal_correction(
            frames, estimates, config
        )

    centers = [abs(float(diagnostics[hand].get("center_m", np.inf))) for hand in HANDS]
    center_gate = bool(
        all(diagnostics[hand].get("accepted", False) for hand in HANDS)
        and max(centers) <= config.absolute_center_gate_m
    )
    interaction_core, protection = interaction_protection_weight(payloads, frames, config)
    common = 0.5 * (corrections["left"] + corrections["right"])
    delta: dict[str, np.ndarray] = {}
    world_shift: dict[str, np.ndarray] = {}
    for hand in HANDS:
        own = config.fusion_beta * corrections[hand] if center_gate else np.zeros(frames)
        shared = config.fusion_beta * common if center_gate else np.zeros(frames)
        delta[hand] = (1.0 - protection) * own + protection * shared
        world_shift[hand] = camera_z_to_world(payloads[hand]["camera_R_c2w"], delta[hand])

    before = run_detector(
        payloads,
        {hand: payloads[hand]["vertices_world"] for hand in HANDS},
        args.h3mr_code_root.resolve(),
    )
    tentative = {
        hand: payloads[hand]["vertices_world"].astype(np.float64)
        + world_shift[hand][:, None, :]
        for hand in HANDS
    }
    after = run_detector(payloads, tentative, args.h3mr_code_root.resolve())
    new_penetration = after["penetration_detected"] & ~before["penetration_detected"]
    tentative_new_frames = int(new_penetration.sum())
    before_energy = float(before["penetration_energy"].sum())
    after_energy = float(after["penetration_energy"].sum())
    penetration_gate = bool(not new_penetration.any() and after_energy <= before_energy + 1e-10)
    fallback_reason = None
    if not center_gate:
        fallback_reason = "center_or_support_gate_failed"
    elif not penetration_gate:
        fallback_reason = "penetration_gate_failed"
        for hand in HANDS:
            delta[hand] = np.zeros(frames, dtype=np.float64)
        after = before

    outputs = {}
    for hand in HANDS:
        candidate = apply_root_depth_correction(payloads[hand], delta[hand], after, config)
        output = candidate_path(args.output_root.resolve(), sequence, hand)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        atomic_npz(output, candidate)
        outputs[hand] = {"path": str(output), "sha256": sha256(output)}
    return {
        "sequence": sequence,
        "branch": "v2_ordinary",
        "candidate_generation_uses_gt": False,
        "ordinary_branch_evidence": {
            "valid_keyframes": int(record["valid_keyframes"]),
            "scale_relative_mad": float(record["scale_relative_mad"]),
        },
        "gates": {
            "center_gate": center_gate,
            "penetration_gate": penetration_gate,
            "fallback_reason": fallback_reason,
        },
        "interaction": {
            "core_frames": int(interaction_core.sum()),
            "protected_support_frames": int((protection > 0).sum()),
        },
        "hand_diagnostics": diagnostics,
        "penetration": {
            "before_frames": int(before["penetration_detected"].sum()),
            "after_frames": int(after["penetration_detected"].sum()),
            "new_frames": int(
                (after["penetration_detected"] & ~before["penetration_detected"]).sum()
            ),
            "tentative_new_frames_before_fallback": tentative_new_frames,
            "before_energy": before_energy,
            "after_energy": float(after["penetration_energy"].sum()),
        },
        "inputs": {
            "vda_depth": {"path": str(depth_path), "sha256": sha256(depth_path)},
            "stage3_bir": {
                hand: {
                    "path": str(find_hand_npz(stage3_root, sequence, hand)),
                    "sha256": sha256(find_hand_npz(stage3_root, sequence, hand)),
                }
                for hand in HANDS
            },
        },
        "outputs": outputs,
    }


def materialize_v1_sequence(
    record: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    sequence = str(record["sequence"])
    outputs = {}
    inputs = {}
    for hand in HANDS:
        source = find_hand_npz(args.v1_hard_root.resolve(), sequence, hand)
        destination = candidate_path(args.output_root.resolve(), sequence, hand)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"V1 copy SHA mismatch for {sequence}/{hand}")
        inputs[hand] = {"path": str(source), "sha256": sha256(source)}
        outputs[hand] = {"path": str(destination), "sha256": sha256(destination)}
    return {
        "sequence": sequence,
        "branch": "v1_hard_dense_recovery",
        "candidate_generation_uses_gt": False,
        "hard_branch_evidence": {
            "valid_keyframes": int(record["valid_keyframes"]),
            "scale_relative_mad": float(record["scale_relative_mad"]),
        },
        "inputs": {"frozen_v1": inputs},
        "outputs": outputs,
    }


def generate(args: argparse.Namespace, preflight_record: Mapping[str, Any]) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "candidate_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(preflight_record["selected_gpu"]["physical_index"])
    config = AdaptiveVDAConfig()
    records = load_records(args.scale_summary.resolve())
    sequence_records = []
    for index, record in enumerate(records, start=1):
        sequence = str(record["sequence"])
        hard = is_hard_sequence(
            int(record["valid_keyframes"]), float(record["scale_relative_mad"]), config
        )
        print(f"[generate {index:02d}/{len(records):02d}] {sequence}", flush=True)
        result = (
            materialize_v1_sequence(record, args)
            if hard
            else generate_v2_sequence(record, args, config)
        )
        sequence_manifest = output_root / "manifests" / f"{sequence}.json"
        atomic_json(sequence_manifest, result)
        sequence_records.append(
            {
                "sequence": sequence,
                "branch": result["branch"],
                "manifest": str(sequence_manifest),
                "manifest_sha256": sha256(sequence_manifest),
            }
        )
    result = {
        "name": "Stage-3-BIR-plus-Adaptive-VDA",
        "status": "candidate_generated",
        "version": VERSION,
        "candidate_generation_uses_gt": False,
        "routing": (
            "V1 for frozen hard trigger; V2 for ordinary sequences; "
            "guarded fallback preserves Stage-3-BIR"
        ),
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "preflight": preflight_record,
        "sequences": sequence_records,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest_path, result)
    return result


def distribution(parts: list[np.ndarray]) -> dict[str, float | int]:
    finite = [part[np.isfinite(part)] for part in parts if np.isfinite(part).any()]
    values = np.concatenate(finite) if finite else np.asarray([], dtype=np.float64)
    if not len(values):
        raise RuntimeError("metric contains no finite samples")
    return {
        "samples": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def h2o_sequence_dir(h2o_root: Path, sequence: str) -> Path:
    subject, activity, take = sequence.split("__")
    return h2o_root / subject / activity / take / "cam4"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_generation_uses_gt") is not False:
        raise RuntimeError("candidate manifest does not certify GT-free generation")
    for item in manifest["sequences"]:
        sequence_manifest = Path(item["manifest"])
        if sha256(sequence_manifest) != item["manifest_sha256"]:
            raise RuntimeError(f"sequence manifest SHA mismatch: {item['sequence']}")
        payload = json.loads(sequence_manifest.read_text(encoding="utf-8"))
        for hand in HANDS:
            if sha256(Path(payload["outputs"][hand]["path"])) != payload["outputs"][hand]["sha256"]:
                raise RuntimeError(f"candidate SHA mismatch: {item['sequence']}/{hand}")

    h2o_root = resolve_h2o_root(args)
    evaluation_dir = (args.evaluation_dir or args.h3mr_code_root / "evaluation").resolve()
    hawor_root = (args.hawor_root or args.h3mr_code_root / "hawor").resolve()
    previous = Path.cwd()
    sys.path.insert(0, str(hawor_root))
    sys.path.insert(0, str(evaluation_dir))
    try:
        os.chdir(hawor_root)
        from evaluate_hawor_h2o_mano_baseline import (  # noqa: PLC0415
            evaluate_hand,
            load_h2o_ground_truth,
        )

        collected = {
            method: {metric: [] for metric in METRICS}
            for method in ("stage3_bir", "stage3_bir_plus_adaptive_vda")
        }
        per_frame_payload = {}
        per_sequence = []
        records = load_records(args.scale_summary.resolve())
        for index, record in enumerate(records, start=1):
            sequence = str(record["sequence"])
            print(f"[evaluate {index:02d}/{len(records):02d}] {sequence}", flush=True)
            gt = load_h2o_ground_truth(h2o_sequence_dir(h2o_root, sequence))
            sequence_parts = {
                method: {metric: [] for metric in METRICS}
                for method in collected
            }
            shared_counts = {}
            for hand in HANDS:
                base = load_npz(find_hand_npz(args.stage3_root.resolve(), sequence, hand))
                candidate = load_npz(candidate_path(output_root, sequence, hand))
                shared = (
                    gt[hand]["valid"]
                    & base["valid"].astype(bool)
                    & candidate["valid"].astype(bool)
                )
                shared_counts[hand] = int(shared.sum())
                gt_shared = dict(gt[hand])
                gt_shared["valid"] = shared
                predictions = {
                    "stage3_bir": {
                        "valid": shared,
                        "trans_world": base["transl"],
                        "joints_world": base["joints_3d_world"],
                    },
                    "stage3_bir_plus_adaptive_vda": {
                        "valid": shared,
                        "trans_world": candidate["transl"],
                        "joints_world": candidate["joints_3d_world"],
                    },
                }
                for method, prediction in predictions.items():
                    per_frame, _ = evaluate_hand(gt_shared, prediction, 100)
                    for metric in METRICS:
                        collected[method][metric].append(per_frame[metric])
                        sequence_parts[method][metric].append(per_frame[metric])
                        key = f"{sequence}__{hand}__{method}__{metric}"
                        per_frame_payload[key] = per_frame[metric]
                per_frame_payload[f"{sequence}__{hand}__shared_valid"] = shared
            per_sequence.append(
                {
                    "sequence": sequence,
                    "shared_valid_hand_frames": shared_counts,
                    "metrics": {
                        method: {
                            metric: distribution(parts)["mean"]
                            for metric, parts in metrics.items()
                        }
                        for method, metrics in sequence_parts.items()
                    },
                }
            )
    finally:
        os.chdir(previous)

    aggregate = {
        method: {metric: distribution(parts) for metric, parts in metrics.items()}
        for method, metrics in collected.items()
    }
    comparison = {}
    for metric in METRICS:
        before = aggregate["stage3_bir"][metric]["mean"]
        after = aggregate["stage3_bir_plus_adaptive_vda"][metric]["mean"]
        comparison[metric] = {
            "stage3_bir": before,
            "stage3_bir_plus_adaptive_vda": after,
            "absolute_improvement": before - after,
            "relative_improvement_percent": (before - after) / before * 100.0,
        }
    result = {
        "name": "Stage-3-BIR-vs-Stage-3-BIR-plus-Adaptive-VDA",
        "status": "complete",
        "candidate_generation_uses_gt": False,
        "gt_first_read_after_candidate_manifest": True,
        "shared_validity": "GT valid AND Stage-3-BIR valid AND candidate valid",
        "aggregate": aggregate,
        "comparison": comparison,
        "per_sequence": per_sequence,
        "candidate_manifest_sha256": sha256(manifest_path),
    }
    evaluation_root = output_root / "evaluation"
    atomic_npz(evaluation_root / "per_frame_metrics.npz", per_frame_payload)
    atomic_json(evaluation_root / "summary.json", result)
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequence_results = []
    passed = manifest.get("candidate_generation_uses_gt") is False
    for item in manifest["sequences"]:
        sequence = item["sequence"]
        detail_path = Path(item["manifest"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        record: dict[str, Any] = {"sequence": sequence, "branch": detail["branch"], "hands": {}}
        for hand in HANDS:
            output = candidate_path(output_root, sequence, hand)
            output_sha_ok = sha256(output) == detail["outputs"][hand]["sha256"]
            candidate = load_npz(output)
            if detail["branch"] == "v1_hard_dense_recovery":
                source = Path(detail["inputs"]["frozen_v1"][hand]["path"])
                checks = {
                    "output_sha_matches_manifest": output_sha_ok,
                    "frozen_v1_copy_is_exact": sha256(source) == sha256(output),
                }
                errors = {}
            else:
                base = load_npz(find_hand_npz(args.stage3_root.resolve(), sequence, hand))
                unchanged_failures = [
                    key
                    for key in base
                    if key not in ALLOWED_V2_CHANGES and not arrays_equal(base[key], candidate[key])
                ]
                joints_expected = (
                    np.einsum(
                        "tij,tnj->tni",
                        candidate["camera_R_c2w"].astype(np.float64),
                        candidate["joints_3d_camera"].astype(np.float64),
                    )
                    + candidate["camera_t_c2w"].astype(np.float64)[:, None, :]
                )
                vertices_expected = (
                    np.einsum(
                        "tij,tnj->tni",
                        candidate["camera_R_c2w"].astype(np.float64),
                        candidate["vertices_camera"].astype(np.float64),
                    )
                    + candidate["camera_t_c2w"].astype(np.float64)[:, None, :]
                )
                projected = project_camera_points(
                    candidate["joints_3d_camera"], candidate["camera_intrinsics"]
                )
                local_before = (
                    base["joints_3d_world"].astype(np.float64)
                    - base["joints_3d_world"][:, :1]
                )
                local_after = (
                    candidate["joints_3d_world"].astype(np.float64)
                    - candidate["joints_3d_world"][:, :1]
                )
                errors = {
                    "joint_world_consistency_max_m": float(
                        np.max(np.abs(joints_expected - candidate["joints_3d_world"]))
                    ),
                    "vertex_world_consistency_max_m": float(
                        np.max(np.abs(vertices_expected - candidate["vertices_world"]))
                    ),
                    "joint_2d_reprojection_max_px": float(
                        np.nanmax(np.abs(projected - candidate["joints_2d"]))
                    ),
                    "within_hand_geometry_change_max_m": float(
                        np.max(np.abs(local_after - local_before))
                    ),
                }
                checks = {
                    "output_sha_matches_manifest": output_sha_ok,
                    "all_original_fields_present": set(base).issubset(candidate),
                    "unchanged_non_whitelist_fields": not unchanged_failures,
                    "valid_mask_unchanged": arrays_equal(base["valid"], candidate["valid"]),
                    "camera_fields_unchanged": all(
                        arrays_equal(base[key], candidate[key])
                        for key in (
                            "camera_R_w2c",
                            "camera_t_w2c",
                            "camera_R_c2w",
                            "camera_t_c2w",
                            "camera_intrinsics",
                        )
                    ),
                    "fallback_preserves_all_original_fields": (
                        detail["gates"]["fallback_reason"] is None
                        or all(arrays_equal(base[key], candidate[key]) for key in base)
                    ),
                    "modified_fields_are_finite": all(
                        key in candidate
                        and (
                            not np.issubdtype(candidate[key].dtype, np.number)
                            or np.isfinite(candidate[key]).all()
                        )
                        for key in V2_FINITE_FIELDS
                    ),
                    "coordinate_consistency": (
                        errors["joint_world_consistency_max_m"] < 5e-7
                        and errors["vertex_world_consistency_max_m"] < 5e-7
                        and errors["joint_2d_reprojection_max_px"] < 2e-4
                    ),
                    "within_hand_geometry_preserved": (
                        errors["within_hand_geometry_change_max_m"] < 2e-7
                    ),
                }
            hand_passed = all(checks.values())
            passed &= hand_passed
            record["hands"][hand] = {"passed": hand_passed, "checks": checks, "errors": errors}
        sequence_results.append(record)
    temporary_files = sorted(
        str(path)
        for path in output_root.rglob(".*")
        if path.is_file() and path.name.startswith(".")
    )
    passed &= not temporary_files
    result = {
        "name": "Stage-3-BIR-plus-Adaptive-VDA-structural-audit",
        "status": "passed" if passed else "failed",
        "candidate_generation_uses_gt": manifest.get("candidate_generation_uses_gt"),
        "sequences": sequence_results,
        "temporary_files": temporary_files,
    }
    atomic_json(output_root / "audit.json", result)
    if not passed:
        raise RuntimeError("Adaptive VDA audit failed")
    return result


def main() -> int:
    args = parse_args()
    require_gt = args.command in ("evaluate", "all")
    preflight_record = preflight(args, require_gt=require_gt)
    if args.command == "preflight":
        print(json.dumps(preflight_record, indent=2, ensure_ascii=False))
        return 0
    lease = None
    try:
        if args.command in ("generate", "all"):
            lease_path, lease = acquire_gpu_lease(
                int(preflight_record["selected_gpu"]["physical_index"])
            )
            preflight_record["gpu_lease_path"] = str(lease_path)
            print(json.dumps(generate(args, preflight_record), indent=2, ensure_ascii=False))
        if args.command in ("evaluate", "all"):
            print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))
        if args.command in ("audit", "all"):
            print(json.dumps(audit(args), indent=2, ensure_ascii=False))
    finally:
        if lease is not None:
            lease.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
