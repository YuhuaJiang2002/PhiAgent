#!/usr/bin/env python3
"""Generate MiniMax-H3 proposals for one frozen camera-frame challenge.

This entry point is intended to be staged beside the already pinned MiniMax-H3
runner helpers used by the historical T-shirt strategy run. It inventories and
leases physical GPUs, saves the UUID mapping and package state, refuses output
reuse, and treats every video as an unevaluated visual proposal.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_minimax_h3_fl2va_tshirt_tts import (  # noqa: E402
    _gpu_inventory,
    _lease_gpus,
    _probe_video,
    _run,
    _runtime_environment,
    _runtime_probe,
    _select_gpus,
    _sha256,
    _submit_and_wait,
    _utc_now,
    _wait_for_server,
    _write_json,
)
try:  # Keep the staged 2026-08-23 runner ABI usable for append-only reruns.
    from scripts.run_minimax_h3_fl2va_tshirt_tts import _declared_runtime  # noqa: E402
except ImportError:  # pragma: no cover - exercised by the pinned remote runtime
    def _declared_runtime(spec: Mapping[str, Any]) -> tuple[str, str, None]:
        runtime = spec["model"]["runtime"]
        return str(runtime["torch"]), str(runtime["cuda"]), None
from scripts.run_minimax_h3_ref2va_optical_module import (  # noqa: E402
    _build_model_view,
    _checkpoint_sources,
)


H0_VERSION = "tshirt-direct-fold-h0"
H1_VERSION = "tshirt-canonicalize-verify-fold-h1"
H1_FAST_VERSION = "tshirt-timeboxed-canonicalize-fold-h1-fast"
H1_BALANCED_VERSION = "tshirt-timeboxed-canonicalize-fold-h1-balanced"
H1_REFERENCE_VERSION = "tshirt-reference-anchored-timeboxed-fold-h1r"
H2_VERSION = "tshirt-material-law-canonicalize-fold-h2"
H3_VERSION = "tshirt-short-sleeve-material-lock-h3"
H3_CAMERA_ALIGNED_VERSION = "tshirt-camera-aligned-short-sleeve-material-lock-h3c"
H18_VISUAL_ACTION_PROMPT_VERSION = (
    "tshirt-visual-action-prompt-terminal-containment-h18"
)
H19_FUSED_ACTION_FLOW_VERSION = "tshirt-fused-action-flow-terminal-containment-h19"
H4_VERSION = "tshirt-verified-incremental-recovery-h4"
FIGURE_BLANKET_VERSION = "figure-two-robot-blanket-fold-photorealistic-v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--challenge-frame", type=Path, required=True)
    parser.add_argument(
        "--harness-version",
        choices=(
            H0_VERSION,
            H1_VERSION,
            H1_FAST_VERSION,
            H1_BALANCED_VERSION,
            H1_REFERENCE_VERSION,
            H2_VERSION,
            H3_VERSION,
            H3_CAMERA_ALIGNED_VERSION,
            H18_VISUAL_ACTION_PROMPT_VERSION,
            H19_FUSED_ACTION_FLOW_VERSION,
            H4_VERSION,
            FIGURE_BLANKET_VERSION,
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--transformer-overlay", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30371)
    parser.add_argument("--master-port", type=int, default=30372)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument(
        "--motion-reference",
        type=Path,
        help="optional hash-bound 1024x768/24 FPS/192-frame motion reference",
    )
    parser.add_argument(
        "--identity-exemplar",
        type=Path,
        help="optional second Ref2VA image condition carrying garment identity only",
    )
    parser.add_argument(
        "--identity-exemplar-sha256",
        help="required expected SHA-256 when --identity-exemplar is supplied",
    )
    parser.add_argument(
        "--visual-action-prompt",
        type=Path,
        help="optional hash-bound 1024x768/24 FPS/192-frame Video-2 action prompt",
    )
    parser.add_argument(
        "--python-include-root",
        type=Path,
        help=(
            "optional pinned include root containing python3.12/Python.h and "
            "x86_64-linux-gnu/python3.12/pyconfig.h"
        ),
    )
    parser.add_argument(
        "--physical-gpu-indices",
        help="comma-separated physical GPU indices; overrides only the run lane",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="stop after a successful media-contract-checked smoke render",
    )
    parser.add_argument("--seed-start-index", type=int, default=0)
    parser.add_argument("--seed-count", type=int)
    return parser


def _required(path: Path, label: str, *, directory: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _payload(
    *,
    spec: Mapping[str, Any],
    prompt: str,
    image: Path,
    identity_exemplar: Path | None,
    video: Path,
    visual_action_prompt: Path | None,
    seed: int,
    steps: int,
) -> dict[str, object]:
    generation = spec["generation"]
    conditions = [
        {"type": "image", "uri": image.resolve().as_uri(), "role": "reference"}
    ]
    if identity_exemplar is not None:
        conditions.append(
            {
                "type": "image",
                "uri": identity_exemplar.resolve().as_uri(),
                "role": "reference",
            }
        )
    conditions.append(
        {"type": "video", "uri": video.resolve().as_uri(), "role": "reference"}
    )
    if visual_action_prompt is not None:
        conditions.append(
            {
                "type": "video",
                "uri": visual_action_prompt.resolve().as_uri(),
                "role": "reference",
            }
        )
    return {
        "model": spec["model"]["id"],
        "prompt": prompt,
        "seconds": generation["target"]["duration_seconds"],
        "task": "ref2va",
        "conditions": conditions,
        "target": generation["target"],
        "num_outputs_per_prompt": 1,
        "num_inference_steps": steps,
        "flow_shift": generation["flow_shift"],
        "audio_flow_shift": generation["audio_flow_shift"],
        "quality": generation["quality"],
        "seed": seed,
    }


def _static_reference(frame: Path, output: Path, spec: Mapping[str, Any]) -> list[str]:
    generation = spec["generation"]
    expected_frames = int(generation["expected_frames"])
    fps = float(generation["fps"])
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(frame),
        "-frames:v",
        str(expected_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = _run(command, timeout=600)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "failed to create static reference")
    probe = _probe_video(output)
    stream = probe["streams"][0]
    if (
        int(stream["width"]) != int(spec["challenge"]["width"])
        or int(stream["height"]) != int(spec["challenge"]["height"])
        or int(stream["nb_read_frames"]) != expected_frames
        or stream["r_frame_rate"] != "24/1"
    ):
        raise ValueError("static challenge reference differs from the frozen media contract")
    return command


def _validate_media_contract(
    video: Path,
    *,
    challenge: Mapping[str, Any],
    generation: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    probe = _probe_video(video)
    stream = probe["streams"][0]
    if (
        int(stream["width"]) != int(challenge["width"])
        or int(stream["height"]) != int(challenge["height"])
        or int(stream["nb_read_frames"]) != int(generation["expected_frames"])
        or stream["r_frame_rate"] != f"{int(generation['fps'])}/1"
    ):
        raise ValueError(f"{label} violates the frozen media contract")
    return probe


def main() -> int:
    args = _parser().parse_args()
    spec_path = _required(args.spec, "generation spec")
    frame = _required(args.challenge_frame, "frozen challenge frame")
    runtime_env = _required(args.runtime_env, "pinned H3 runtime", directory=True)
    overlay = _required(args.transformer_overlay, "Ref2VA transformer overlay", directory=True)
    motion_reference = (
        None
        if args.motion_reference is None
        else _required(args.motion_reference, "motion reference")
    )
    identity_exemplar = (
        None
        if args.identity_exemplar is None
        else _required(args.identity_exemplar, "garment identity exemplar")
    )
    visual_action_prompt = (
        None
        if args.visual_action_prompt is None
        else _required(args.visual_action_prompt, "visual action prompt")
    )
    if (
        args.harness_version == H18_VISUAL_ACTION_PROMPT_VERSION
    ) != (visual_action_prompt is not None):
        raise ValueError("H18 requires exactly one --visual-action-prompt")
    if (identity_exemplar is None) != (args.identity_exemplar_sha256 is None):
        raise ValueError(
            "--identity-exemplar and --identity-exemplar-sha256 must be supplied together"
        )
    if identity_exemplar is not None and _sha256(identity_exemplar) != str(
        args.identity_exemplar_sha256
    ):
        raise ValueError("garment identity exemplar hash mismatch")
    python_include_root = (
        None
        if args.python_include_root is None
        else _required(
            args.python_include_root,
            "pinned Python include root",
            directory=True,
        )
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse generation run directory: {output}")
    component_config = output / "model_view/Ref2VA/text_encoder/config.json"
    filelock_name = (
        "0" * 64
        + "-"
        + str(component_config).lstrip("/").replace("/", "-")
        + ".lock"
    )
    if len(os.fsencode(filelock_name)) > 255:
        raise ValueError(
            "generation output path is too long for the SGLang component lock; "
            "use a shorter append-only run ID"
        )
    if args.smoke_steps < 1:
        raise ValueError("smoke steps must be positive")
    if args.smoke_only and args.skip_smoke:
        raise ValueError("--smoke-only and --skip-smoke are mutually exclusive")
    if args.master_port == args.port:
        raise ValueError("server and distributed master ports must differ")
    if args.seed_start_index < 0 or (
        args.seed_count is not None and args.seed_count < 1
    ):
        raise ValueError("seed selection requires a non-negative start and positive count")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("generation spec must be one JSON object")
    if spec.get("status") == "BLOCKED":
        raise ValueError(
            "generation spec is BLOCKED by its frozen execution preflight; "
            "resolve the recorded workflow and acceptance-contract blockers and "
            "prepare a new append-only spec before GPU execution"
        )
    challenge = spec.get("challenge")
    model = spec.get("model")
    generation = spec.get("generation")
    compute = spec.get("compute")
    raw_cases = spec.get("cases")
    if not all(isinstance(item, dict) for item in (challenge, model, generation, compute)):
        raise ValueError("generation spec requires challenge, model, generation, and compute")
    if not isinstance(raw_cases, list) or not all(isinstance(item, dict) for item in raw_cases):
        raise ValueError("generation spec cases must be an array of objects")
    cases = []
    for item in raw_cases:
        if item.get("harness_version") != args.harness_version:
            continue
        declared_seeds = list(item.get("seeds", []))
        stop = (
            None
            if args.seed_count is None
            else args.seed_start_index + args.seed_count
        )
        selected = declared_seeds[args.seed_start_index:stop]
        if not selected:
            continue
        cases.append(
            {
                **item,
                "seeds": selected,
                "declared_seed_indices": list(
                    range(
                        args.seed_start_index,
                        args.seed_start_index + len(selected),
                    )
                ),
            }
        )
    if not cases:
        raise ValueError("the selected harness requires at least one generation case")
    if (
        _sha256(frame) != challenge["initial_frame_sha256"]
        or not str(challenge.get("challenge_id", "")).strip()
        or any(item.get("challenge_sha256") != challenge["challenge_sha256"] for item in cases)
    ):
        raise ValueError("challenge frame or case hash differs from the frozen contract")
    if (
        model.get("id") != "MiniMaxAI/MiniMax-H3"
        or model.get("partition") != "Ref2VA"
        or model.get("dtype") != "bfloat16"
        or model.get("quantization") is not None
        or model.get("runtime", {}).get("attention_backend") != "torch_sdpa"
    ):
        raise ValueError(
            "generation requires unquantized H3 BF16 Ref2VA with the verified "
            "torch_sdpa backend"
        )
    if (
        generation.get("task") != "ref2va"
        or generation.get("num_inference_steps") != 50
        or generation.get("fps") != 24
        or generation.get("expected_frames") != 192
    ):
        raise ValueError(
            "generation requires Ref2VA, 50 steps, 24 FPS, and 192 frames"
        )

    output.mkdir(parents=True)
    (output / "cache").mkdir()
    frozen = output / "frozen-inputs"
    frozen.mkdir()
    frozen_spec = frozen / "remote-generation-spec.json"
    frozen_frame = frozen / f"{challenge['challenge_id']}.png"
    shutil.copy2(spec_path, frozen_spec)
    shutil.copy2(frame, frozen_frame)
    frozen_identity_exemplar = None
    identity_exemplar_record = None
    if identity_exemplar is not None:
        frozen_identity_exemplar = frozen / "identity-exemplar.png"
        shutil.copy2(identity_exemplar, frozen_identity_exemplar)
        identity_exemplar_record = {
            "role": "garment_identity_exemplar",
            "source": str(identity_exemplar),
            "path": str(frozen_identity_exemplar),
            "sha256": _sha256(frozen_identity_exemplar),
        }
    if motion_reference is None:
        reference_video = frozen / "static-challenge-reference.mp4"
        reference_command = _static_reference(frozen_frame, reference_video, spec)
        reference_record = {
            "role": "static_scene_identity",
            "path": str(reference_video),
            "sha256": _sha256(reference_video),
        }
    else:
        reference_video = frozen / "motion-reference.mp4"
        shutil.copy2(motion_reference, reference_video)
        expected_motion_sha = str(cases[0].get("motion_reference_sha256", ""))
        if not expected_motion_sha or _sha256(reference_video) != expected_motion_sha:
            raise ValueError("motion reference differs from the frozen spec")
        reference_probe = _probe_video(reference_video)
        reference_stream = reference_probe["streams"][0]
        if (
            int(reference_stream["width"]) != int(challenge["width"])
            or int(reference_stream["height"]) != int(challenge["height"])
            or int(reference_stream["nb_read_frames"]) != int(generation["expected_frames"])
            or reference_stream["r_frame_rate"] != "24/1"
        ):
            raise ValueError("motion reference differs from the frozen media contract")
        reference_command = None
        reference_record = {
            "role": "positive_fold_motion",
            "source": str(motion_reference),
            "path": str(reference_video),
            "sha256": _sha256(reference_video),
            "probe": reference_probe,
        }
    frozen_visual_action_prompt = None
    visual_action_prompt_record = None
    if visual_action_prompt is not None:
        frozen_visual_action_prompt = frozen / "visual-action-prompt.mp4"
        shutil.copy2(visual_action_prompt, frozen_visual_action_prompt)
        visual_prompt_probe = _probe_video(frozen_visual_action_prompt)
        visual_prompt_stream = visual_prompt_probe["streams"][0]
        if (
            int(visual_prompt_stream["width"]) != int(challenge["width"])
            or int(visual_prompt_stream["height"]) != int(challenge["height"])
            or int(visual_prompt_stream["nb_read_frames"])
            != int(generation["expected_frames"])
            or visual_prompt_stream["r_frame_rate"] != "24/1"
        ):
            raise ValueError("visual action prompt differs from the frozen media contract")
        expected_visual_prompt_sha = str(
            cases[0].get("visual_action_prompt_sha256", "")
        )
        if (
            not expected_visual_prompt_sha
            or _sha256(frozen_visual_action_prompt) != expected_visual_prompt_sha
        ):
            raise ValueError("visual action prompt differs from the frozen spec")
        visual_action_prompt_record = {
            "role": "camera_aligned_visual_action_prompt",
            "source": str(visual_action_prompt),
            "path": str(frozen_visual_action_prompt),
            "sha256": expected_visual_prompt_sha,
            "probe": visual_prompt_probe,
        }
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "decision": "PREFLIGHT",
        "created_at": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "harness_version": args.harness_version,
        "challenge_sha256": challenge["challenge_sha256"],
        "challenge_frame_sha256": _sha256(frozen_frame),
        "spec_sha256": _sha256(frozen_spec),
        "selected_cases": [item["label"] for item in cases],
        "seed_start_index": args.seed_start_index,
        "seed_count": args.seed_count,
        "candidate_budget": sum(len(item["seeds"]) for item in cases),
        "launcher_command": [sys.executable, *sys.argv],
        "commands": [] if reference_command is None else [reference_command],
        "video_reference": reference_record,
        "identity_exemplar": identity_exemplar_record,
        "visual_action_prompt": visual_action_prompt_record,
        "completed_candidates": [],
    }
    _write_json(output / "metadata.json", metadata)
    server: subprocess.Popen[str] | None = None
    leases: list[BinaryIO] = []
    log_stream = None
    try:
        run_compute = dict(compute)
        if args.physical_gpu_indices:
            indices_override = [
                int(value) for value in args.physical_gpu_indices.split(",")
            ]
            if len(indices_override) != int(run_compute["num_gpus"]):
                raise ValueError(
                    "physical GPU override count must match compute.num_gpus"
                )
            if len(set(indices_override)) != len(indices_override):
                raise ValueError("physical GPU override indices must be unique")
            run_compute["physical_gpu_indices"] = indices_override
        config = {"compute": run_compute}
        inventory = _gpu_inventory()
        selected_gpus = _select_gpus(inventory, config)
        indices = [int(item["physical_index"]) for item in selected_gpus]
        leases = _lease_gpus(indices)
        sources = _checkpoint_sources(Path(model["checkpoint_root"]), overlay)
        expected_torch, expected_cuda, cuda_home = _declared_runtime(spec)
        environment_parameters = inspect.signature(_runtime_environment).parameters
        if "cuda_home" in environment_parameters:
            environment = _runtime_environment(
                runtime_env,
                indices,
                output,
                cuda_home=cuda_home,
                expected_cuda=expected_cuda,
            )
        else:
            environment = _runtime_environment(runtime_env, indices, output)
        compiler_preflight = None
        if python_include_root is not None:
            python_include = _required(
                python_include_root / "python3.12/Python.h",
                "pinned Python.h",
            )
            pyconfig = _required(
                python_include_root / "x86_64-linux-gnu/python3.12/pyconfig.h",
                "pinned pyconfig.h",
            )
            include_path = (
                f"{python_include_root / 'python3.12'}:{python_include_root}"
            )
            if environment.get("CPATH"):
                include_path += f":{environment['CPATH']}"
            environment["CPATH"] = include_path
            completed = subprocess.run(
                ["/usr/bin/gcc", "-x", "c", "-fsyntax-only", "-"],
                input="#include <Python.h>\nint main(void) { return 0; }\n",
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            compiler_preflight = {
                "command": ["/usr/bin/gcc", "-x", "c", "-fsyntax-only", "-"],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "python_h": str(python_include),
                "pyconfig_h": str(pyconfig),
                "cpath": include_path,
            }
            if completed.returncode:
                raise RuntimeError(
                    completed.stderr.strip()
                    or "pinned Python-header compiler preflight failed"
                )
        probe_parameters = inspect.signature(_runtime_probe).parameters
        if "expected_torch" in probe_parameters:
            runtime = _runtime_probe(
                runtime_env,
                environment,
                run_compute["required_gpu_name"],
                expected_torch=expected_torch,
                expected_cuda=expected_cuda,
                expected_device_count=int(run_compute["num_gpus"]),
            )
        else:
            runtime = _runtime_probe(
                runtime_env,
                environment,
                run_compute["required_gpu_name"],
            )
        _write_json(
            output / "gpu_selection.json",
            {"created_at": _utc_now(), "inventory": inventory, "selected": selected_gpus},
        )
        _write_json(output / "runtime_probe.json", runtime)
        if compiler_preflight is not None:
            _write_json(output / "compiler_preflight.json", compiler_preflight)
        _write_json(
            output / "checkpoint_evidence.json",
            {
                "checkpoint_root": model["checkpoint_root"],
                "revision": model["revision"],
                "partition": model["partition"],
                "dtype": model["dtype"],
                "quantization": model["quantization"],
                "transformer_shards": [
                    {"path": str(path), "bytes": path.stat().st_size}
                    for path in sources["transformer_paths"]
                ],
                "text_encoder_shards": [
                    {"path": str(path), "bytes": path.stat().st_size}
                    for path in sources["text_encoder_paths"]
                ],
            },
        )
        model_view = _build_model_view(output, sources)
        freeze = _run(
            [str(runtime_env / "bin/python"), "-m", "pip", "freeze", "--all"],
            env=environment,
            timeout=300,
        )
        if freeze.returncode:
            raise RuntimeError(freeze.stderr.strip() or "failed to capture packages")
        (output / "packages.txt").write_text(freeze.stdout, encoding="utf-8")
        command = [
            str(runtime_env / "bin/python"),
            str(runtime_env / "bin/sglang"),
            "serve",
            "--model-path",
            str(model_view),
            "--model-type",
            "diffusion",
            "--model-variant",
            "ref2va",
            "--num-gpus",
            str(run_compute["num_gpus"]),
            "--ulysses-degree",
            str(run_compute["ulysses_degree"]),
            "--use-fsdp-inference",
            str(run_compute["use_fsdp_inference"]).lower(),
            "--performance-mode",
            str(model["runtime"]["performance_mode"]),
            "--enable-torch-compile",
            "false",
            "--attention-backend",
            str(model["runtime"]["attention_backend"]),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--master-port",
            str(args.master_port),
        ]
        metadata["commands"].append(command)
        metadata["decision"] = "SERVER_STARTING"
        _write_json(output / "metadata.json", metadata)
        log_stream = (output / "server.log").open("w", encoding="utf-8")
        server = subprocess.Popen(
            command,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        (output / "server.pid").write_text(str(server.pid) + "\n", encoding="utf-8")
        api = f"http://127.0.0.1:{args.port}"
        _wait_for_server(api, server)
        metadata["decision"] = "SERVER_READY"
        metadata["server_ready_at"] = _utc_now()
        _write_json(output / "metadata.json", metadata)

        if args.skip_smoke:
            metadata["smoke"] = {"status": "SKIPPED_BY_DECLARED_RUN_POLICY"}
        else:
            smoke_payload = _payload(
                spec=spec,
                prompt=str(cases[0]["prompt"]),
                image=frozen_frame,
                identity_exemplar=frozen_identity_exemplar,
                video=reference_video,
                visual_action_prompt=frozen_visual_action_prompt,
                seed=int(cases[0]["seeds"][0]),
                steps=args.smoke_steps,
            )
            metadata["smoke"] = _submit_and_wait(
                api=api,
                payload=smoke_payload,
                candidate_dir=output / "smoke",
            )
            smoke_video = Path(str(metadata["smoke"]["video"]))
            metadata["smoke"]["video_probe"] = _validate_media_contract(
                smoke_video,
                challenge=challenge,
                generation=generation,
                label="smoke candidate",
            )
        if args.smoke_only:
            metadata["status"] = "PARTIAL"
            metadata["decision"] = "SMOKE_COMPLETE_PENDING_NATIVE_REVIEW"
            metadata["completed_at"] = _utc_now()
            metadata["claim_boundary"] = spec["claim_boundary"]
            _write_json(output / "metadata.json", metadata)
            print(
                json.dumps(
                    {
                        "status": metadata["status"],
                        "decision": metadata["decision"],
                        "output_dir": str(output),
                        "harness_version": args.harness_version,
                        "challenge_sha256": challenge["challenge_sha256"],
                        "smoke_video_sha256": metadata["smoke"]["video_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        metadata["decision"] = "GENERATING"
        _write_json(output / "metadata.json", metadata)
        for case in cases:
            case_id = str(case["label"])
            for candidate_offset, seed_value in enumerate(case["seeds"]):
                seed = int(seed_value)
                candidate_index = int(case["declared_seed_indices"][candidate_offset])
                candidate_dir = output / "candidates" / case_id / f"seed-{seed}"
                request = _payload(
                    spec=spec,
                    prompt=str(case["prompt"]),
                    image=frozen_frame,
                    identity_exemplar=frozen_identity_exemplar,
                    video=reference_video,
                    visual_action_prompt=frozen_visual_action_prompt,
                    seed=seed,
                    steps=50,
                )
                result = _submit_and_wait(
                    api=api,
                    payload=request,
                    candidate_dir=candidate_dir,
                )
                video = Path(str(result["video"]))
                probe = _validate_media_contract(
                    video,
                    challenge=challenge,
                    generation=generation,
                    label=f"candidate {case_id}/{seed}",
                )
                record = {
                    "case_id": case_id,
                    "candidate_index": candidate_index,
                    "harness_version": args.harness_version,
                    "strategy": case["strategy"],
                    "recovery_policy": case.get("recovery_policy"),
                    "challenge_sha256": challenge["challenge_sha256"],
                    "task_plan_sha256": case["task_plan_sha256"],
                    "harness_sha256": case["harness_sha256"],
                    "prompt_sha256": __import__("hashlib").sha256(
                        str(case["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "seed": seed,
                    "inference_steps": 50,
                    **result,
                    "video": str(video),
                    "video_sha256": _sha256(video),
                    "video_probe": probe,
                    "evaluation_status": "UNKNOWN",
                }
                _write_json(candidate_dir / "tri-evolve-record.json", record)
                metadata["completed_candidates"].append(record)
                _write_json(output / "metadata.json", metadata)
        metadata["status"] = "PARTIAL"
        metadata["decision"] = "GENERATION_COMPLETE_PENDING_EVALUATION"
        metadata["completed_at"] = _utc_now()
        metadata["claim_boundary"] = spec["claim_boundary"]
        _write_json(output / "metadata.json", metadata)
        print(
            json.dumps(
                {
                    "status": metadata["status"],
                    "decision": metadata["decision"],
                    "output_dir": str(output),
                    "harness_version": args.harness_version,
                    "challenge_sha256": challenge["challenge_sha256"],
                    "completed_candidates": len(metadata["completed_candidates"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        metadata["status"] = "PARTIAL"
        metadata["decision"] = "RUN_FAILED"
        metadata["error"] = f"{type(error).__name__}: {error}"
        metadata["failed_at"] = _utc_now()
        _write_json(output / "metadata.json", metadata)
        raise
    finally:
        if server is not None and server.poll() is None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if log_stream is not None:
            log_stream.close()
        for stream in leases:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
