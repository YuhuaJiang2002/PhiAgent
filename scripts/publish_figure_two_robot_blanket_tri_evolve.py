#!/usr/bin/env python3
"""Publish one diagnosis-only E2 blanket video without overriding hard gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly four times",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "publication command failed")


def _diagnostic_key(report: dict[str, Any]) -> tuple[int, float, float]:
    failed = len(report["failed_automatic_hard_gate_ids"])
    metrics = report["automatic_metrics"]
    background = float(metrics["minimum_background_patch_ssim"])
    terminal = float(
        metrics["terminal_flow"][
            "maximum_pairwise_vector_magnitude_rms_pixels"
        ]
    )
    return (failed, -background, terminal)


def main() -> int:
    args = _parser().parse_args()
    campaign = args.campaign_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse publication directory: {output}")
    evaluation_dirs = [item.expanduser().resolve() for item in args.evaluation_dir]
    if len(evaluation_dirs) != 4:
        raise ValueError("publication requires exactly four evaluation directories")
    reports = [_read_json(item / "evaluation.json") for item in evaluation_dirs]
    seeds = [int(item["candidate"]["seed"]) for item in reports]
    if len(set(seeds)) != 4:
        raise ValueError("publication requires four unique candidate seeds")
    for report in reports:
        candidate = Path(report["candidate"]["path"])
        if not candidate.is_file() or _sha256(candidate) != report["candidate"]["sha256"]:
            raise ValueError("candidate path or hash differs from its evaluation")
        if report.get("overall_visual_acceptance") is not False:
            raise ValueError("publication is diagnosis-only before native review")
        if report.get("physical_promotion", {}).get("promote") is not False:
            raise ValueError("generated RGB cannot be published as physical promotion")

    output.mkdir(parents=True)
    evaluations_out = output / "evaluations"
    evaluations_out.mkdir()
    candidate_rows = []
    for evaluation_dir, report in sorted(
        zip(evaluation_dirs, reports, strict=True),
        key=lambda item: int(item[1]["candidate"]["seed"]),
    ):
        seed = int(report["candidate"]["seed"])
        label = str(report["candidate"]["case_id"])
        destination = evaluations_out / f"{label}-seed-{seed}.json"
        shutil.copy2(evaluation_dir / "evaluation.json", destination)
        metrics = report["automatic_metrics"]
        candidate_rows.append(
            {
                "case_id": label,
                "strategy": report["candidate"]["strategy"],
                "seed": seed,
                "authoritative_video_sha256": report["candidate"]["sha256"],
                "raw_video_sha256": report["candidate"]["raw_sha256"],
                "evaluation_path": str(destination.relative_to(output)),
                "evaluation_sha256": _sha256(destination),
                "automatic_decision": report["automatic_decision"],
                "failed_automatic_hard_gate_ids": report[
                    "failed_automatic_hard_gate_ids"
                ],
                "first_frame_ssim": metrics["first_frame_ssim"],
                "minimum_background_patch_ssim": metrics[
                    "minimum_background_patch_ssim"
                ],
                "terminal_flow_rms_pixels": metrics["terminal_flow"][
                    "maximum_pairwise_vector_magnitude_rms_pixels"
                ],
                "overall_visual_acceptance": False,
                "physical_promotion": False,
            }
        )

    selected = min(reports, key=_diagnostic_key)
    selected_seed = int(selected["candidate"]["seed"])
    selected_eval_dir = evaluation_dirs[reports.index(selected)]
    authoritative = Path(selected["candidate"]["path"])
    presentation = output / "video.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(authoritative),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(presentation),
        ]
    )
    poster = output / "poster.jpg"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(presentation),
            "-vf",
            "select=eq(n\\,96)",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(poster),
        ]
    )
    shutil.copy2(
        selected_eval_dir / "native-review-contact-sheet.jpg",
        output / "contact-sheet.jpg",
    )
    shutil.copy2(
        selected_eval_dir / "native-review-template.json",
        output / "native-review-template.json",
    )
    shutil.copy2(
        selected_eval_dir / "evaluation.json",
        output / "evaluation.json",
    )

    config = _read_json(campaign / "inputs/campaign-config.json")
    campaign_manifest = _read_json(campaign / "manifest.json")
    accepted = [
        item for item in reports if item.get("overall_visual_acceptance") is True
    ]
    automatic_rejections = [
        item for item in reports if item["automatic_decision"] == "REJECT"
    ]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": config["campaign_id"],
        "status": "PARTIAL",
        "strict_result": {
            "candidate_count": 4,
            "automatic_rejection_count": len(automatic_rejections),
            "visual_acceptance_count": len(accepted),
            "physical_promotion_count": 0,
            "decision": (
                "REJECT_ALL_AUTOMATIC"
                if len(automatic_rejections) == 4
                else "NO_ACCEPTED_CANDIDATE_PENDING_NATIVE_REVIEW"
            ),
            "aggregate_override_allowed": False,
        },
        "difficulty_comparison": campaign_manifest["difficulty_comparison"],
        "candidates": candidate_rows,
        "presentation": {
            "selection_scope": "retrospective diagnosis only",
            "selection_seed": selected_seed,
            "selection_case_id": selected["candidate"]["case_id"],
            "selection_reason": (
                "Fewest automatic failures, then highest frozen-background SSIM, then "
                "lowest terminal flow. This ranking cannot create acceptance."
            ),
            "authoritative_video_sha256": selected["candidate"]["sha256"],
            "presentation_video_path": "video.mp4",
            "presentation_video_sha256": _sha256(presentation),
            "poster_sha256": _sha256(poster),
            "contact_sheet_sha256": _sha256(output / "contact-sheet.jpg"),
        },
        "campaign_hashes": campaign_manifest["hashes"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
