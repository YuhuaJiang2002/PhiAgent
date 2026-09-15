#!/usr/bin/env python3
"""Verify the lightweight final source/evidence release and optional weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "releases" / "step-016000"
CHUNK_BYTES = 8 * 1024 * 1024


class VerificationError(RuntimeError):
    """Raised when a release contract or artifact does not match."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path}")
    return value


def parse_payload_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise VerificationError(f"cannot read payload manifest {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise VerificationError(f"malformed manifest line {line_number}") from error
        candidate = PurePosixPath(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or candidate.is_absolute()
            or not candidate.parts
            or ".." in candidate.parts
            or relative in entries
        ):
            raise VerificationError(f"unsafe or invalid manifest line {line_number}")
        entries[relative] = digest
    return entries


def verify_release(release: Path = DEFAULT_RELEASE) -> dict[str, Any]:
    release = release.resolve()
    contract_path = release / "release.json"
    contract = load_json_object(contract_path)
    if contract.get("schema") != 1:
        raise VerificationError("unsupported release schema")

    payload = contract.get("payload")
    if not isinstance(payload, dict):
        raise VerificationError("release payload contract is missing")
    manifest_path = release / str(payload.get("manifest", ""))
    if not manifest_path.is_file():
        raise VerificationError(f"payload manifest is missing: {manifest_path}")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != payload.get("manifest_sha256"):
        raise VerificationError("payload manifest SHA-256 mismatch")

    entries = parse_payload_manifest(manifest_path)
    if len(entries) != payload.get("file_count"):
        raise VerificationError("payload file count differs from release.json")
    declared_directories = []
    for key in (
        "source_overlay",
        "integration_overlay",
        "evidence",
        "integration_evidence",
    ):
        section = contract.get(key)
        if not isinstance(section, dict) or not isinstance(section.get("directory"), str):
            raise VerificationError(f"release {key} contract is missing")
        declared_directories.append(section["directory"])
        count = sum(path.startswith(section["directory"] + "/") for path in entries)
        if count != section.get("file_count"):
            raise VerificationError(f"{key} file count differs from release.json")

    actual_paths: set[str] = set()
    for directory in declared_directories:
        root = release / directory
        if not root.is_dir():
            raise VerificationError(f"payload directory is missing: {root}")
        for path in root.rglob("*"):
            relative_path = path.relative_to(release)
            if "__pycache__" in relative_path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise VerificationError(f"payload symlinks are forbidden: {path}")
            if path.is_file():
                actual_paths.add(relative_path.as_posix())
    if actual_paths != set(entries):
        missing = sorted(set(entries) - actual_paths)
        extra = sorted(actual_paths - set(entries))
        raise VerificationError(f"payload inventory mismatch; missing={missing}, extra={extra}")

    for relative, expected in entries.items():
        path = release / relative
        if sha256_file(path) != expected:
            raise VerificationError(f"payload SHA-256 mismatch: {relative}")

    checkpoint = contract.get("checkpoint")
    snapshot = contract.get("inference_snapshot")
    if not isinstance(checkpoint, dict) or not isinstance(snapshot, dict):
        raise VerificationError("checkpoint contracts are missing")
    sidecar = load_json_object(release / "evidence/basic_stage1_24gpu/step-016000.json")
    snapshot_sidecar = load_json_object(
        release
        / "evidence/validation/checkpoints/"
        / f"{snapshot.get('sha256')}.json"
    )
    metrics = load_json_object(release / "evidence/validation/metrics/latest_validation.json")
    source = load_json_object(
        release / "evidence/migration/20260912-shared/source-state/h20-4/source.json"
    )
    integration_root = (
        release / "integration_evidence/taco-20230927-032-camera-only"
    )
    integration_review = load_json_object(integration_root / "review.json")
    integration_preflight = load_json_object(integration_root / "preflight.json")
    integration_inference = load_json_object(integration_root / "inference.json")
    integration_provenance = load_json_object(
        integration_root / "generation-provenance.json"
    )
    integration_evaluation = load_json_object(
        integration_root / "heldout-evaluation.json"
    )
    integration_contract = contract.get("camera_only_integration_check")
    if not isinstance(integration_contract, dict):
        raise VerificationError("camera-only integration contract is missing")
    cross_checks = (
        sidecar.get("checkpoint_sha256") == checkpoint.get("sha256"),
        sidecar.get("checkpoint_bytes") == checkpoint.get("bytes"),
        sidecar.get("step") == checkpoint.get("step"),
        snapshot_sidecar.get("snapshot_sha256") == snapshot.get("sha256"),
        snapshot_sidecar.get("source_sha256") == checkpoint.get("sha256"),
        metrics.get("checkpoint_sha256") == snapshot.get("sha256"),
        metrics.get("checkpoint_source_sha256") == checkpoint.get("sha256"),
        source.get("commit") == contract.get("upstream", {}).get("commit"),
        source.get("archive_sha256")
        == contract.get("upstream", {}).get("source_archive_sha256"),
        integration_preflight.get("status") == "preflight_passed",
        integration_preflight.get("checkpoint_step") == checkpoint.get("step"),
        integration_inference.get("status") == "generated",
        integration_inference.get("checkpoint_sha256") == checkpoint.get("sha256"),
        integration_inference.get("target_reference_used_during_generation") is False,
        integration_inference.get("has_target_reference") is False,
        integration_inference.get("k") == 4,
        integration_inference.get("context_policy") == "repeat_single_source",
        integration_provenance.get("status") == "generated",
        integration_provenance.get("actual_checkpoint_sha256")
        == checkpoint.get("sha256"),
        integration_provenance.get("output_sha256")
        == integration_review.get("generated_sha256"),
        integration_evaluation.get("target_reference_used_during_generation") is False,
        integration_evaluation.get("heldout_target_absent_from_generation_inputs")
        is True,
        integration_evaluation.get("generated_sha256")
        == integration_review.get("generated_sha256"),
        integration_evaluation.get("heldout_target_sha256")
        == integration_review.get("heldout_target_sha256"),
        integration_evaluation.get("comparison_sha256")
        == integration_review.get("comparison_sha256"),
        integration_review.get("status") == integration_contract.get("status"),
        integration_review.get("reviewed_frames")
        == integration_contract.get("reviewed_frames"),
        sha256_file(release / "integration_overlay/phiagent/rendering/plenoptic.py")
        == integration_provenance.get("adapter_module_sha256"),
        sha256_file(release / "integration_overlay/scripts/run_plenoptic_phiagent.py")
        == integration_provenance.get("launcher_sha256"),
        sha256_file(release / "integration_overlay/scripts/evaluate_plenoptic_heldout.py")
        == integration_evaluation.get("evaluator_sha256"),
    )
    if not all(cross_checks):
        raise VerificationError("release evidence does not cross-bind to release.json")
    return {
        "status": "verified",
        "release_id": contract.get("release_id"),
        "payload_files": len(entries),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "checkpoint_checked": False,
    }


def verify_checkpoint(path: Path, contract: dict[str, Any]) -> None:
    checkpoint = contract.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise VerificationError("checkpoint contract is missing")
    if not path.is_file():
        raise VerificationError(f"checkpoint is missing: {path}")
    if path.stat().st_size != checkpoint.get("bytes"):
        raise VerificationError(
            f"checkpoint size mismatch: expected {checkpoint.get('bytes')}, "
            f"found {path.stat().st_size}"
        )
    actual = sha256_file(path)
    if actual != checkpoint.get("sha256"):
        raise VerificationError(f"checkpoint SHA-256 mismatch: {actual}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--json", action="store_true", dest="as_json")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        report = verify_release(args.release)
        if args.checkpoint is not None:
            contract = load_json_object(args.release.resolve() / "release.json")
            verify_checkpoint(args.checkpoint.resolve(), contract)
            report["checkpoint_checked"] = True
    except VerificationError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(report, sort_keys=True))
    else:
        suffix = " and checkpoint" if report["checkpoint_checked"] else ""
        print(
            f"verified {report['release_id']}: {report['payload_files']} payload files{suffix}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
