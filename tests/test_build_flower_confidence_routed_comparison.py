from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_flower_confidence_routed_comparison.py"
SPEC = importlib.util.spec_from_file_location("flower_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _accepted_manifest(candidate: Path, digest: str) -> dict:
    return {
        "status": "accepted",
        "outputs": {"video": str(candidate), "video_sha256": digest},
        "acceptance": {name: True for name in MODULE.REQUIRED_ACCEPTANCE_GATES},
    }


def test_route_preserves_exact_accepted_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.mp4"
    candidate.write_bytes(b"immutable video bytes")
    digest = MODULE._sha256(candidate)

    route = MODULE._route_candidate(_accepted_manifest(candidate, digest), candidate)

    assert route["accepted"] is True
    assert route["decision"] == "preserve_candidate_all_frames"
    assert route["failed_gates"] == []


def test_route_rejects_failed_gate_or_changed_bytes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.mp4"
    candidate.write_bytes(b"new bytes")
    manifest = _accepted_manifest(candidate, "0" * 64)
    manifest["acceptance"]["background_lock_passed"] = False

    route = MODULE._route_candidate(manifest, candidate)

    assert route["accepted"] is False
    assert route["decision"] == "reject_candidate"
    assert route["failed_gates"] == ["background_lock_passed"]
    assert len(route["reasons"]) == 2


def test_vertical_filter_has_two_labeled_rows() -> None:
    graph = MODULE._comparison_filter(672, 384)

    assert "REAL HUMAN INPUT" in graph
    assert "CONFIDENCE-ROUTED ROBOT" in graph
    assert "[v0][v1]vstack=inputs=2[out]" in graph
    assert "pad=672:384" in graph


def test_vertical_filter_can_match_unlabeled_reference_layout() -> None:
    graph = MODULE._comparison_filter(672, 384, labels=False)

    assert "drawtext" not in graph
    assert "drawbox" not in graph
    assert "[v0][v1]vstack=inputs=2[out]" in graph
