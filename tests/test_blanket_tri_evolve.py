from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from phiagent.harness.blanket_tri_evolve import (
    BlanketDifficulty,
    build_failure_repair_directive,
    compare_difficulty,
    evaluate_hard_gates,
    physical_promotion_decision,
)
from scripts.bind_exact_first_frame import bind_exact_first_frame
from scripts.prepare_figure_two_robot_blanket_tri_evolve_campaign import main as prepare


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _difficulty(environment_id: str, level: str, extra: int) -> BlanketDifficulty:
    return BlanketDifficulty(
        environment_id=environment_id,
        level=level,
        coordinate_frame="camera:figure_blanket_fold_1024x768_pixels",
        obstacle_count=extra,
        quilt_rotation_degrees=5.0 * extra,
        off_table_overhang_fraction=0.05 * extra,
        self_occlusion_fraction=0.02 + 0.04 * extra,
        required_action_stages=4 + extra,
        required_regrasps=1 + extra,
        terminal_transport_fraction=0.1 * extra,
    )


def test_difficulty_challenger_is_monotone_and_hash_bound() -> None:
    result = compare_difficulty(
        _difficulty("blanket-flat-e0", "E0", 0),
        _difficulty("blanket-obstacle-e2", "E2", 2),
    )
    assert result["passed"] is True
    assert len(result["increased_dimensions"]) >= 4
    assert not result["regressed_dimensions"]
    assert len(result["challenger_sha256"]) == 64


def test_missing_hard_gate_fails_closed_and_cannot_be_overridden() -> None:
    result = evaluate_hard_gates(
        ("exact_first_frame", "contact_before_motion"),
        {"exact_first_frame": "PASS"},
    )
    assert result["passed"] is False
    assert result["reason"] == "hard_gate_failed"
    assert result["failed_or_unavailable_gate_ids"] == ["contact_before_motion"]
    assert result["aggregate_override_allowed"] is False


def test_failure_repair_retains_exact_ids_and_physical_boundary() -> None:
    directive = build_failure_repair_directive(
        ("exact_first_frame", "contact_before_motion")
    )
    assert "exact_first_frame" in directive
    assert "contact_before_motion" in directive
    assert "Preserve every frozen threshold" in directive
    assert physical_promotion_decision(
        proposal_ready=True,
        physical_gate_states={"absolute_scale_verified": False},
        independent_source_groups=1,
    ) == {"promote": False, "reason": "proposal_not_physical_calibration"}


def test_prepare_writes_new_hash_bound_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "campaign"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(output)])
    assert prepare() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    spec = json.loads((output / "inputs/generation-spec.json").read_text())
    assert manifest["candidate_budget"] == 4
    assert manifest["difficulty_comparison"]["passed"] is True
    assert [case["label"] for case in spec["cases"]] == [
        "h0-direct-obstacle-fold",
        "h1-canonicalize-verify-fold",
        "h2-contact-repair-transport-lock",
    ]
    assert spec["generation"]["boundary_binding"]["thresholds_unchanged"] is True


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for media binding",
)
def test_exact_boundary_binding_preserves_raw_and_media_contract(tmp_path: Path) -> None:
    raw = tmp_path / "raw.mp4"
    output = tmp_path / "bound.mp4"
    frame = (
        PROJECT_ROOT
        / "demo/showcase/source-scenes/"
        "figure-two-robot-blanket-fold-tri-evolve-e2-v1-1024x768.png"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1024x768:r=24:d=8",
            "-frames:v",
            "192",
            "-pix_fmt",
            "yuv420p",
            str(raw),
        ],
        check=True,
    )
    record = bind_exact_first_frame(
        raw_video=raw,
        exact_frame=frame,
        output_video=output,
        expected_frames=192,
        fps=24,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(output),
            ],
            text=True,
        )
    )["streams"][0]
    assert raw.is_file()
    assert record["raw_video_sha256"]
    assert record["thresholds_unchanged"] is True
    assert probe == {
        "width": 1024,
        "height": 768,
        "r_frame_rate": "24/1",
        "nb_read_frames": "192",
    }
