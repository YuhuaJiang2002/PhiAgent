from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_multistem_cli_keeps_unscaled_proposal_partial(tmp_path: Path) -> None:
    frames, stems, nodes = 16, 2, 5
    observations = np.zeros((frames, stems, nodes, 3), dtype=np.float64)
    observations[:, 0, :, 1] = -0.1
    observations[:, 1, :, 1] = 0.1
    observations[:, :, :, 2] = np.linspace(0.0, 0.4, nodes)[None, None, :]
    observations[:, :, 1:, 0] = np.linspace(0.0, 0.02, frames)[:, None, None]
    confidence = np.ones((frames, stems, nodes), dtype=np.float64)
    observations[5:9, 0] = np.nan
    confidence[5:9, 0] = 0.0
    proposal = tmp_path / "proposal.npz"
    np.savez_compressed(
        proposal,
        centerline_proposals=observations,
        confidence=confidence,
        instance_ids=np.asarray(("stem-a", "stem-b")),
        source_frame_indices=np.arange(frames),
        source_video_sha256=np.asarray("a" * 64),
        coordinate_frame=np.asarray("world:vdpm_relative"),
        timeline=np.asarray("frame:source_video"),
        fps=np.asarray(24.0),
        geometry_evidence=np.asarray("foundation_model_estimate"),
    )
    instance_spec = tmp_path / "instances.json"
    instance_spec.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "instance_id": "stem-a",
                        "root_node": 0,
                        "root_mode": "fixed",
                    },
                    {
                        "instance_id": "stem-b",
                        "root_node": 0,
                        "root_mode": "fixed",
                    },
                ]
            }
        )
    )
    output = tmp_path / "output"
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/optimize_multistem_flower_state.py"),
            "--proposal",
            str(proposal),
            "--instance-spec",
            str(instance_spec),
            "--output-dir",
            str(output),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["optimization"]["structural_passed"] is True
    assert report["optimization"]["promotion_eligible"] is False
    assert report["metric_binding"]["verified"] is False
    assert report["honest_status"] == "PARTIAL"
