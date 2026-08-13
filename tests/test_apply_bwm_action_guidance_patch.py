from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from phiagent.acwm.adapters import BWM_REPOSITORY_COMMIT


def test_action_guidance_patch_applies_and_is_idempotent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "external" / "boundless-world-model"
    repository = tmp_path / "bwm"
    for relative in (
        Path("scripts/infer.py"),
        Path("wan_video_action/pipelines/wan_video_action.py"),
    ):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=source,
            check=True,
            capture_output=True,
        ).stdout
        target.write_bytes(content)
    (repository / ".phiagent-source-revision").write_text(
        BWM_REPOSITORY_COMMIT + "\n"
    )
    command = [
        sys.executable,
        str(root / "scripts" / "apply_bwm_action_guidance_patch.py"),
        "--repository",
        str(repository),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
    repeated = subprocess.run(command, check=True, capture_output=True, text=True)

    assert "epsilon_hold + scale" in repeated.stdout
    infer = (repository / "scripts/infer.py").read_text()
    pipeline = (
        repository / "wan_video_action/pipelines/wan_video_action.py"
    ).read_text()
    assert "_build_hold_action_baseline" in infer
    assert "target_vs_hold" in infer
    assert "noise_pred_baseline + float(cfg_scale)" in pipeline
    assert "if float(cfg_scale) == 1.0" in pipeline
