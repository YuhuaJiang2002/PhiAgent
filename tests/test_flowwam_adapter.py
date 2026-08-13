from __future__ import annotations

import json
import sys
from pathlib import Path

from phiagent.acwm.adapters import (
    BWM_BASE_MODEL_REVISION,
    FLOWWAM_MODEL_BYTES,
    FLOWWAM_MODEL_REVISION,
    FLOWWAM_MODEL_SHA256,
    FLOWWAM_REPOSITORY_COMMIT,
    FLOWWAM_TOKENIZER_REVISION,
    FlowWAMConfig,
    FlowWAMRenderer,
)
from phiagent.acwm.schema import ACWMActionCondition, ACWMCase, ActionRepresentation


def _file(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _case(tmp_path: Path, *, with_assets: bool = True) -> ACWMCase:
    condition = ACWMActionCondition(
        label="flow-action",
        instruction="Render the robot-only optical flow.",
        timeline="frame-aligned robot flow",
        representation=ActionRepresentation.ROBOT_FLOW,
        coordinate_frame="camera:head_rgb",
        timestamps_s=(0.0, 1 / 24),
        channels=("flow_mean_u_px", "flow_mean_v_px"),
        values=((0.0, 0.0), (1.0, 0.0)),
        visual_condition=_file(tmp_path / "robot-flow.mp4"),
    )
    assets = (
        (
            ("robot_urdf", _file(tmp_path / "robot.urdf")),
            ("camera_calibration", _file(tmp_path / "camera.json")),
            ("flow_provenance", _file(tmp_path / "flow-provenance.json")),
        )
        if with_assets
        else ()
    )
    return ACWMCase(
        case_id="flow-action",
        first_frame=_file(tmp_path / "first.png"),
        source_video=_file(tmp_path / "source.mp4"),
        action=condition,
        prompt="Render one robot following the supplied flow.",
        auxiliary_inputs=assets,
    )


def test_flowwam_requires_dense_flow_and_geometry_provenance(tmp_path: Path) -> None:
    renderer = FlowWAMRenderer(
        FlowWAMConfig(
            repository=tmp_path / "repo",
            base_model_root=tmp_path / "base",
            checkpoint_path=tmp_path / "flowwam.safetensors",
        )
    )

    assert renderer.supports(_case(tmp_path / "complete")).supported
    report = renderer.supports(_case(tmp_path / "missing", with_assets=False))
    assert not report.supported
    assert set(report.reasons) == {
        "requires robot_urdf",
        "requires camera_calibration",
        "requires flow_provenance",
    }


def test_flowwam_preflight_pins_source_and_checkpoint(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    python = repository / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable))
    (repository / ".phiagent-source-revision").write_text(
        FLOWWAM_REPOSITORY_COMMIT + "\n"
    )
    base = tmp_path / "base"
    base.mkdir()
    (base / ".phiagent-flowwam-base-revisions.json").write_text(
        json.dumps(
            {
                "wan22_ti2v_revision": (
                    BWM_BASE_MODEL_REVISION
                ),
                "wan21_tokenizer_revision": FLOWWAM_TOKENIZER_REVISION,
            }
        )
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "flowwam_worldarena_stage1.safetensors"
    with checkpoint.open("wb") as handle:
        handle.truncate(FLOWWAM_MODEL_BYTES)
    (checkpoint_dir / ".phiagent-model-revision").write_text(
        FLOWWAM_MODEL_REVISION + "\n"
    )
    (checkpoint_dir / ".phiagent-model-verification.json").write_text(
        json.dumps(
            {
                "bytes": FLOWWAM_MODEL_BYTES,
                "sha256": FLOWWAM_MODEL_SHA256,
            }
        )
    )
    renderer = FlowWAMRenderer(
        FlowWAMConfig(
            repository=repository,
            base_model_root=base,
            checkpoint_path=checkpoint,
        )
    )

    report = renderer.preflight(select_cuda_device=False)

    assert report["repository_commit"] == FLOWWAM_REPOSITORY_COMMIT
    assert report["model"]["revision"] == FLOWWAM_MODEL_REVISION
    assert report["model"]["checkpoint_sha256"] == FLOWWAM_MODEL_SHA256
