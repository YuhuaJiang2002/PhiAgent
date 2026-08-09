from __future__ import annotations

import math
from pathlib import Path

import pytest

from phiagent.perception.geometry import rotation_matrix_to_quaternion
from phiagent.perception.object.foundation_pose import FoundationPoseOutputReader
from phiagent.physical_language.schema import FrameKind, FrameRef
from phiagent.retargeting.dex_retargeting import epl_landmarks
from tests.test_physical_language import _chunk


def test_rotation_matrix_conversion_validates_proper_rotation() -> None:
    quaternion = rotation_matrix_to_quaternion(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    assert quaternion == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    with pytest.raises(ValueError, match="orthonormal"):
        rotation_matrix_to_quaternion(((1, 0, 0), (0, 2, 0), (0, 0, 1)))


def test_foundationpose_output_is_imported_as_object_in_camera(tmp_path: Path) -> None:
    poses = tmp_path / "ob_in_cam"
    poses.mkdir()
    (poses / "000001.txt").write_text("1 0 0 0.1\n0 1 0 0.2\n0 0 1 0.3\n0 0 0 1\n")
    (poses / "000002.txt").write_text("1 0 0 0.2\n0 1 0 0.2\n0 0 1 0.3\n0 0 0 1\n")
    camera = FrameRef(FrameKind.CAMERA, "front")
    observations = FoundationPoseOutputReader().load(
        poses, (0.0, 0.1), "cube", camera, confidence=0.8
    )
    assert observations[0].pose.source_frame == FrameRef(FrameKind.OBJECT, "cube")
    assert observations[0].pose.target_frame == camera
    assert observations[1].pose.translation_m == (0.2, 0.2, 0.3)


def test_dex_adapter_exposes_only_epl_v01_landmarks() -> None:
    landmarks = epl_landmarks(_chunk())
    assert set(landmarks) == {0, 4, 8, 12, 16, 20}
