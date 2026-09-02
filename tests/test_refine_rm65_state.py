from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "refine_rm65_state_from_candidates",
    SCRIPTS / "refine_rm65_state_from_candidates.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _six_joint_model() -> mujoco.MjModel:
    bodies = []
    for index in range(1, 7):
        joint_range = "-6.28 6.28" if index == 6 else "-3.106 3.106"
        bodies.append(
            f'<body name="left_link_{index}" pos="0 0 0.05">'
            f'<joint name="left_joint_{index}" type="hinge" axis="0 0 1" '
            f'range="{joint_range}"/>'
            '<geom type="sphere" size="0.01" mass="0.01"/>'
        )
    closing = "</body>" * 6
    xml = (
        '<mujoco><compiler angle="radian"/><worldbody>'
        + "".join(bodies)
        + '<site name="left_eef" pos="0 0 0.05"/>'
        + closing
        + "</worldbody></mujoco>"
    )
    return mujoco.MjModel.from_xml_string(xml)


def test_periodic_joint_is_wrapped_to_nearest_reviewed_branch() -> None:
    model = _six_joint_model()
    q = np.zeros((2, 6), dtype=np.float64)
    q[:, 5] = (7.0, 7.1)
    reference = np.zeros_like(q)
    reference[:, 5] = (0.7, 0.8)

    canonical, manifest = MODULE.canonicalize_periodic_joints(
        model, q, "left", reference
    )

    np.testing.assert_allclose(canonical[:, 5], q[:, 5] - 2.0 * np.pi)
    assert manifest["periodic_joint_indices_1based"] == [6]
    assert manifest["canonicalized_frames_by_joint_1based"] == {"6": 2}
    assert manifest["selection_prior"] == "reviewed_joint_branch"


def test_narrow_range_joint_is_not_treated_as_periodic() -> None:
    model = _six_joint_model()
    q = np.zeros((1, 6), dtype=np.float64)
    q[0, 0] = 3.2

    canonical, manifest = MODULE.canonicalize_periodic_joints(
        model, q, "left", None
    )

    assert canonical[0, 0] == pytest.approx(3.2)
    assert manifest["periodic_joint_indices_1based"] == [6]


def test_triangular_smoothing_uses_edge_padding() -> None:
    values = np.asarray((0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float64)

    smoothed, kernel = MODULE.triangular_smooth(values, 5)

    np.testing.assert_allclose(kernel, (1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9))
    np.testing.assert_allclose(smoothed, (1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9))


def test_triangular_smoothing_rejects_even_window() -> None:
    with pytest.raises(ValueError, match="positive odd integer"):
        MODULE.triangular_smooth(np.zeros(3), 4)
