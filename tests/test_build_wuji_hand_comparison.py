from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_wuji_hand_comparison.py"


def _load_module():
    spec = spec_from_file_location("build_wuji_hand_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_detection_mask_distinguishes_upstream_holds():
    module = _load_module()
    first = object()
    second = object()
    assert module.direct_detection_mask([first, first, second, second]) == [
        True,
        False,
        True,
        False,
    ]


def test_sha256_file(tmp_path):
    module = _load_module()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"PhiAgent Wuji\n")
    assert module.sha256_file(payload) == (
        "9f7d86d4d7ebbcc31fc141cd07318b0459032e5161f3652b5476570c5d51ba05"
    )


def test_urdf_velocity_limits_follow_requested_joint_order(tmp_path):
    module = _load_module()
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        '<robot name="hand">'
        '<joint name="joint_b" type="revolute"><limit velocity="2.0"/></joint>'
        '<joint name="joint_a" type="revolute"><limit velocity="1.0"/></joint>'
        "</robot>"
    )
    assert module.urdf_velocity_limits(urdf, ["joint_a", "joint_b"]) == [1.0, 2.0]
