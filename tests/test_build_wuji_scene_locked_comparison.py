from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_wuji_scene_locked_comparison.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_wuji_scene_locked_comparison", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_detection_mask_distinguishes_held_object() -> None:
    module = load_module()
    first = object()
    second = object()
    assert module.direct_detection_mask([first, first, second, None, second]) == [
        True,
        False,
        True,
        False,
        False,
    ]


def test_sha256_file(tmp_path: Path) -> None:
    module = load_module()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"Wuji scene lock\n")
    assert (
        module.sha256_file(payload)
        == "efcf474315d33d14dc9f791b25c1a7bf9bbaa540eae328c997c521bbaab476a9"
    )


def test_urdf_velocity_limits_follow_requested_order(tmp_path: Path) -> None:
    module = load_module()
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        "<robot name='hand'>"
        "<joint name='b' type='revolute'><limit velocity='2.5'/></joint>"
        "<joint name='a' type='revolute'><limit velocity='1.5'/></joint>"
        "</robot>"
    )
    assert module.urdf_velocity_limits(urdf, ["a", "b"]) == [1.5, 2.5]


def test_stable_forearm_widths_are_thin_bounded_and_deterministic() -> None:
    module = load_module()
    assert tuple(round(value, 6) for value in module.stable_forearm_half_widths(100.0)) == (
        24.0,
        28.0,
    )
    assert module.stable_forearm_half_widths(10.0) == (21.0, 24.0)
    assert module.stable_forearm_half_widths(1000.0) == (30.0, 34.0)
