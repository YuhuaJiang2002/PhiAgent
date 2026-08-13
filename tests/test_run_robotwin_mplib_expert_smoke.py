from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_robotwin_mplib_expert_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("run_robotwin_mplib_expert_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mplib_fallback_patch_is_explicit_and_gripper_only(tmp_path: Path) -> None:
    module = _module()
    planner = tmp_path / "envs" / "robot" / "planner.py"
    planner.parent.mkdir(parents=True)
    planner.write_text(
        "prefix\n"
        "    traceback.print_exc()\n\n\n"
        "# ********************** MplibPlanner\n"
        "suffix\n"
    )

    hashes = module.patch_planner_for_mplib(tmp_path)
    rendered = planner.read_text()

    assert hashes["before_sha256"] != hashes["after_sha256"]
    assert module.FALLBACK_MARKER in rendered
    assert "def plan_grippers" in rendered
