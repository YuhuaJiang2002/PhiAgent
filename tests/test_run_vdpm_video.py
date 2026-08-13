from __future__ import annotations

from pathlib import Path

from scripts.run_vdpm_video import _window_starts


def test_vdpm_window_starts_preserve_overlap_and_tail() -> None:
    assert _window_starts(10, 6, 2) == [0, 4, 8]


def test_vdpm_runner_requires_local_hashable_checkpoint() -> None:
    source = Path("scripts/run_vdpm_video.py").read_text()

    assert 'parser.add_argument("--checkpoint"' in source
    assert "torch.load(" in source
    assert "load_state_dict_from_url" not in source
    assert '"checkpoint_sha256": _sha256(checkpoint)' in source
