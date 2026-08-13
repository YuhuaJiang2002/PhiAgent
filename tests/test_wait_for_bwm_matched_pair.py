from __future__ import annotations

import pytest

from phiagent.rendering.wan_animate import GPUInfo
from scripts.wait_for_bwm_matched_pair import ready_gpu_indices


def test_ready_gpu_indices_requires_every_requested_gpu() -> None:
    gpus = (
        GPUInfo(1, "A800", 81920, 1000, 80920),
        GPUInfo(4, "A800", 81920, 30000, 51920),
    )

    assert ready_gpu_indices(gpus, (1,), 61000) is True
    assert ready_gpu_indices(gpus, (1, 4), 61000) is False
    with pytest.raises(ValueError):
        ready_gpu_indices(gpus, (1, 1), 61000)
