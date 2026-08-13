from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_bwm_factory_batch.py"
    spec = importlib.util.spec_from_file_location("run_bwm_factory_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_shards_balances_contiguous_ranges() -> None:
    shards = _module().plan_shards(total=10, physical_gpus=[0, 2, 3], start_index=5)
    assert shards == (
        {"worker_index": 0, "physical_gpu": 0, "start_index": 5, "samples": 4},
        {"worker_index": 1, "physical_gpu": 2, "start_index": 9, "samples": 3},
        {"worker_index": 2, "physical_gpu": 3, "start_index": 12, "samples": 3},
    )


def test_plan_shards_does_not_create_empty_workers() -> None:
    shards = _module().plan_shards(total=2, physical_gpus=[0, 2, 3], start_index=0)
    assert [shard["physical_gpu"] for shard in shards] == [0, 2]
    assert [shard["samples"] for shard in shards] == [1, 1]


@pytest.mark.parametrize(
    ("total", "gpus", "start"),
    ((0, [0], 0), (1, [], 0), (1, [0, 0], 0), (1, [0], -1)),
)
def test_plan_shards_rejects_invalid_inputs(total: int, gpus: list[int], start: int) -> None:
    with pytest.raises(ValueError):
        _module().plan_shards(total=total, physical_gpus=gpus, start_index=start)
