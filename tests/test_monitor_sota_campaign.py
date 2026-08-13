from __future__ import annotations

from scripts.monitor_sota_campaign import classify_gpu_lines


def test_monitor_strictly_reserves_any_allocated_gpu() -> None:
    result = classify_gpu_lines(
        [
            "0, GPU-free, A800, 81920, 3",
            "1, GPU-process, A800, 81920, 800",
            "2, GPU-memory, A800, 81920, 2048",
        ],
        ["GPU-process, 123, python, 794"],
    )

    assert [row["classification"] for row in result] == [
        "free",
        "reserved_or_busy",
        "reserved_or_busy",
    ]
