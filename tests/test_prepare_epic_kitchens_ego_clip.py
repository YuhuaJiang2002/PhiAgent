from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.prepare_epic_kitchens_ego_clip import intersecting_annotations


def test_intersecting_annotations_use_half_open_interval() -> None:
    rows = [
        {
            "video_id": "P03_28",
            "start_timestamp": "00:00:24.83",
            "stop_timestamp": "00:00:25.80",
            "narration": "pick up bottle",
        },
        {
            "video_id": "P03_28",
            "start_timestamp": "00:00:34.82",
            "stop_timestamp": "00:00:35.60",
            "narration": "place bottle",
        },
        {
            "video_id": "P03_29",
            "start_timestamp": "00:00:25.00",
            "stop_timestamp": "00:00:26.00",
            "narration": "wrong video",
        },
    ]

    selected = intersecting_annotations(
        rows, video_id="P03_28", start_s=24.83, end_s=34.83
    )

    assert [item["narration"] for item in selected] == [
        "pick up bottle",
        "place bottle",
    ]


def test_epic_kitchens_preparer_has_dataset_arguments() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_epic_kitchens_ego_clip.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--video-id" in completed.stdout
    assert "--license" in completed.stdout
