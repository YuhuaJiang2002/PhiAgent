from __future__ import annotations

import pytest

from scripts.build_long_real_scene_showcase import parse_ffprobe


def test_parse_ffprobe_normalizes_video_metadata() -> None:
    result = parse_ffprobe(
        {
            "streams": [
                {
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "24/1",
                    "nb_frames": "660",
                    "duration": "27.500000",
                }
            ]
        }
    )

    assert result == {
        "width": 1280,
        "height": 720,
        "frames": 660,
        "fps": 24.0,
        "duration_seconds": 27.5,
    }


def test_parse_ffprobe_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="denominator"):
        parse_ffprobe(
            {
                "streams": [
                    {
                        "width": 1,
                        "height": 1,
                        "avg_frame_rate": "0/0",
                        "nb_frames": "1",
                        "duration": "1",
                    }
                ]
            }
        )
