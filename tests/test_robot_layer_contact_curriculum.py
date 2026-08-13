from __future__ import annotations

from scripts.build_robot_layer_contact_curriculum import select_curriculum_starts


def test_curriculum_holds_all_validation_frames_after_late_boundary() -> None:
    limits = {
        "palette_surprisal": 2.0,
        "high_chroma_fraction": 0.2,
        "skin_like_fraction": 0.2,
        "spatial_chroma_tv": 4.0,
        "replacement_coverage": 0.5,
        "arm_replacement_coverage": 0.5,
        "hand_replacement_coverage": 0.5,
        "grid_topology_coverage": 0.5,
    }
    rows = [
        {
            "frame": frame,
            "palette_surprisal": 1.0,
            "high_chroma_fraction": 0.1,
            "skin_like_fraction": 0.1,
            "spatial_chroma_tv": 2.0,
            "replacement_coverage": 0.9,
            "arm_replacement_coverage": 0.9,
            "hand_replacement_coverage": 0.9,
            "grid_topology_coverage": 0.9,
            "contact_required": frame % 5 == 0,
            "contact_pass": True,
        }
        for frame in range(80)
    ]

    train, validation = select_curriculum_starts(
        rows,
        limits,
        train_clips=3,
        validation_clips=2,
        frames=5,
        source_frame_step=2,
        late_start=40,
        minimum_pass_fraction=1.0,
    )

    assert all(max(item["indices"]) < 40 for item in train)
    assert all(min(item["indices"]) >= 40 for item in validation)
