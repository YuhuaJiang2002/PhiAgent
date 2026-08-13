from __future__ import annotations

from scripts.build_full_flower_shadow_temporal_delivery import (
    detect_unsupported_transitions,
    repair_transition_neighborhoods,
)


def test_detect_unsupported_transitions_uses_source_and_local_evidence() -> None:
    np = __import__("numpy")
    candidate = np.ones(20, dtype=np.float64)
    source = np.ones(20, dtype=np.float64)
    candidate[5] = 8.0
    candidate[12] = 7.0
    source[12] = 6.5

    assert detect_unsupported_transitions(
        np,
        candidate,
        source,
        minimum_local_ratio=2.0,
        minimum_source_excess=4.0,
        local_radius=3,
    ) == [6]


def test_local_repair_preserves_flowers_phase_contact_and_background() -> None:
    cv2 = __import__("cv2")
    np = __import__("numpy")
    frames = [np.full((12, 16, 3), index * 10, dtype=np.uint8) for index in range(7)]
    frames[3][:] = 180
    robot = np.zeros((7, 12, 16), dtype=bool)
    robot[:, 3:10, 4:13] = True
    flowers = np.zeros_like(robot)
    flowers[:, 7:9, 7:9] = True
    safety = np.zeros((12, 16), dtype=bool)
    safety[2:11, 3:14] = True
    phase = {3: flowers[3].copy()}
    phase[3][4:6, 6:8] = True

    repaired, records, touched = repair_transition_neighborhoods(
        cv2,
        np,
        frames,
        [3],
        robot,
        flowers,
        safety,
        phase,
        radius=2,
        dilation_pixels=1,
        feather_sigma=1.0,
        mode="crossfade",
    )

    assert records[0]["transition_frame"] == 3
    assert touched.any()
    for index in range(7):
        assert np.array_equal(repaired[index][~safety], frames[index][~safety])
        assert np.array_equal(repaired[index][flowers[index]], frames[index][flowers[index]])
    assert np.array_equal(repaired[3][phase[3]], frames[3][phase[3]])
