from __future__ import annotations

import numpy as np

from phiagent.agent.flower_task_adaptation import HandPhase
from scripts.compile_flower_task_contract import (
    _build_phases,
    _contact_intervals,
    _plan_generation_jobs,
)


def test_contact_hysteresis_rejects_short_noise_and_merges_small_gap() -> None:
    values = np.full(50, 40.0, dtype=np.float32)
    values[4:7] = 2.0
    values[12:22] = 3.0
    values[25:35] = 4.0

    intervals = _contact_intervals(
        values,
        on_threshold=8.0,
        off_threshold=18.0,
        minimum_frames=6,
        merge_gap=5,
    )

    assert intervals == ((12, 35),)


def test_phase_compiler_covers_full_timeline_and_names_contact_flower() -> None:
    phases = _build_phases(
        80,
        ((20, 60),),
        approach_frames=8,
        grasp_frames=6,
        release_frames=6,
        retract_frames=8,
    )

    assert phases[0].start_frame == 0
    assert phases[-1].end_frame_exclusive == 80
    assert all(first.end_frame_exclusive == second.start_frame for first, second in zip(phases, phases[1:]))
    contact = [phase for phase in phases if phase.right_phase in {HandPhase.GRASP, HandPhase.MANIPULATE, HandPhase.RELEASE}]
    assert contact
    assert all(phase.right_flower_id == "active-stem-proxy-00" for phase in contact)
    assert all(phase.left_flower_id == "bouquet-main" for phase in phases)


def test_generation_jobs_cover_660_frames_without_frame_level_mixing() -> None:
    phases = _build_phases(
        660,
        ((236, 316),),
        approach_frames=12,
        grasp_frames=6,
        release_frames=6,
        retract_frames=12,
    )
    jobs = _plan_generation_jobs(phases, (), 660)
    covered = {
        frame
        for job in jobs
        for frame in range(int(job["start_frame"]), int(job["end_frame_exclusive"]))
    }

    assert covered == set(range(660))
    assert all(job["frame_level_candidate_mixing"] is False for job in jobs)
