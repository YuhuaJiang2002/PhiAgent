from __future__ import annotations

import numpy as np

from phiagent.agent.flower_task_adaptation import HandPhase
from scripts.generate_flower_task_vace_dataset import (
    canonical_contact_phases,
    causal_flower_grip_trajectory,
)


def test_canonical_schedule_contains_every_required_action_in_order() -> None:
    phases = canonical_contact_phases(17)

    assert len(phases) == 17
    assert list(dict.fromkeys(phases)) == [
        HandPhase.APPROACH,
        HandPhase.GRASP,
        HandPhase.MANIPULATE,
        HandPhase.RELEASE,
        HandPhase.RETRACT,
    ]


def test_active_flower_is_static_when_free_and_attached_during_contact() -> None:
    phases = canonical_contact_phases(17)
    hand = np.stack((np.arange(17, dtype=np.float32), np.full(17, 20.0)), axis=1)
    flower = causal_flower_grip_trajectory(np, hand, phases)
    contact = np.asarray(
        [
            phase in {HandPhase.GRASP, HandPhase.MANIPULATE, HandPhase.RELEASE}
            for phase in phases
        ]
    )
    first = int(np.flatnonzero(contact)[0])
    last = int(np.flatnonzero(contact)[-1])

    assert np.allclose(flower[:first], flower[first])
    assert np.allclose(flower[contact], hand[contact])
    assert np.allclose(flower[last + 1 :], flower[last])


def test_schedule_rejects_non_vace_frame_count() -> None:
    try:
        canonical_contact_phases(18)
    except ValueError as error:
        assert "4n+1" in str(error)
    else:
        raise AssertionError("expected invalid frame count to fail")
