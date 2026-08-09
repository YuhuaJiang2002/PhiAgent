from __future__ import annotations

from collections import Counter

import pytest

from phiagent.physical_language.schema import ContactState, ManipulationPhase
from phiagent.training.epl_agent import (
    RepairAction,
    encode_example,
    feature_names,
    generate_policy_examples,
)


def test_policy_examples_are_reproducible_and_cover_actions() -> None:
    left = generate_policy_examples(600, seed=7)
    right = generate_policy_examples(600, seed=7)

    assert left == right
    counts = Counter(example.action for example in left)
    assert set(counts) == set(RepairAction)
    assert min(counts.values()) >= 50


def test_epl_mask_preserves_diagnostics_and_zeros_context() -> None:
    example = generate_policy_examples(12, seed=3)[0]
    conditioned = encode_example(example, include_epl=True)
    masked = encode_example(example, include_epl=False)
    diagnostic_count = len(example.diagnostics)

    assert len(conditioned) == len(feature_names())
    assert masked[:-diagnostic_count] == (0.0,) * (
        len(masked) - diagnostic_count
    )
    assert masked[-diagnostic_count:] == conditioned[-diagnostic_count:]


def test_ambiguous_noise_uses_epl_contact_context() -> None:
    examples = generate_policy_examples(6000, seed=11)
    noisy = [
        example
        for example in examples
        if example.diagnostics[1] > 0.5
    ]

    assert any(
        example.action is RepairAction.CONTACT_SAFE_REPLAN
        and (
            example.contact_state in {ContactState.STABLE, ContactState.SLIPPING}
            or example.phase in {
                ManipulationPhase.GRASP,
                ManipulationPhase.MANIPULATE,
            }
        )
        for example in noisy
    )
    assert any(
        example.action is RepairAction.SMOOTH_TRAJECTORY for example in noisy
    )


def test_dataset_rejects_too_few_examples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        generate_policy_examples(5, seed=0)
