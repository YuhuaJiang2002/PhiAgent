from __future__ import annotations

import pytest

from phiagent.agent.contact_dynamics_evolution import (
    ArchitectureAssessment,
    ArchitectureEvolutionContract,
    derive_structural_repairs,
    select_architecture,
)


GATES = ("topology", "metric_contact", "causal_stem_motion", "human_review")


def _row(group: str, architecture: str, passed: bool, utility: float) -> ArchitectureAssessment:
    return ArchitectureAssessment(
        group_id=group,
        architecture_id=architecture,
        hard_gates=tuple((gate, passed) for gate in GATES),
        utility=utility,
        cost_units=1.0,
        evidence_path=f"{group}/{architecture}.json",
    )


def _contract() -> ArchitectureEvolutionContract:
    return ArchitectureEvolutionContract(
        required_gates=GATES,
        required_groups=("contact-a", "contact-b"),
        architecture_ids=("pixel", "state-space"),
        maximum_cost_units=2.0,
    )


def test_evolution_promotes_only_complete_all_gate_winner() -> None:
    result = select_architecture(
        [
            _row("contact-a", "pixel", False, 0.9),
            _row("contact-b", "pixel", False, 0.9),
            _row("contact-a", "state-space", True, 0.7),
            _row("contact-b", "state-space", True, 0.8),
        ],
        _contract(),
    )
    assert result["promoted"] is True
    assert result["selected_architecture"] == "state-space"


def test_evolution_rejects_incomplete_tournament() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        select_architecture([_row("contact-a", "pixel", True, 1.0)], _contract())


def test_evolution_never_uses_utility_to_override_a_hard_gate() -> None:
    rows = [
        _row(group, architecture, False, 100.0)
        for group in ("contact-a", "contact-b")
        for architecture in ("pixel", "state-space")
    ]
    result = select_architecture(rows, _contract())
    assert result["promoted"] is False
    assert result["selected_architecture"] is None


def test_failed_gates_create_architecture_repairs_not_parameter_sweeps() -> None:
    result = select_architecture(
        [
            ArchitectureAssessment(
                group_id=group,
                architecture_id=architecture,
                hard_gates=tuple(
                    (gate, gate not in {"metric_contact", "causal_stem_motion"})
                    for gate in GATES
                ),
                utility=100.0,
                cost_units=1.0,
                evidence_path=f"{group}/{architecture}.json",
            )
            for group in ("contact-a", "contact-b")
            for architecture in ("pixel", "state-space")
        ],
        _contract(),
    )
    # The production contract uses the fully named form of this gate.  Verify
    # unknown shorthand cannot silently fall back to a numeric search.
    with pytest.raises(ValueError, match="no architecture-level repair"):
        derive_structural_repairs(result)


def test_production_gate_maps_to_explicit_state_mutation() -> None:
    selection = {
        "architectures": [
            {
                "architecture_id": "pixel",
                "failed_gates_by_group": {
                    "scene-a": ["metric_force_closure", "causal_stem_motion"]
                },
            }
        ]
    }
    repairs = derive_structural_repairs(selection)
    assert {item["component"] for item in repairs} == {
        "contact_state",
        "deformable_object_state",
    }
    assert all(item["mutation_class"] == "architecture_not_hyperparameter" for item in repairs)
