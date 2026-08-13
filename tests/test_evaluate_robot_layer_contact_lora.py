from __future__ import annotations

from scripts.evaluate_robot_layer_contact_lora import adapter_gates


def test_adapter_gates_reject_contact_regression() -> None:
    teacher = {
        "inside_teacher_similarity": 1.0,
        "contact_teacher_similarity": 1.0,
        "outside_source_similarity": 0.95,
        "topology_edge_teacher_similarity": 1.0,
        "temporal_teacher_similarity": 1.0,
        "canonical_palette_surprisal": 2.0,
        "high_chroma_fraction": 0.0,
        "replacement_coverage": 0.9,
    }
    zero = {**teacher, "contact_teacher_similarity": 0.5, "inside_teacher_similarity": 0.5}
    adapted = {**zero, "contact_teacher_similarity": 0.4}

    gates = adapter_gates(zero, adapted, teacher, distinctness=1.0)

    assert gates["contact_teacher_similarity_not_regressed"] is False
