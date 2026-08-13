from __future__ import annotations

from phiagent.agent.foundation_contact_skill_eval import (
    BehaviorEvalItem,
    compare_behavior_manifests,
    evaluate_items,
    extract_json_object,
)


def test_extract_json_object_accepts_fenced_output() -> None:
    assert extract_json_object('```json\n{"promote": false}\n```') == {"promote": False}


def test_extract_json_object_skips_reasoning_prefix() -> None:
    assert extract_json_object('Decision follows. {"promote": true} trailing') == {
        "promote": True
    }


def test_evaluate_items_requires_exact_object() -> None:
    items = [
        BehaviorEvalItem(
            uid="gate",
            question="question",
            expected_json={"promote": False, "reason": "hard_gate_failed"},
        )
    ]

    def completion(system: str, question: str):
        assert "immutable evidence gates" in system
        assert question == "question"
        return '{"promote": false, "reason": "hard_gate_failed"}', {
            "eval_count": 12,
            "eval_duration": 1_000_000_000,
        }

    result = evaluate_items(items, skill_text="contract", completion=completion)
    assert result["status"] == "WORKING"
    assert result["physical_model_promoted"] is False
    assert result["score"] == 1.0
    assert result["generation_tokens_per_second"] == 12.0


def test_evaluate_items_rejects_extra_keys() -> None:
    items = [
        BehaviorEvalItem(uid="gate", question="q", expected_json={"promote": False})
    ]

    def completion(_system: str, _question: str):
        return '{"promote": false, "confidence": 1}', {}

    result = evaluate_items(items, skill_text="contract", completion=completion)
    assert result["status"] == "PARTIAL"
    assert result["score"] == 0.0


def test_evaluate_items_retries_invalid_json_once() -> None:
    items = [BehaviorEvalItem(uid="gate", question="q", expected_json={"ok": True})]
    outputs = iter(("not-json", '{"ok": true}'))

    def completion(_system: str, question: str):
        if "prior response" in question:
            assert "Retry from the decision ABI" in question
        return next(outputs), {"eval_count": 1, "eval_duration": 1_000_000_000}

    result = evaluate_items(
        items, skill_text="contract", completion=completion, max_retries=1
    )
    assert result["score"] == 1.0
    assert result["traces"][0]["attempt_count"] == 2
    assert result["generated_tokens"] == 2


def _manifest(score: float, passing: list[str], *, model: str = "model") -> dict:
    return {
        "config": {"model": model, "seed": 42, "retries": 0},
        "inputs": {"split_sha256": "split"},
        "evaluation": {
            "score": score,
            "traces": [
                {"uid": uid, "passed": uid in passing} for uid in ("a", "b", "c")
            ],
        },
    }


def test_compare_behavior_manifests_promotes_strict_no_regression_gain() -> None:
    decision = compare_behavior_manifests(
        _manifest(1 / 3, ["a"]), _manifest(2 / 3, ["a", "b"])
    )
    assert decision["promote_behavior_skill"] is True
    assert decision["promote_physical_model"] is False


def test_compare_behavior_manifests_rejects_model_change_and_regression() -> None:
    decision = compare_behavior_manifests(
        _manifest(1 / 3, ["a"]), _manifest(2 / 3, ["b", "c"], model="other")
    )
    assert decision["promote_behavior_skill"] is False
    assert decision["regressed_uids"] == ["a"]
