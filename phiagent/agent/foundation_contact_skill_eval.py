"""Direct, reproducible behavior evaluation for the contact-evolution skill.

The SkillHone Claude-Agent bridge is useful for tool-using models, but a model
can understand a promotion contract without being able to emit Anthropic tool
calls.  This module therefore keeps protocol compliance separate from decision
quality: it evaluates a local foundation model through Ollama JSON mode and
compares the resulting object with private, exact expectations.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]
Completion = Callable[[str, str], tuple[str, Mapping[str, Any]]]


@dataclass(frozen=True)
class BehaviorEvalItem:
    """One exact-decision evaluation item."""

    uid: str
    question: str
    expected_json: JsonObject


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_items(path: Path, limit: int = 0) -> list[BehaviorEvalItem]:
    """Load exact JSON expectations from a JSONL split."""

    items: list[BehaviorEvalItem] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            expected = record.get("expected_json")
            if not isinstance(expected, dict):
                raise ValueError(f"{path}:{line_number}: expected_json must be an object")
            items.append(
                BehaviorEvalItem(
                    uid=str(record["uid"]),
                    question=str(record["question"]),
                    expected_json=expected,
                )
            )
            if limit and len(items) >= limit:
                break
    if not items:
        raise ValueError(f"no evaluation items in {path}")
    return items


def extract_json_object(text: str) -> JsonObject:
    """Extract the first complete JSON object without accepting scalar output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for offset, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output contains no complete JSON object")


def ollama_completion(
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    seed: int,
    num_predict: int = 256,
) -> Completion:
    """Create a deterministic Ollama JSON-mode completion callable."""

    endpoint = f"{base_url.rstrip('/')}/api/chat"

    def complete(system_prompt: str, question: str) -> tuple[str, Mapping[str, Any]]:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": "15m",
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_predict": num_predict,
            },
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
        message = payload.get("message") or {}
        content = message.get("content") or ""
        if not content and message.get("thinking"):
            content = message["thinking"]
        return str(content), payload

    return complete


def evaluate_items(
    items: Iterable[BehaviorEvalItem],
    *,
    skill_text: str,
    completion: Completion,
    max_retries: int = 0,
) -> JsonObject:
    """Evaluate exact decisions and retain timings and non-secret raw outputs."""

    system_prompt = (
        "You are the promotion supervisor described by the skill below. "
        "Apply its immutable evidence gates literally. Return only the exact JSON "
        "object requested by the user; do not add keys or Markdown.\n\n"
        f"{skill_text}"
    )
    traces: list[JsonObject] = []
    wall_start = time.perf_counter()
    for item in items:
        started = time.perf_counter()
        raw = ""
        metadata: Mapping[str, Any] = {}
        error = ""
        predicted: JsonObject | None = None
        attempt_records: list[JsonObject] = []
        for attempt in range(max_retries + 1):
            retry_instruction = ""
            if attempt:
                retry_instruction = (
                    "\n\nYour prior response was not a complete exact JSON object. "
                    "Retry from the decision ABI. Return only the requested keys."
                )
            try:
                raw, metadata = completion(
                    system_prompt, f"{item.question}{retry_instruction}"
                )
                predicted = extract_json_object(raw)
                error = ""
            except Exception as exc:  # noqa: BLE001 - retain evaluation failure
                error = f"{type(exc).__name__}: {exc}"
            attempt_records.append(
                {
                    "attempt": attempt + 1,
                    "raw_output": raw,
                    "error": error,
                    "prompt_eval_count": int(metadata.get("prompt_eval_count", 0) or 0),
                    "eval_count": int(metadata.get("eval_count", 0) or 0),
                    "eval_duration_ns": int(metadata.get("eval_duration", 0) or 0),
                }
            )
            if predicted is not None:
                break
        passed = predicted == item.expected_json
        prompt_eval_count = sum(int(record["prompt_eval_count"]) for record in attempt_records)
        eval_count = sum(int(record["eval_count"]) for record in attempt_records)
        eval_duration_ns = sum(int(record["eval_duration_ns"]) for record in attempt_records)
        traces.append(
            {
                "uid": item.uid,
                "passed": passed,
                "expected_json": item.expected_json,
                "predicted_json": predicted,
                "raw_output": raw,
                "error": error,
                "duration_s": round(time.perf_counter() - started, 6),
                "attempt_count": len(attempt_records),
                "attempts": attempt_records,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "eval_duration_ns": eval_duration_ns,
            }
        )

    wall_s = time.perf_counter() - wall_start
    passed_count = sum(bool(trace["passed"]) for trace in traces)
    generated_tokens = sum(int(trace["eval_count"]) for trace in traces)
    model_eval_s = sum(int(trace["eval_duration_ns"]) for trace in traces) / 1e9
    return {
        "status": "WORKING" if passed_count == len(traces) else "PARTIAL",
        "behavioral_eval_only": True,
        "physical_model_promoted": False,
        "score": passed_count / len(traces),
        "passed": passed_count,
        "total": len(traces),
        "wall_s": round(wall_s, 6),
        "items_per_second": round(len(traces) / wall_s, 6) if wall_s else None,
        "generated_tokens": generated_tokens,
        "generation_tokens_per_second": (
            round(generated_tokens / model_eval_s, 6) if model_eval_s else None
        ),
        "traces": traces,
    }


def compare_behavior_manifests(
    incumbent: Mapping[str, Any], challenger: Mapping[str, Any]
) -> JsonObject:
    """Decide whether a skill-only challenger may replace its incumbent.

    This promotion is deliberately scoped to the supervisor's behavioral skill;
    it never promotes the underlying video or contact model.
    """

    inc_config = incumbent["config"]
    cha_config = challenger["config"]
    inc_inputs = incumbent["inputs"]
    cha_inputs = challenger["inputs"]
    comparable = {
        "same_model": inc_config["model"] == cha_config["model"],
        "same_seed": inc_config["seed"] == cha_config["seed"],
        "same_retries": inc_config.get("retries", 0) == cha_config.get("retries", 0),
        "same_split": inc_inputs["split_sha256"] == cha_inputs["split_sha256"],
    }
    inc_eval = incumbent["evaluation"]
    cha_eval = challenger["evaluation"]
    incumbent_pass = {
        trace["uid"] for trace in inc_eval["traces"] if bool(trace["passed"])
    }
    challenger_pass = {
        trace["uid"] for trace in cha_eval["traces"] if bool(trace["passed"])
    }
    regressed = sorted(incumbent_pass - challenger_pass)
    score_gain = float(cha_eval["score"]) - float(inc_eval["score"])
    promote = all(comparable.values()) and score_gain > 0 and not regressed
    reasons: list[str] = []
    reasons.extend(name for name, passed in comparable.items() if not passed)
    if score_gain <= 0:
        reasons.append("no_strict_score_gain")
    if regressed:
        reasons.append("previously_passing_items_regressed")
    return {
        "promote_behavior_skill": promote,
        "promote_physical_model": False,
        "comparable": comparable,
        "incumbent_score": float(inc_eval["score"]),
        "challenger_score": float(cha_eval["score"]),
        "score_gain": score_gain,
        "regressed_uids": regressed,
        "reasons": reasons or ["strict_score_gain_without_regression"],
    }
