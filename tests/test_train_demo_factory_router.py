from __future__ import annotations

import json
import sys

from scripts import train_demo_factory_router
from tests.test_demo_factory import _contract, _records


def test_training_cli_writes_promoted_checkpoint(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "records.jsonl"
    with dataset.open("w") as handle:
        for record in _records():
            handle.write(json.dumps(record.to_dict()) + "\n")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"contract": _contract().to_dict()}))
    output = tmp_path / "training"
    monkeypatch.setattr(train_demo_factory_router, "_git_state", lambda: {"head": "test"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_demo_factory_router.py",
            "--dataset",
            str(dataset),
            "--contract",
            str(contract),
            "--experiment-root",
            str(output),
            "--minimum-acceptance-rate",
            "1.0",
        ],
    )

    assert train_demo_factory_router.main() == 0
    experiment = next(output.iterdir())
    manifest = json.loads((experiment / "manifest.json").read_text())
    checkpoint = json.loads((experiment / "policy.json").read_text())

    assert manifest["status"] == "promoted"
    assert checkpoint["promoted"] is True
    assert manifest["artifacts"]["distillation_preferences"]["rows"] == 4
