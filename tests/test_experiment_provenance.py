from __future__ import annotations

from scripts.experiment_provenance import package_inventory


def test_package_inventory_is_nonempty_and_deterministic() -> None:
    first = package_inventory()
    second = package_inventory()
    assert first == second
    assert "pytest==" in first.lower()
    assert first.endswith("\n")
