"""Lightweight experiment provenance helpers shared by optional GPU launchers."""

from __future__ import annotations

import importlib.metadata


def package_inventory() -> str:
    """Return a deterministic package inventory without requiring pip."""
    packages = set()
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata.get("Name")
            version = distribution.version
        except (OSError, UnicodeError):
            continue
        if name:
            packages.add(f"{name}=={version}")
    if not packages:
        raise RuntimeError("could not inventory installed Python packages")
    return "\n".join(sorted(packages, key=str.casefold)) + "\n"
