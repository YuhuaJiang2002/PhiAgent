"""One-EPL-to-many-embodiment orchestration and canonical masking."""

from __future__ import annotations

from dataclasses import dataclass

from phiagent.data.schema import CanonicalAction
from phiagent.physical_language.schema import EPLSequence
from phiagent.retargeting.base import (
    LinearEPLRetargeter,
    LinearRetargetingConfig,
    RetargetingResult,
)


@dataclass(frozen=True)
class MultiEmbodimentResult:
    canonical_dimension: int
    results: dict[str, RetargetingResult]
    canonical_actions: dict[str, tuple[CanonicalAction, ...]]


def retarget_multiple(
    epl: EPLSequence, configs: tuple[LinearRetargetingConfig, ...]
) -> MultiEmbodimentResult:
    if len(configs) < 2:
        raise ValueError("multi-embodiment retargeting requires at least two configs")
    names = [config.embodiment.name for config in configs]
    if len(set(names)) != len(names):
        raise ValueError("multi-embodiment config names must be unique")
    dimension = max(config.embodiment.dof for config in configs)
    results = {
        config.embodiment.name: LinearEPLRetargeter(config).retarget(epl)
        for config in configs
    }
    canonical = {
        name: result.trajectory.canonical_actions(dimension)
        for name, result in results.items()
    }
    return MultiEmbodimentResult(dimension, results, canonical)
