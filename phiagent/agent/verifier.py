"""Structured verification reports from measured simulator results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phiagent.agent.tools import PhysicsTools
from phiagent.simulation.base import SimulationResult


@dataclass(frozen=True)
class VerificationReport:
    accepted: bool
    collision: dict[str, object]
    contact: dict[str, object]
    reachability: dict[str, object]
    diagnoses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "collision": self.collision,
            "contact": self.contact,
            "reachability": self.reachability,
            "diagnoses": list(self.diagnoses),
        }


class AgentVerifier:
    """Apply explicit acceptance rules without hidden language-model judgment."""

    def verify(self, result: SimulationResult) -> VerificationReport:
        collision = PhysicsTools.check_collision(result)
        contact = PhysicsTools.check_contact(result)
        reachability = PhysicsTools.check_reachability(result)
        diagnoses: list[str] = []
        if not collision["passed"]:
            diagnoses.append("forbidden collision detected")
        if not reachability["passed"]:
            diagnoses.append("trajectory violates reachability or joint-limit constraints")
        if result.task_success is False:
            diagnoses.append("task criteria were not satisfied")
        accepted = (
            result.physically_valid
            and result.task_success is not False
            and bool(collision["passed"])
            and bool(reachability["passed"])
        )
        return VerificationReport(
            accepted=accepted,
            collision=collision,
            contact=contact,
            reachability=reachability,
            diagnoses=tuple(diagnoses),
        )
