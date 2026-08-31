"""Robot/end-effector capability manifests for the L5 adapter ecosystem."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from phiagent.benchmark.schema import BenchmarkCase, EmbodimentSpec


@dataclass(frozen=True)
class EndEffectorLimits:
    mass_kg: float
    max_stroke_m: float
    max_grip_force_n: float
    speed_modes_m_s: tuple[float, ...]
    default_speed_m_s: float
    repeatability_m: float
    voltage_v: float
    max_current_a: float
    feedback_channels: tuple[str, ...]
    source_url: str
    source_revision: str
    high_speed_requires_confirmation: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EndEffectorLimits":
        def positive(name: str) -> float:
            value = float(payload[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"end-effector {name} must be positive")
            return value

        modes = tuple(float(value) for value in payload["speed_modes_m_s"])
        if not modes or any(not math.isfinite(value) or value <= 0 for value in modes):
            raise ValueError("end-effector speed modes must be positive")
        default_speed = positive("default_speed_m_s")
        if default_speed not in modes:
            raise ValueError("default end-effector speed must be one declared mode")
        feedback = tuple(str(value) for value in payload["feedback_channels"])
        if not feedback or len(set(feedback)) != len(feedback):
            raise ValueError("end-effector feedback channels must be unique and non-empty")
        confirmation = payload.get("high_speed_requires_confirmation")
        if not isinstance(confirmation, bool):
            raise ValueError("high_speed_requires_confirmation must be boolean")
        return cls(
            mass_kg=positive("mass_kg"),
            max_stroke_m=positive("max_stroke_m"),
            max_grip_force_n=positive("max_grip_force_n"),
            speed_modes_m_s=modes,
            default_speed_m_s=default_speed,
            repeatability_m=positive("repeatability_m"),
            voltage_v=positive("voltage_v"),
            max_current_a=positive("max_current_a"),
            feedback_channels=feedback,
            source_url=str(payload["source_url"]),
            source_revision=str(payload["source_revision"]),
            high_speed_requires_confirmation=confirmation,
        )


@dataclass(frozen=True)
class HardwareAdapterManifest:
    adapter_name: str
    adapter_version: str
    target: EmbodimentSpec
    joint_names: tuple[tuple[str, ...], ...]
    action_interfaces: tuple[str, ...]
    control_rate_hz: float
    telemetry_channels: tuple[str, ...]
    safety_channels: tuple[str, ...]
    execution_enabled: bool
    evidence_only: bool
    end_effector_limits: EndEffectorLimits | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardwareAdapterManifest":
        target = EmbodimentSpec.from_dict(payload["target"])
        joint_names = tuple(tuple(str(name) for name in arm) for arm in payload["joint_names"])
        if len(joint_names) != target.arm_count or any(
            len(names) != dof for names, dof in zip(joint_names, target.arm_dof)
        ):
            raise ValueError("hardware joint_names must match target arm count and DOF")
        action_interfaces = tuple(str(value) for value in payload["action_interfaces"])
        telemetry = tuple(str(value) for value in payload["telemetry_channels"])
        safety = tuple(str(value) for value in payload["safety_channels"])
        if not action_interfaces or not telemetry or not safety:
            raise ValueError("hardware adapter must declare action, telemetry, and safety channels")
        execution_enabled = payload.get("execution_enabled")
        evidence_only = payload.get("evidence_only")
        if not isinstance(execution_enabled, bool) or not isinstance(evidence_only, bool):
            raise ValueError("hardware execution flags must be boolean")
        if execution_enabled and evidence_only:
            raise ValueError("an evidence-only adapter cannot enable execution")
        rate = float(payload["control_rate_hz"])
        if rate <= 0:
            raise ValueError("control_rate_hz must be positive")
        raw_limits = payload.get("end_effector_limits")
        if raw_limits is not None and not isinstance(raw_limits, Mapping):
            raise ValueError("end_effector_limits must be an object")
        return cls(
            adapter_name=str(payload["adapter_name"]),
            adapter_version=str(payload["adapter_version"]),
            target=target,
            joint_names=joint_names,
            action_interfaces=action_interfaces,
            control_rate_hz=rate,
            telemetry_channels=telemetry,
            safety_channels=safety,
            execution_enabled=execution_enabled,
            evidence_only=evidence_only,
            end_effector_limits=(
                EndEffectorLimits.from_dict(raw_limits) if raw_limits is not None else None
            ),
        )

    @classmethod
    def from_json(cls, path: Path) -> "HardwareAdapterManifest":
        payload = json.loads(path.expanduser().resolve().read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("hardware adapter manifest must be a JSON object")
        return cls.from_dict(payload)

    def compatibility(self, case: BenchmarkCase) -> dict[str, Any]:
        checks = {
            "robot_model": self.target.robot_model == case.target.robot_model,
            "end_effector": self.target.end_effector == case.target.end_effector,
            "arm_count": self.target.arm_count == case.target.arm_count,
            "arm_dof": self.target.arm_dof == case.target.arm_dof,
            "telemetry": all(
                required in self.telemetry_channels
                for required in ("timestamp_s", "joint_position", "eef_pose", "gripper_width")
            ),
            "safety": all(
                required in self.safety_channels
                for required in (
                    "collision_count",
                    "emergency_stop",
                    "human_intervention",
                    "force_limit_violation",
                )
            ),
            "end_effector_limits": self.end_effector_limits is not None,
        }
        return {
            "case_id": case.case_id,
            "adapter_name": self.adapter_name,
            "compatible": all(checks.values()),
            "checks": checks,
            "execution_enabled": self.execution_enabled,
            "evidence_only": self.evidence_only,
        }
