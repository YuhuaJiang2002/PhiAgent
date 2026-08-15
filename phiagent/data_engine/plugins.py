"""Lightweight plugin discovery and contract validation for the data engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Iterable, Protocol

from phiagent.data_engine.schema import CampaignSpec, ClaimScope


PLUGIN_ENTRYPOINT_GROUP = "phiagent.data_engine.plugins"


@dataclass(frozen=True)
class PluginDescriptor:
    name: str
    stage: str
    version: str
    description: str
    capabilities: tuple[str, ...]
    heavyweight: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DataEnginePlugin(Protocol):
    """Runtime ABI implemented by optional model and infrastructure packages."""

    descriptor: PluginDescriptor

    def doctor(self) -> dict[str, object]:
        """Return a read-only readiness report without starting a job."""


BUILTIN_DESCRIPTORS = (
    PluginDescriptor(
        name="local-video-source",
        stage="source",
        version="1.0.0",
        description="Hash-bound local or object-store video ingestion.",
        capabilities=("video", "lineage"),
    ),
    PluginDescriptor(
        name="dex-retarget",
        stage="retarget",
        version="1.0.0",
        description="MediaPipe/Dexpilot hand retargeting adapter.",
        capabilities=("hand", "joint_trajectory", "robot_base_frame"),
        heavyweight=True,
    ),
    PluginDescriptor(
        name="epl-retarget",
        stage="retarget",
        version="1.0.0",
        description="EPL to full-embodiment trajectory adapter.",
        capabilities=("full_embodiment", "joint_trajectory", "robot_base_frame"),
    ),
    PluginDescriptor(
        name="wan-animate2",
        stage="generate",
        version="1.0.0",
        description="Windowed Wan-Animate-2 replacement adapter.",
        capabilities=("hand", "full_embodiment", "visual_training_data"),
        heavyweight=True,
    ),
    PluginDescriptor(
        name="oscar",
        stage="generate",
        version="1.0.0",
        description="OSCAR native articulated action-conditioned video adapter.",
        capabilities=("full_embodiment", "visual_training_data"),
        heavyweight=True,
    ),
    PluginDescriptor(
        name="local-video-auditor",
        stage="audit",
        version="1.0.0",
        description="Deterministic identity, motion, object, temporal, and background gates.",
        capabilities=("visual_training_data", "read_only"),
    ),
    PluginDescriptor(
        name="physical-auditor",
        stage="audit",
        version="1.0.0",
        description="Metric camera, exact trajectory, geometry, and force gates.",
        capabilities=("physically_grounded", "read_only"),
        heavyweight=True,
    ),
)


class PluginRegistry:
    def __init__(self, descriptors: Iterable[PluginDescriptor] = BUILTIN_DESCRIPTORS) -> None:
        self._descriptors: dict[str, PluginDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: PluginDescriptor) -> None:
        if descriptor.name in self._descriptors:
            raise ValueError(f"duplicate data-engine plugin {descriptor.name!r}")
        self._descriptors[descriptor.name] = descriptor

    def discover(self) -> None:
        """Load optional plugins only when discovery is explicitly requested."""

        entry_points = metadata.entry_points()
        selected = (
            entry_points.select(group=PLUGIN_ENTRYPOINT_GROUP)
            if hasattr(entry_points, "select")
            else entry_points.get(PLUGIN_ENTRYPOINT_GROUP, ())
        )
        for entry_point in selected:
            plugin = entry_point.load()
            descriptor = plugin.descriptor if hasattr(plugin, "descriptor") else plugin
            if not isinstance(descriptor, PluginDescriptor):
                raise TypeError(
                    f"plugin entry point {entry_point.name!r} did not expose PluginDescriptor"
                )
            self.register(descriptor)

    def get(self, name: str) -> PluginDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as exc:
            raise ValueError(f"unknown data-engine plugin {name!r}") from exc

    def descriptors(self) -> tuple[PluginDescriptor, ...]:
        return tuple(sorted(self._descriptors.values(), key=lambda item: (item.stage, item.name)))

    def validate_campaign(self, campaign: CampaignSpec) -> tuple[PluginDescriptor, ...]:
        selected = [
            self.get(campaign.pipeline.source_plugin),
            self.get(campaign.pipeline.generator_plugin),
            *(self.get(name) for name in campaign.pipeline.auditor_plugins),
            *(self.get(target.retarget_plugin) for target in campaign.targets),
        ]
        if selected[0].stage != "source":
            raise ValueError("source_plugin must provide the source stage")
        if selected[1].stage != "generate":
            raise ValueError("generator_plugin must provide the generate stage")
        if any(item.stage != "audit" for item in selected[2 : 2 + len(campaign.pipeline.auditor_plugins)]):
            raise ValueError("every auditor_plugin must provide the audit stage")
        generator = selected[1]
        retargeters = {
            descriptor.name: descriptor
            for descriptor in selected
            if descriptor.stage == "retarget"
        }
        for target in campaign.targets:
            if target.replacement_scope.value not in generator.capabilities:
                raise ValueError(
                    f"generator {generator.name!r} does not support "
                    f"{target.replacement_scope.value!r}"
                )
            retargeter = retargeters.get(target.retarget_plugin)
            if retargeter is None:
                raise ValueError(
                    f"retarget plugin {target.retarget_plugin!r} must provide the retarget stage"
                )
            if target.replacement_scope.value not in retargeter.capabilities:
                raise ValueError(
                    f"retarget plugin {retargeter.name!r} does not support "
                    f"{target.replacement_scope.value!r}"
                )
        required_claim = campaign.pipeline.claim_scope.value
        if not any(
            required_claim in descriptor.capabilities
            for descriptor in selected
            if descriptor.stage == "audit"
        ):
            raise ValueError(f"no auditor supports claim scope {required_claim!r}")
        if (
            campaign.pipeline.claim_scope is ClaimScope.PHYSICALLY_GROUNDED
            and "physical-auditor" not in campaign.pipeline.auditor_plugins
        ):
            raise ValueError("physically grounded campaigns require physical-auditor")
        unique = {item.name: item for item in selected}
        return tuple(sorted(unique.values(), key=lambda item: (item.stage, item.name)))
