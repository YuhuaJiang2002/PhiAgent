"""Fail-closed routing for public 4-D flower and deformable-object trackers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TrackModelSpec:
    name: str
    revision: str
    repository: str
    code_license: str
    weight_license: str | None
    custom_video: bool
    supports_rgbd: bool
    persistent_3d_tracks: bool
    semantic_stem_ids: bool
    public_code: bool
    public_weights: bool
    minimum_gpu_memory_mib: int

    def validate(self) -> None:
        required = (
            self.name,
            self.revision,
            self.repository,
            self.code_license,
        )
        if any(not value.strip() for value in required):
            raise ValueError("track model provenance fields must be non-empty")
        if self.minimum_gpu_memory_mib < 0:
            raise ValueError("track model GPU memory cannot be negative")


@dataclass(frozen=True)
class FlowerTrackRequest:
    source_video_sha256: str
    frames: int
    fps: float
    timeline: str
    camera_frame: str
    instance_ids: tuple[str, ...]
    calibrated_rgbd: bool
    independent_metric_scale: bool
    maximum_gpu_memory_mib: int

    def validate(self) -> None:
        if (
            len(self.source_video_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.source_video_sha256
            )
        ):
            raise ValueError("source video requires a lowercase SHA-256")
        if self.frames < 2 or self.fps <= 0:
            raise ValueError("track request needs at least two frames and positive FPS")
        if not self.timeline.strip() or not self.camera_frame.strip():
            raise ValueError("track request frames and timeline must be named")
        if (
            not self.instance_ids
            or len(set(self.instance_ids)) != len(self.instance_ids)
            or any(not value.strip() for value in self.instance_ids)
        ):
            raise ValueError("track request requires unique persistent instance IDs")
        if self.maximum_gpu_memory_mib < 0:
            raise ValueError("track request GPU memory cannot be negative")
        if self.independent_metric_scale and not self.calibrated_rgbd:
            raise ValueError(
                "independent metric scale requires calibrated RGB-D or equivalent geometry"
            )


def latest_track_model_registry() -> tuple[TrackModelSpec, ...]:
    """Return verified public revisions, not paper-only placeholders."""

    return (
        TrackModelSpec(
            name="V-DPM",
            revision="5e2a57cf6007dfb0511a8b396a0805089b9edcc4",
            repository="https://github.com/eldar/vdpm",
            code_license="MIT",
            weight_license="CC-BY-NC-4.0 plus inherited VGGT terms",
            custom_video=True,
            supports_rgbd=False,
            persistent_3d_tracks=True,
            semantic_stem_ids=False,
            public_code=True,
            public_weights=True,
            minimum_gpu_memory_mib=30000,
        ),
        TrackModelSpec(
            name="SpatialTrackerV2-Offline",
            revision="7e12274c52077860cebfe007a6290777db43b63c",
            repository="https://github.com/henry123-boy/SpaTrackerV2",
            code_license="CC-BY-NC-4.0",
            weight_license="CC-BY-NC-4.0",
            custom_video=True,
            supports_rgbd=True,
            persistent_3d_tracks=True,
            semantic_stem_ids=False,
            public_code=True,
            public_weights=True,
            minimum_gpu_memory_mib=24000,
        ),
        TrackModelSpec(
            name="LiteVDPM",
            revision="297ad2f18c2e1cb023b5dc64c577bd83f87aa68b",
            repository="https://github.com/damanimc/LiteVDPM",
            code_license="UNVERIFIED",
            weight_license="inherits V-DPM and VGGT terms",
            custom_video=True,
            supports_rgbd=False,
            persistent_3d_tracks=True,
            semantic_stem_ids=False,
            public_code=True,
            public_weights=False,
            minimum_gpu_memory_mib=16000,
        ),
        TrackModelSpec(
            name="MultiDLO",
            revision="e0b7fa35739731a96ac7569952c00414ca2ad968",
            repository="https://github.com/RMDLO/multidlo",
            code_license="MIT",
            weight_license=None,
            custom_video=True,
            supports_rgbd=True,
            persistent_3d_tracks=False,
            semantic_stem_ids=True,
            public_code=True,
            public_weights=False,
            minimum_gpu_memory_mib=0,
        ),
    )


def select_flower_track_route(
    request: FlowerTrackRequest,
    availability: Mapping[str, bool],
) -> dict[str, object]:
    """Select proposal generators without upgrading learned geometry to metric state."""

    request.validate()
    registry = {model.name: model for model in latest_track_model_registry()}
    for model in registry.values():
        model.validate()

    def ready(name: str) -> bool:
        model = registry[name]
        return bool(
            availability.get(name, False)
            and model.public_code
            and (model.public_weights or name == "MultiDLO")
            and model.minimum_gpu_memory_mib <= request.maximum_gpu_memory_mib
        )

    selected: list[str] = []
    critics: list[str] = []
    if request.calibrated_rgbd and ready("SpatialTrackerV2-Offline"):
        selected.append("SpatialTrackerV2-Offline")
        if ready("MultiDLO"):
            critics.append("MultiDLO")
        geometry_scope = "calibrated_metric_rgbd"
    elif ready("V-DPM"):
        selected.append("V-DPM")
        if ready("SpatialTrackerV2-Offline"):
            critics.append("SpatialTrackerV2-Offline")
        geometry_scope = "learned_relative_4d"
    elif ready("SpatialTrackerV2-Offline"):
        selected.append("SpatialTrackerV2-Offline")
        geometry_scope = "learned_monocular_4d"
    elif ready("LiteVDPM"):
        selected.append("LiteVDPM")
        geometry_scope = "learned_relative_4d_unverified_code_license"
    else:
        return {
            "status": "BLOCKED",
            "selected_proposals": [],
            "topology_critics": [],
            "geometry_scope": "none",
            "metric_claim_allowed": False,
            "reasons": ["no_compatible_public_track_model_available"],
            "required_next_action": (
                "prepare one pinned V-DPM or SpatialTrackerV2 runtime"
            ),
        }

    metric_claim_allowed = bool(
        request.calibrated_rgbd
        and request.independent_metric_scale
        and geometry_scope == "calibrated_metric_rgbd"
    )
    return {
        "status": "READY",
        "selected_proposals": selected,
        "topology_critics": critics,
        "geometry_scope": geometry_scope,
        "metric_claim_allowed": metric_claim_allowed,
        "semantic_instance_source": "external immutable stem prompts",
        "fusion_rule": (
            "models propose point trajectories; a persistent multi-rod optimizer "
            "owns identity, arc length, roots, occlusion hypotheses, and covariance"
        ),
        "limitations": [
            "No selected tracker assigns flower-stem semantics by itself.",
            "Monocular learned depth or point maps cannot establish an independent metre.",
            "Track confidence cannot be relabelled as force or contact evidence.",
        ],
        "models": [
            registry[name].__dict__
            for name in (*selected, *critics)
        ],
    }
