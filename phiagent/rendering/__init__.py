"""Robot video rendering adapters."""

from phiagent.rendering.base import (
    TrajectoryConditionedRenderRequest,
    TrajectoryConditionedRenderResult,
    TrajectoryConditionedVideoRenderer,
    VideoRenderer,
    VisualTransferRequest,
    VisualTransferResult,
)
from phiagent.rendering.scene_replacement import (
    EntityRole,
    FrameReplacementRoute,
    NormalizedBox,
    ReplacementGranularity,
    ReplacementOperation,
    ReplacementSpec,
    RouteDiagnostic,
    SceneReplacementPlan,
    Shot,
    TrackKeyframe,
    TrackSegment,
)

__all__ = [
    "EntityRole",
    "FrameReplacementRoute",
    "NormalizedBox",
    "ReplacementGranularity",
    "ReplacementOperation",
    "ReplacementSpec",
    "RouteDiagnostic",
    "SceneReplacementPlan",
    "Shot",
    "TrackKeyframe",
    "TrackSegment",
    "TrajectoryConditionedRenderRequest",
    "TrajectoryConditionedRenderResult",
    "TrajectoryConditionedVideoRenderer",
    "VideoRenderer",
    "VisualTransferRequest",
    "VisualTransferResult",
]
