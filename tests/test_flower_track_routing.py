from __future__ import annotations

import pytest

from phiagent.perception.flower_track_routing import (
    FlowerTrackRequest,
    select_flower_track_route,
)


def _request(
    *,
    calibrated_rgbd: bool = False,
    independent_metric_scale: bool = False,
) -> FlowerTrackRequest:
    return FlowerTrackRequest(
        source_video_sha256="a" * 64,
        frames=660,
        fps=24.0,
        timeline="frame:source_video",
        camera_frame="camera:source_pixels",
        instance_ids=("stem-pink-01", "stem-white-02"),
        calibrated_rgbd=calibrated_rgbd,
        independent_metric_scale=independent_metric_scale,
        maximum_gpu_memory_mib=40000,
    )


def test_monocular_route_keeps_vdpm_geometry_relative() -> None:
    route = select_flower_track_route(
        _request(),
        {
            "V-DPM": True,
            "SpatialTrackerV2-Offline": True,
            "MultiDLO": True,
        },
    )

    assert route["status"] == "READY"
    assert route["selected_proposals"] == ["V-DPM"]
    assert route["topology_critics"] == ["SpatialTrackerV2-Offline"]
    assert route["geometry_scope"] == "learned_relative_4d"
    assert route["metric_claim_allowed"] is False


def test_calibrated_rgbd_route_uses_topology_critic_and_allows_metric() -> None:
    route = select_flower_track_route(
        _request(calibrated_rgbd=True, independent_metric_scale=True),
        {
            "SpatialTrackerV2-Offline": True,
            "MultiDLO": True,
        },
    )

    assert route["selected_proposals"] == ["SpatialTrackerV2-Offline"]
    assert route["topology_critics"] == ["MultiDLO"]
    assert route["metric_claim_allowed"] is True


def test_track_route_fails_closed_without_public_runtime() -> None:
    route = select_flower_track_route(_request(), {})

    assert route["status"] == "BLOCKED"
    assert route["metric_claim_allowed"] is False


def test_independent_scale_cannot_exist_without_calibrated_geometry() -> None:
    with pytest.raises(ValueError, match="requires calibrated RGB-D"):
        _request(independent_metric_scale=True).validate()
