"""Evidence-gated, plugin-oriented video data production control plane."""

from phiagent.data_engine.capacity import (
    BenchmarkProfile,
    CapacityAssumptions,
    CapacityEstimate,
    estimate_capacity,
    load_profiles,
)
from phiagent.data_engine.controller import CampaignController
from phiagent.data_engine.planner import CampaignPlan, JobSpec, WindowSpec, compile_campaign
from phiagent.data_engine.plugins import PluginDescriptor, PluginRegistry
from phiagent.data_engine.schema import (
    CampaignSpec,
    ClaimScope,
    PipelineContract,
    ReplacementScope,
    SourceClip,
    TargetAsset,
)
from phiagent.data_engine.state import (
    AuditReport,
    CampaignState,
    EvidenceRef,
    JobState,
    JobStatus,
)

__all__ = [
    "AuditReport",
    "BenchmarkProfile",
    "CampaignPlan",
    "CampaignController",
    "CampaignSpec",
    "CampaignState",
    "CapacityAssumptions",
    "CapacityEstimate",
    "ClaimScope",
    "EvidenceRef",
    "JobSpec",
    "JobState",
    "JobStatus",
    "PipelineContract",
    "PluginDescriptor",
    "PluginRegistry",
    "ReplacementScope",
    "SourceClip",
    "TargetAsset",
    "WindowSpec",
    "compile_campaign",
    "estimate_capacity",
    "load_profiles",
]
