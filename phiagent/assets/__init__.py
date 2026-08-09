"""Optional articulated-asset generation interfaces."""

from phiagent.assets.articraft import (
    ARTICRAFT_COMMIT,
    ArticraftAssetGenerator,
    ArticraftConfig,
    AssetGenerationError,
)
from phiagent.assets.base import (
    ArticulatedAssetGenerator,
    AssetCompilationRequest,
    AssetGenerationRequest,
    AssetGenerationResult,
)

__all__ = [
    "ARTICRAFT_COMMIT",
    "ArticraftAssetGenerator",
    "ArticraftConfig",
    "ArticulatedAssetGenerator",
    "AssetCompilationRequest",
    "AssetGenerationError",
    "AssetGenerationRequest",
    "AssetGenerationResult",
]
