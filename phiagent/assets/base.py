"""Backend-independent articulated-asset generation schemas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AssetGenerationRequest:
    """Inputs for one isolated articulated-asset generation run."""

    description: str
    experiment_root: Path
    reference_image: Path | None = None
    provider: str = "openai"
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("asset description cannot be empty")
        if self.provider not in {"openai", "gemini", "anthropic", "openrouter"}:
            raise ValueError(f"unsupported asset-generation provider: {self.provider!r}")


@dataclass(frozen=True)
class AssetGenerationResult:
    """Persisted outputs from an articulated-asset generation run."""

    artifact: Path
    artifact_format: str
    upstream_run_dir: Path
    experiment_dir: Path
    metadata: Path


@dataclass(frozen=True)
class AssetCompilationRequest:
    """Inputs for compiling an authored ArtiCraft SDK model without an LLM."""

    model_file: Path
    experiment_root: Path


class ArticulatedAssetGenerator(Protocol):
    """Backend-independent articulated-asset generator."""

    def generate(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        """Generate and validate an articulated asset."""
