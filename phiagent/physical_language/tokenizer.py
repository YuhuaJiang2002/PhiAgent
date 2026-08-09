"""Deterministic first-generation tokenizer for continuous EPL chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from phiagent.physical_language.schema import EPLChunk


@dataclass(frozen=True)
class EPLTokenizerConfig:
    position_min_m: float = -2.0
    position_max_m: float = 2.0
    position_bins: int = 256
    aperture_max_m: float = 0.2
    aperture_bins: int = 64
    confidence_bins: int = 16

    def __post_init__(self) -> None:
        if self.position_max_m <= self.position_min_m:
            raise ValueError("position range must be increasing")
        if min(self.position_bins, self.aperture_bins, self.confidence_bins) < 2:
            raise ValueError("all tokenizer bin counts must be at least 2")
        if self.aperture_max_m <= 0:
            raise ValueError("aperture_max_m must be positive")


def _bin(value: float, lower: float, upper: float, bins: int) -> int:
    clipped = min(max(value, lower), upper)
    scaled = (clipped - lower) / (upper - lower)
    return min(bins - 1, math.floor(scaled * bins))


class EPLTokenizer:
    """Tokenize categorical state and important bounded continuous components."""

    def __init__(self, config: EPLTokenizerConfig | None = None) -> None:
        self.config = config or EPLTokenizerConfig()

    def encode_chunk(self, chunk: EPLChunk) -> tuple[str, ...]:
        cfg = self.config
        tokens = [
            f"<PHASE:{chunk.phase.value}>",
            f"<CONTACT:{chunk.contact_state.value}>",
        ]
        for axis, value in zip("XYZ", chunk.eef_delta.translation_m):
            index = _bin(
                value, cfg.position_min_m, cfg.position_max_m, cfg.position_bins
            )
            tokens.append(f"<EEF_D{axis}:{index:03d}>")
        for axis, value in zip("XYZ", chunk.wrist_pose.translation_m):
            index = _bin(
                value, cfg.position_min_m, cfg.position_max_m, cfg.position_bins
            )
            tokens.append(f"<WRIST_{axis}:{index:03d}>")
        aperture = _bin(
            chunk.hand_aperture_m, 0.0, cfg.aperture_max_m, cfg.aperture_bins
        )
        confidence = _bin(chunk.confidence, 0.0, 1.0, cfg.confidence_bins)
        tokens.extend(
            [
                f"<APERTURE:{aperture:02d}>",
                f"<CONF:{confidence:02d}>",
                f"<OBJECT:{'PRESENT' if chunk.object_pose else 'ABSENT'}>",
            ]
        )
        return tuple(tokens)

