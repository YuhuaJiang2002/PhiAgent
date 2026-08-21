#!/usr/bin/env python3
"""Compatibility entry point for the SAM2/SAM3.1 A/B sleeve evaluator."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.evaluation.segmentation_ab import (  # noqa: E402, F401
    MaskGeometryThresholds,
    effective_component_area_threshold,
    score_attachment_distance,
    score_centroid_continuity,
    score_mask_geometry,
)
from scripts.evaluate_joyai_tshirt_segmentation_ab import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
