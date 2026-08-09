"""Measured structural alignment diagnostics for generated robot videos."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from phiagent.rendering.cosmos3 import PreflightError, _probe_control_video, _write_json


@dataclass(frozen=True)
class StructuralAlignmentConfig:
    mean_edge_ssim_threshold: float = 0.35
    minimum_frame_edge_ssim: float = 0.10

    def __post_init__(self) -> None:
        if not 0 <= self.mean_edge_ssim_threshold <= 1:
            raise ValueError("mean edge SSIM threshold must be between zero and one")
        if not 0 <= self.minimum_frame_edge_ssim <= 1:
            raise ValueError("minimum frame edge SSIM must be between zero and one")


_ALL_SSIM = re.compile(r"(?:^|\s)All:([0-9.eE+-]+)")


def parse_ssim_stats(text: str) -> tuple[float, ...]:
    values = tuple(float(match.group(1)) for line in text.splitlines() if (match := _ALL_SSIM.search(line)))
    if not values:
        raise ValueError("SSIM stats contain no per-frame All values")
    return values


class StructuralAlignmentEvaluator:
    """Compare control/output edges without claiming pose-level acceptance."""

    def __init__(self, config: StructuralAlignmentConfig | None = None) -> None:
        self.config = config or StructuralAlignmentConfig()

    @staticmethod
    def _run(command: list[str]) -> None:
        subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def evaluate(
        self,
        control_video: Path,
        generated_video: Path,
        report_path: Path,
    ) -> dict[str, Any]:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise PreflightError("ffmpeg is required for structural alignment evaluation")
        control_stream = _probe_control_video(control_video)
        generated_stream = _probe_control_video(generated_video)
        for field in ("frame_count", "fps", "width", "height"):
            tolerance = 1e-3 if field == "fps" else 0
            if abs(control_stream[field] - generated_stream[field]) > tolerance:
                raise PreflightError(
                    f"alignment requires equal {field}: "
                    f"{control_stream[field]} != {generated_stream[field]}"
                )

        work_dir = report_path.parent / "alignment"
        work_dir.mkdir()
        control_edges = work_dir / "control_edges.mkv"
        generated_edges = work_dir / "generated_edges.mkv"
        stats_path = work_dir / "edge_ssim.txt"
        edge_filter = "format=gray,edgedetect=low=0.1:high=0.4"
        for source, destination in (
            (control_video, control_edges),
            (generated_video, generated_edges),
        ):
            self._run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-vf",
                    edge_filter,
                    "-c:v",
                    "ffv1",
                    str(destination),
                ]
            )
        self._run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(control_edges),
                "-i",
                str(generated_edges),
                "-lavfi",
                f"ssim=stats_file={stats_path}",
                "-f",
                "null",
                "-",
            ]
        )
        per_frame = parse_ssim_stats(stats_path.read_text())
        mean_ssim = fmean(per_frame)
        minimum_ssim = min(per_frame)
        control_alignment_passed = (
            mean_ssim >= self.config.mean_edge_ssim_threshold
            and minimum_ssim >= self.config.minimum_frame_edge_ssim
        )
        report: dict[str, Any] = {
            "status": "measured",
            "accepted": False,
            "control_alignment_passed": control_alignment_passed,
            "mean_edge_ssim": mean_ssim,
            "minimum_edge_ssim": minimum_ssim,
            "per_frame_edge_ssim": per_frame,
            "thresholds": {
                "mean_edge_ssim": self.config.mean_edge_ssim_threshold,
                "minimum_frame_edge_ssim": self.config.minimum_frame_edge_ssim,
            },
            "reason": (
                "structural edge agreement is diagnostic only; pose-level robot/object "
                "alignment remains required"
            ),
        }
        _write_json(report_path, report)
        return report
