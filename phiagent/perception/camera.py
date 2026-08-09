"""Explicit pinhole camera conventions used by EPL visualization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from phiagent.physical_language.schema import FrameKind, Point3D


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("camera focal lengths and dimensions must be positive")

    def project(self, point: Point3D) -> tuple[float, float]:
        if point.frame.kind is not FrameKind.CAMERA:
            raise ValueError(f"projection requires a camera-frame point, got {point.frame.key}")
        x, y, z = point.xyz_m
        if z <= 0:
            raise ValueError("cannot project a point behind or on the camera plane")
        return (self.fx * x / z + self.cx, self.fy * y / z + self.cy)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PinholeIntrinsics:
        return cls(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )
