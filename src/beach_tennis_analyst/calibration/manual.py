from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PixelPoint:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class ManualCalibration:
    """Four court corners ordered near-left, near-right, far-right, far-left."""

    near_left: PixelPoint
    near_right: PixelPoint
    far_right: PixelPoint
    far_left: PixelPoint
    source_width_px: int
    source_height_px: int

    @property
    def source_points(self) -> np.ndarray:
        return np.asarray(
            [
                [self.near_left.x, self.near_left.y],
                [self.near_right.x, self.near_right.y],
                [self.far_right.x, self.far_right.y],
                [self.far_left.x, self.far_left.y],
            ],
            dtype=np.float32,
        )

    def validate(self) -> None:
        if self.source_width_px <= 0 or self.source_height_px <= 0:
            raise ValueError("Source dimensions must be positive")
        points = self.source_points
        if np.unique(points, axis=0).shape[0] != 4:
            raise ValueError("Calibration requires four distinct points")
        contour = points.reshape((-1, 1, 2))
        if abs(cv2.contourArea(contour)) < 100.0:
            raise ValueError("Calibration polygon is degenerate or too small")
        for x, y in points:
            if not (0 <= x < self.source_width_px and 0 <= y < self.source_height_px):
                raise ValueError(f"Calibration point outside frame: {(x, y)}")

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ManualCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        calibration = cls(
            near_left=PixelPoint(**payload["near_left"]),
            near_right=PixelPoint(**payload["near_right"]),
            far_right=PixelPoint(**payload["far_right"]),
            far_left=PixelPoint(**payload["far_left"]),
            source_width_px=int(payload["source_width_px"]),
            source_height_px=int(payload["source_height_px"]),
        )
        calibration.validate()
        return calibration

    @classmethod
    def from_points(
        cls,
        points: Sequence[tuple[float, float]],
        frame_width_px: int,
        frame_height_px: int,
    ) -> "ManualCalibration":
        if len(points) != 4:
            raise ValueError("Expected exactly four points")
        calibration = cls(
            near_left=PixelPoint(*points[0]),
            near_right=PixelPoint(*points[1]),
            far_right=PixelPoint(*points[2]),
            far_left=PixelPoint(*points[3]),
            source_width_px=frame_width_px,
            source_height_px=frame_height_px,
        )
        calibration.validate()
        return calibration
