from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .court import BeachTennisCourt
from .manual import ManualCalibration


@dataclass(frozen=True, slots=True)
class MetricPoint:
    x_m: float
    y_m: float


class CourtProjector:
    """Projects image coordinates onto the metric court plane."""

    def __init__(self, calibration: ManualCalibration, court: BeachTennisCourt | None = None) -> None:
        calibration.validate()
        self.calibration = calibration
        self.court = court or BeachTennisCourt()
        self._matrix = cv2.getPerspectiveTransform(
            calibration.source_points,
            self.court.metric_corners,
        )
        self._inverse_matrix = cv2.getPerspectiveTransform(
            self.court.metric_corners,
            calibration.source_points,
        )

    @property
    def homography(self) -> np.ndarray:
        return self._matrix.copy()

    def pixel_to_metric(self, x_px: float, y_px: float) -> MetricPoint:
        source = np.asarray([[[x_px, y_px]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(source, self._matrix)[0, 0]
        return MetricPoint(float(projected[0]), float(projected[1]))

    def metric_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        source = np.asarray([[[x_m, y_m]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(source, self._inverse_matrix)[0, 0]
        return float(projected[0]), float(projected[1])

    def reprojection_error_px(self) -> float:
        errors: list[float] = []
        for metric, pixel in zip(self.court.metric_corners, self.calibration.source_points, strict=True):
            x_px, y_px = self.metric_to_pixel(float(metric[0]), float(metric[1]))
            errors.append(float(np.hypot(x_px - pixel[0], y_px - pixel[1])))
        return float(np.mean(errors))
