from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .tracking import BallDetection


@dataclass(frozen=True, slots=True)
class BallDetectorConfig:
    min_area_px: float = 3.0
    max_area_px: float = 180.0
    min_circularity: float = 0.35
    min_confidence: float = 0.20


class BallDetector:
    """Initial visual detector based on motion, brightness and compact shape."""

    def __init__(self, config: BallDetectorConfig | None = None) -> None:
        self.config = config or BallDetectorConfig()
        self._background = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=20, detectShadows=False
        )

    def detect(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        timestamp_s: float,
    ) -> list[BallDetection]:
        foreground = self._background.apply(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bright = cv2.inRange(hsv, (0, 0, 145), (179, 255, 255))
        mask = cv2.bitwise_and(foreground, bright)
        mask = cv2.medianBlur(mask, 3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[BallDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.config.min_area_px <= area <= self.config.max_area_px:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.config.min_circularity:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x = float(moments["m10"] / moments["m00"])
            y = float(moments["m01"] / moments["m00"])
            area_score = min(1.0, area / max(self.config.min_area_px * 4.0, 1.0))
            confidence = max(0.0, min(1.0, 0.55 * circularity + 0.45 * area_score))
            if confidence >= self.config.min_confidence:
                detections.append(
                    BallDetection(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        x_px=x,
                        y_px=y,
                        confidence=confidence,
                    )
                )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)
