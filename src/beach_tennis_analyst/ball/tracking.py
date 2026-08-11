from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


@dataclass(frozen=True, slots=True)
class BallDetection:
    frame_index: int
    timestamp_s: float
    x_px: float
    y_px: float
    confidence: float


@dataclass(frozen=True, slots=True)
class BallTrackPoint:
    frame_index: int
    timestamp_s: float
    x_px: float
    y_px: float
    vx_px_s: float | None
    vy_px_s: float | None
    speed_px_s: float | None
    confidence: float
    interpolated: bool = False


@dataclass(frozen=True, slots=True)
class BallTrackerConfig:
    max_gap_frames: int = 4
    max_jump_px: float = 180.0
    minimum_confidence: float = 0.20


class BallTracker:
    """Builds a conservative single-ball track from per-frame detections."""

    def __init__(self, config: BallTrackerConfig | None = None) -> None:
        self.config = config or BallTrackerConfig()

    def build(self, detections: Iterable[BallDetection]) -> list[BallTrackPoint]:
        items = sorted(detections, key=lambda item: item.frame_index)
        accepted: list[BallDetection] = []
        for detection in items:
            if detection.confidence < self.config.minimum_confidence:
                continue
            if accepted:
                previous = accepted[-1]
                gap = detection.frame_index - previous.frame_index
                distance = hypot(detection.x_px - previous.x_px, detection.y_px - previous.y_px)
                if gap <= 0 or distance > self.config.max_jump_px * max(1, gap):
                    continue
            accepted.append(detection)

        points: list[BallTrackPoint] = []
        for index, item in enumerate(accepted):
            vx = vy = speed = None
            if index:
                previous = accepted[index - 1]
                dt = item.timestamp_s - previous.timestamp_s
                if dt > 0:
                    vx = (item.x_px - previous.x_px) / dt
                    vy = (item.y_px - previous.y_px) / dt
                    speed = hypot(vx, vy)
            points.append(
                BallTrackPoint(
                    frame_index=item.frame_index,
                    timestamp_s=item.timestamp_s,
                    x_px=item.x_px,
                    y_px=item.y_px,
                    vx_px_s=vx,
                    vy_px_s=vy,
                    speed_px_s=speed,
                    confidence=item.confidence,
                )
            )
        return points
