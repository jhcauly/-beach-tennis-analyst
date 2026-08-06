from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from statistics import median
from typing import Iterable

from beach_tennis_analyst.domain.models import (
    Confidence,
    IdentityStatus,
    ObservationStatus,
    PlayerFrame,
)


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    smoothing_window: int = 5
    max_interpolation_gap_frames: int = 5
    max_speed_mps: float = 9.0
    max_acceleration_mps2: float = 16.0
    minimum_confidence: float = 0.35

    def __post_init__(self) -> None:
        if self.smoothing_window < 1 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd number")
        if self.max_interpolation_gap_frames < 0:
            raise ValueError("max_interpolation_gap_frames must be non-negative")
        if self.max_speed_mps <= 0 or self.max_acceleration_mps2 <= 0:
            raise ValueError("physical limits must be positive")


@dataclass(frozen=True, slots=True)
class PredictedPosition:
    x_m: float
    y_m: float
    confidence: float


class ConstantVelocityPredictor:
    """Predicts a short missing segment from the two latest valid observations."""

    def predict(self, history: list[PlayerFrame], timestamp_s: float) -> PredictedPosition | None:
        valid = [
            frame
            for frame in history
            if frame.observation_status not in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}
            and frame.confidence.overall > 0.0
        ]
        if len(valid) < 2:
            return None
        previous, latest = valid[-2], valid[-1]
        dt = latest.timestamp_s - previous.timestamp_s
        future_dt = timestamp_s - latest.timestamp_s
        if dt <= 0.0 or future_dt <= 0.0:
            return None
        vx = (latest.x_m - previous.x_m) / dt
        vy = (latest.y_m - previous.y_m) / dt
        decay = max(0.0, 1.0 - future_dt / 0.5)
        return PredictedPosition(
            x_m=latest.x_m + vx * future_dt,
            y_m=latest.y_m + vy * future_dt,
            confidence=latest.confidence.overall * decay,
        )


class TrajectoryProcessor:
    def __init__(self, config: TrajectoryConfig | None = None) -> None:
        self.config = config or TrajectoryConfig()

    def process(self, frames: Iterable[PlayerFrame]) -> list[PlayerFrame]:
        ordered = sorted(frames, key=lambda item: item.frame_index)
        if not ordered:
            return []
        validated = self._reject_physical_jumps(ordered)
        interpolated = self._interpolate_short_gaps(validated)
        smoothed = self._median_smooth(interpolated)
        return self._derive_motion(smoothed)

    def _reject_physical_jumps(self, frames: list[PlayerFrame]) -> list[PlayerFrame]:
        output: list[PlayerFrame] = []
        previous_valid: PlayerFrame | None = None
        previous_speed: float | None = None
        for frame in frames:
            if frame.confidence.overall < self.config.minimum_confidence:
                output.append(replace(frame, observation_status=ObservationStatus.SUSPECT))
                continue
            if previous_valid is None:
                output.append(frame)
                previous_valid = frame
                continue
            dt = frame.timestamp_s - previous_valid.timestamp_s
            if dt <= 0.0:
                output.append(replace(frame, observation_status=ObservationStatus.REJECTED))
                continue
            distance = hypot(frame.x_m - previous_valid.x_m, frame.y_m - previous_valid.y_m)
            speed = distance / dt
            acceleration = 0.0 if previous_speed is None else abs(speed - previous_speed) / dt
            if speed > self.config.max_speed_mps or acceleration > self.config.max_acceleration_mps2:
                output.append(replace(frame, observation_status=ObservationStatus.SUSPECT))
                continue
            output.append(frame)
            previous_valid = frame
            previous_speed = speed
        return output

    def _interpolate_short_gaps(self, frames: list[PlayerFrame]) -> list[PlayerFrame]:
        output = list(frames)
        index = 0
        while index < len(output):
            if output[index].observation_status not in {ObservationStatus.SUSPECT, ObservationStatus.REJECTED}:
                index += 1
                continue
            start = index
            while index < len(output) and output[index].observation_status in {
                ObservationStatus.SUSPECT,
                ObservationStatus.REJECTED,
            }:
                index += 1
            end = index - 1
            gap_size = end - start + 1
            if (
                start == 0
                or index >= len(output)
                or gap_size > self.config.max_interpolation_gap_frames
            ):
                continue
            left = output[start - 1]
            right = output[index]
            total_dt = right.timestamp_s - left.timestamp_s
            if total_dt <= 0.0:
                continue
            for gap_index in range(start, index):
                current = output[gap_index]
                ratio = (current.timestamp_s - left.timestamp_s) / total_dt
                confidence = Confidence(
                    detection=0.0,
                    identity=min(left.confidence.identity, right.confidence.identity) * 0.75,
                    projection=min(left.confidence.projection, right.confidence.projection),
                    trajectory=min(left.confidence.trajectory, right.confidence.trajectory) * 0.65,
                )
                output[gap_index] = replace(
                    current,
                    x_m=left.x_m + (right.x_m - left.x_m) * ratio,
                    y_m=left.y_m + (right.y_m - left.y_m) * ratio,
                    speed_mps=None,
                    acceleration_mps2=None,
                    observation_status=ObservationStatus.INTERPOLATED,
                    identity_status=IdentityStatus.PROBABLE,
                    confidence=confidence,
                )
        return output

    def _median_smooth(self, frames: list[PlayerFrame]) -> list[PlayerFrame]:
        radius = self.config.smoothing_window // 2
        output: list[PlayerFrame] = []
        for index, frame in enumerate(frames):
            if frame.observation_status in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}:
                output.append(frame)
                continue
            window = [
                candidate
                for candidate in frames[max(0, index - radius) : min(len(frames), index + radius + 1)]
                if candidate.observation_status not in {
                    ObservationStatus.REJECTED,
                    ObservationStatus.SUSPECT,
                }
            ]
            x_m = median(candidate.x_m for candidate in window)
            y_m = median(candidate.y_m for candidate in window)
            status = (
                frame.observation_status
                if frame.observation_status == ObservationStatus.INTERPOLATED
                else ObservationStatus.SMOOTHED
            )
            output.append(replace(frame, x_m=x_m, y_m=y_m, observation_status=status))
        return output

    def _derive_motion(self, frames: list[PlayerFrame]) -> list[PlayerFrame]:
        output: list[PlayerFrame] = []
        previous: PlayerFrame | None = None
        previous_speed: float | None = None
        for frame in frames:
            if frame.observation_status in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}:
                output.append(replace(frame, speed_mps=None, acceleration_mps2=None))
                continue
            if previous is None:
                output.append(replace(frame, speed_mps=0.0, acceleration_mps2=0.0))
                previous = output[-1]
                previous_speed = 0.0
                continue
            dt = frame.timestamp_s - previous.timestamp_s
            if dt <= 0.0:
                output.append(replace(frame, speed_mps=None, acceleration_mps2=None))
                continue
            speed = hypot(frame.x_m - previous.x_m, frame.y_m - previous.y_m) / dt
            acceleration = 0.0 if previous_speed is None else (speed - previous_speed) / dt
            current = replace(frame, speed_mps=speed, acceleration_mps2=acceleration)
            output.append(current)
            previous = current
            previous_speed = speed
        return output
