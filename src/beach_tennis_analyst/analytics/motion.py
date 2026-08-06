from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from beach_tennis_analyst.domain.models import ObservationStatus, PlayerFrame


@dataclass(frozen=True, slots=True)
class MotionSummary:
    athlete_id: str
    valid_samples: int
    total_distance_m: float
    duration_s: float
    average_speed_mps: float
    maximum_speed_mps: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float


def summarize_motion(frames: Iterable[PlayerFrame]) -> MotionSummary:
    ordered = sorted(frames, key=lambda item: item.frame_index)
    if not ordered:
        raise ValueError("At least one frame is required")

    athlete_ids = {frame.athlete_id for frame in ordered}
    if len(athlete_ids) != 1:
        raise ValueError("Motion summary requires frames from exactly one athlete")

    valid = [
        frame
        for frame in ordered
        if frame.observation_status not in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}
    ]
    if not valid:
        return MotionSummary(
            athlete_id=ordered[0].athlete_id,
            valid_samples=0,
            total_distance_m=0.0,
            duration_s=0.0,
            average_speed_mps=0.0,
            maximum_speed_mps=0.0,
            maximum_acceleration_mps2=0.0,
            maximum_deceleration_mps2=0.0,
        )

    total_distance = 0.0
    for previous, current in zip(valid, valid[1:]):
        total_distance += hypot(current.x_m - previous.x_m, current.y_m - previous.y_m)

    duration = max(0.0, valid[-1].timestamp_s - valid[0].timestamp_s)
    speeds = [frame.speed_mps for frame in valid if frame.speed_mps is not None]
    accelerations = [
        frame.acceleration_mps2 for frame in valid if frame.acceleration_mps2 is not None
    ]

    return MotionSummary(
        athlete_id=valid[0].athlete_id,
        valid_samples=len(valid),
        total_distance_m=total_distance,
        duration_s=duration,
        average_speed_mps=0.0 if duration == 0.0 else total_distance / duration,
        maximum_speed_mps=max(speeds, default=0.0),
        maximum_acceleration_mps2=max(accelerations, default=0.0),
        maximum_deceleration_mps2=min(accelerations, default=0.0),
    )
