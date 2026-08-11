from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ObservationStatus(StrEnum):
    OBSERVED = "observed"
    SMOOTHED = "smoothed"
    INTERPOLATED = "interpolated"
    SUSPECT = "suspect"
    REJECTED = "rejected"


class IdentityStatus(StrEnum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"
    OCCLUDED = "occluded"
    LOST = "lost"
    REIDENTIFIED = "reidentified"


@dataclass(frozen=True, slots=True)
class Confidence:
    detection: float
    identity: float
    projection: float
    trajectory: float

    @property
    def overall(self) -> float:
        values = (self.detection, self.identity, self.projection, self.trajectory)
        return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class PlayerFrame:
    frame_index: int
    timestamp_s: float
    athlete_id: str
    x_m: float
    y_m: float
    speed_mps: float | None
    acceleration_mps2: float | None
    observation_status: ObservationStatus
    identity_status: IdentityStatus
    confidence: Confidence
