from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import sqrt


class TeamSide(StrEnum):
    NEAR = "near"
    FAR = "far"


class PlayerLane(StrEnum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Invalid bounding box")

    @property
    def foot_point(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True, slots=True)
class Detection:
    detector_track_id: int | None
    bbox: BoundingBox
    confidence: float
    appearance_vector: tuple[float, ...] | None = None


@dataclass(slots=True)
class AthleteTrack:
    athlete_id: str
    team_side: TeamSide
    initial_lane: PlayerLane
    last_frame_index: int = -1
    last_position_m: tuple[float, float] | None = None
    missed_frames: int = 0
    detector_ids: set[int] = field(default_factory=set)
    appearance_vector: tuple[float, ...] | None = None

    def register_detector_id(self, detector_track_id: int | None) -> None:
        if detector_track_id is not None:
            self.detector_ids.add(detector_track_id)

    def update_appearance(self, vector: tuple[float, ...] | None, alpha: float = 0.20) -> None:
        if vector is None:
            return
        if self.appearance_vector is None or len(self.appearance_vector) != len(vector):
            self.appearance_vector = vector
            return
        mixed = [
            (1.0 - alpha) * old + alpha * new
            for old, new in zip(self.appearance_vector, vector, strict=True)
        ]
        norm = sqrt(sum(value * value for value in mixed))
        if norm > 0:
            mixed = [value / norm for value in mixed]
        self.appearance_vector = tuple(mixed)
