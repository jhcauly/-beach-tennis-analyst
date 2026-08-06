from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True, slots=True)
class FrameConfidence:
    detection: float
    identity: float
    projection: float
    trajectory: float

    def __post_init__(self) -> None:
        for name, value in (
            ("detection", self.detection),
            ("identity", self.identity),
            ("projection", self.projection),
            ("trajectory", self.trajectory),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} confidence must be in [0, 1]")

    @property
    def overall(self) -> float:
        # Geometric-style balance: one weak component should matter.
        product = self.detection * self.identity * self.projection * self.trajectory
        return product ** 0.25


def distance_confidence(distance_m: float, maximum_distance_m: float) -> float:
    if distance_m < 0 or maximum_distance_m <= 0:
        raise ValueError("invalid distance confidence inputs")
    ratio = distance_m / maximum_distance_m
    return max(0.0, min(1.0, exp(-2.0 * ratio * ratio)))


def identity_confidence(
    *,
    detector_id_seen_before: bool,
    side_consistent: bool,
    distance_score: float,
    missed_frames: int,
) -> float:
    base = 0.35 + 0.35 * distance_score
    if detector_id_seen_before:
        base += 0.20
    if side_consistent:
        base += 0.10
    base -= min(0.35, missed_frames * 0.04)
    return max(0.0, min(1.0, base))
