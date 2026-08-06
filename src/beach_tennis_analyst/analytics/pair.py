from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from beach_tennis_analyst.domain.models import ObservationStatus, PlayerFrame


@dataclass(frozen=True, slots=True)
class PairFrame:
    frame_index: int
    timestamp_s: float
    team_side: str
    left_athlete_id: str
    right_athlete_id: str
    centroid_x_m: float
    centroid_y_m: float
    partner_distance_m: float
    lateral_opening_m: float
    depth_misalignment_m: float
    is_too_open: bool
    is_depth_misaligned: bool


@dataclass(frozen=True, slots=True)
class PairSummary:
    team_side: str
    valid_samples: int
    average_partner_distance_m: float
    maximum_partner_distance_m: float
    average_lateral_opening_m: float
    maximum_lateral_opening_m: float
    average_depth_misalignment_m: float
    maximum_depth_misalignment_m: float
    too_open_ratio: float
    depth_misaligned_ratio: float


def build_pair_frames(
    left_frames: Iterable[PlayerFrame],
    right_frames: Iterable[PlayerFrame],
    *,
    team_side: str,
    too_open_threshold_m: float = 5.0,
    depth_misalignment_threshold_m: float = 2.0,
) -> list[PairFrame]:
    left_by_frame = {
        item.frame_index: item
        for item in left_frames
        if item.observation_status not in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}
    }
    right_by_frame = {
        item.frame_index: item
        for item in right_frames
        if item.observation_status not in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}
    }
    frames: list[PairFrame] = []
    for frame_index in sorted(left_by_frame.keys() & right_by_frame.keys()):
        left = left_by_frame[frame_index]
        right = right_by_frame[frame_index]
        lateral = abs(right.x_m - left.x_m)
        depth = abs(right.y_m - left.y_m)
        distance = hypot(right.x_m - left.x_m, right.y_m - left.y_m)
        frames.append(
            PairFrame(
                frame_index=frame_index,
                timestamp_s=max(left.timestamp_s, right.timestamp_s),
                team_side=team_side,
                left_athlete_id=left.athlete_id,
                right_athlete_id=right.athlete_id,
                centroid_x_m=(left.x_m + right.x_m) / 2.0,
                centroid_y_m=(left.y_m + right.y_m) / 2.0,
                partner_distance_m=distance,
                lateral_opening_m=lateral,
                depth_misalignment_m=depth,
                is_too_open=lateral > too_open_threshold_m,
                is_depth_misaligned=depth > depth_misalignment_threshold_m,
            )
        )
    return frames


def summarize_pair(frames: Iterable[PairFrame]) -> PairSummary:
    items = list(frames)
    if not items:
        raise ValueError("At least one pair frame is required")
    count = len(items)
    return PairSummary(
        team_side=items[0].team_side,
        valid_samples=count,
        average_partner_distance_m=sum(item.partner_distance_m for item in items) / count,
        maximum_partner_distance_m=max(item.partner_distance_m for item in items),
        average_lateral_opening_m=sum(item.lateral_opening_m for item in items) / count,
        maximum_lateral_opening_m=max(item.lateral_opening_m for item in items),
        average_depth_misalignment_m=sum(item.depth_misalignment_m for item in items) / count,
        maximum_depth_misalignment_m=max(item.depth_misalignment_m for item in items),
        too_open_ratio=sum(item.is_too_open for item in items) / count,
        depth_misaligned_ratio=sum(item.is_depth_misaligned for item in items) / count,
    )
