from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from beach_tennis_analyst.calibration.projection import CourtProjector

from .models import AthleteTrack, Detection, PlayerLane, TeamSide


class TrackingInitializationError(RuntimeError):
    """Raised when the required valid on-court athletes cannot be initialized."""


@dataclass(frozen=True, slots=True)
class TrackAssignment:
    athlete_id: str
    detection: Detection
    x_m: float
    y_m: float
    assignment_confidence: float


class FourPlayerTracker:
    """Owns stable athlete identities independently from detector track IDs.

    In near-only homologation mode, only athletes on the camera-side half of the
    court are eligible. Detections beyond the calibrated net line are rejected so
    a lost near-side identity cannot jump to an athlete on the far side.
    """

    def __init__(
        self,
        projector: CourtProjector,
        maximum_match_distance_m: float = 2.5,
        near_only: bool = False,
        near_side_tolerance_m: float = 0.35,
    ) -> None:
        if maximum_match_distance_m <= 0:
            raise ValueError("maximum_match_distance_m must be positive")
        if near_side_tolerance_m < 0:
            raise ValueError("near_side_tolerance_m cannot be negative")
        self.projector = projector
        self.maximum_match_distance_m = maximum_match_distance_m
        self.near_only = near_only
        self.near_side_tolerance_m = near_side_tolerance_m
        self.tracks: dict[str, AthleteTrack] = {}

    @property
    def expected_track_count(self) -> int:
        return 2 if self.near_only else 4

    @property
    def near_side_max_y_m(self) -> float:
        return self.projector.court.net_y_m + self.near_side_tolerance_m

    def initialize(self, detections: Iterable[Detection], frame_index: int = 0) -> list[TrackAssignment]:
        projected = self._project_valid_detections(detections)

        if self.near_only:
            near_candidates = [item for item in projected if self._is_near_side(item[2])]
            if len(near_candidates) < 2:
                raise TrackingInitializationError(
                    "Expected at least two athlete detections on the camera-side half of the court, "
                    f"got near={len(near_candidates)}, total_on_court={len(projected)}"
                )

            # The homologation target is the near-side pair. Once the net line is
            # used as a hard eligibility boundary, confidence is a safer initializer
            # than taking candidates from the far half by depth.
            near = sorted(
                sorted(near_candidates, key=lambda item: item[0].confidence, reverse=True)[:2],
                key=lambda item: item[1],
            )
            groups: list[tuple[TeamSide, list[tuple[Detection, float, float]]]] = [
                (TeamSide.NEAR, near)
            ]
        else:
            near_candidates = [
                item for item in projected if item[2] < self.projector.court.net_y_m
            ]
            far_candidates = [
                item for item in projected if item[2] >= self.projector.court.net_y_m
            ]
            if len(near_candidates) < 2 or len(far_candidates) < 2:
                raise TrackingInitializationError(
                    "Expected at least two athletes on each side of the net, "
                    f"got near={len(near_candidates)}, far={len(far_candidates)}, "
                    f"total_on_court={len(projected)}"
                )
            near = sorted(
                sorted(near_candidates, key=lambda item: item[0].confidence, reverse=True)[:2],
                key=lambda item: item[1],
            )
            far = sorted(
                sorted(far_candidates, key=lambda item: item[0].confidence, reverse=True)[:2],
                key=lambda item: item[1],
            )
            groups = [(TeamSide.NEAR, near), (TeamSide.FAR, far)]

        assignments: list[TrackAssignment] = []
        for side, group in groups:
            for lane, item in zip((PlayerLane.LEFT, PlayerLane.RIGHT), group, strict=True):
                detection, x_m, y_m = item
                athlete_id = f"{side.value}_{lane.value}"
                track = AthleteTrack(
                    athlete_id=athlete_id,
                    team_side=side,
                    initial_lane=lane,
                    last_frame_index=frame_index,
                    last_position_m=(x_m, y_m),
                )
                track.register_detector_id(detection.detector_track_id)
                self.tracks[athlete_id] = track
                assignments.append(
                    TrackAssignment(
                        athlete_id=athlete_id,
                        detection=detection,
                        x_m=x_m,
                        y_m=y_m,
                        assignment_confidence=detection.confidence,
                    )
                )
        return assignments

    def update(self, detections: Iterable[Detection], frame_index: int) -> list[TrackAssignment]:
        if len(self.tracks) != self.expected_track_count:
            return self.initialize(detections, frame_index)

        candidates = self._project_valid_detections(detections)
        unused = list(candidates)
        assignments: list[TrackAssignment] = []

        for athlete_id, track in sorted(self.tracks.items()):
            if track.last_position_m is None or not unused:
                track.missed_frames += 1
                continue

            if self.near_only:
                eligible = [item for item in unused if self._is_near_side(item[2])]
            else:
                eligible = [
                    item for item in unused
                    if self.projector.court.side_for_y(item[2]) == track.team_side.value
                ]
            if not eligible:
                track.missed_frames += 1
                continue

            previous_x, previous_y = track.last_position_m
            detection, x_m, y_m = min(
                eligible,
                key=lambda item: hypot(item[1] - previous_x, item[2] - previous_y),
            )
            distance = hypot(x_m - previous_x, y_m - previous_y)
            if distance > self.maximum_match_distance_m:
                track.missed_frames += 1
                continue

            unused.remove((detection, x_m, y_m))
            track.last_position_m = (x_m, y_m)
            track.last_frame_index = frame_index
            track.missed_frames = 0
            track.register_detector_id(detection.detector_track_id)
            proximity_confidence = max(0.0, 1.0 - distance / self.maximum_match_distance_m)
            assignments.append(
                TrackAssignment(
                    athlete_id=athlete_id,
                    detection=detection,
                    x_m=x_m,
                    y_m=y_m,
                    assignment_confidence=(detection.confidence + proximity_confidence) / 2.0,
                )
            )
        return assignments

    def _is_near_side(self, y_m: float) -> bool:
        return y_m <= self.near_side_max_y_m

    def _project_valid_detections(self, detections: Iterable[Detection]) -> list[tuple[Detection, float, float]]:
        projected: list[tuple[Detection, float, float]] = []
        for detection in detections:
            x_px, y_px = detection.bbox.foot_point
            point = self.projector.pixel_to_metric(x_px, y_px)
            if self.projector.court.contains(point.x_m, point.y_m, margin_m=0.25):
                projected.append((detection, point.x_m, point.y_m))
        return projected
