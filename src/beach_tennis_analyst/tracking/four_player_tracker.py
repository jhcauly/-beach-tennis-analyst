from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from beach_tennis_analyst.calibration.projection import CourtProjector

from .models import AthleteTrack, Detection, PlayerLane, TeamSide


class TrackingInitializationError(RuntimeError):
    """Raised when four valid on-court athletes cannot be initialized."""


@dataclass(frozen=True, slots=True)
class TrackAssignment:
    athlete_id: str
    detection: Detection
    x_m: float
    y_m: float
    assignment_confidence: float


class FourPlayerTracker:
    """Owns stable athlete identities independently from detector track IDs.

    This first slice implements deterministic initialization and conservative
    nearest-position association. Appearance-based re-identification and
    occlusion recovery will extend this class without changing its public IDs.
    """

    def __init__(
        self,
        projector: CourtProjector,
        maximum_match_distance_m: float = 2.5,
    ) -> None:
        if maximum_match_distance_m <= 0:
            raise ValueError("maximum_match_distance_m must be positive")
        self.projector = projector
        self.maximum_match_distance_m = maximum_match_distance_m
        self.tracks: dict[str, AthleteTrack] = {}

    def initialize(self, detections: Iterable[Detection], frame_index: int = 0) -> list[TrackAssignment]:
        projected = self._project_valid_detections(detections)
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

        # Extra detections can happen because of officials, bystanders or duplicate
        # detector boxes. For initialization, keep the two most confident person
        # detections on each side and then assign stable left/right identities by x.
        near = sorted(
            sorted(near_candidates, key=lambda item: item[0].confidence, reverse=True)[:2],
            key=lambda item: item[1],
        )
        far = sorted(
            sorted(far_candidates, key=lambda item: item[0].confidence, reverse=True)[:2],
            key=lambda item: item[1],
        )

        assignments: list[TrackAssignment] = []
        for side, group in ((TeamSide.NEAR, near), (TeamSide.FAR, far)):
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
        if len(self.tracks) != 4:
            return self.initialize(detections, frame_index)

        candidates = self._project_valid_detections(detections)
        unused = list(candidates)
        assignments: list[TrackAssignment] = []

        for athlete_id, track in sorted(self.tracks.items()):
            if track.last_position_m is None or not unused:
                track.missed_frames += 1
                continue

            same_side = [
                item for item in unused
                if self.projector.court.side_for_y(item[2]) == track.team_side.value
            ]
            if not same_side:
                track.missed_frames += 1
                continue

            previous_x, previous_y = track.last_position_m
            detection, x_m, y_m = min(
                same_side,
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

    def _project_valid_detections(self, detections: Iterable[Detection]) -> list[tuple[Detection, float, float]]:
        projected: list[tuple[Detection, float, float]] = []
        for detection in detections:
            x_px, y_px = detection.bbox.foot_point
            point = self.projector.pixel_to_metric(x_px, y_px)
            if self.projector.court.contains(point.x_m, point.y_m, margin_m=0.25):
                projected.append((detection, point.x_m, point.y_m))
        return projected
