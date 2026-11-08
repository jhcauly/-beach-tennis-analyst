from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import hypot
from typing import Iterable

from beach_tennis_analyst.calibration.projection import CourtProjector

from .confidence import distance_confidence, identity_confidence
from .models import AthleteTrack, Detection


@dataclass(frozen=True, slots=True)
class CandidateAssignment:
    athlete_id: str
    detection: Detection
    x_m: float
    y_m: float
    distance_m: float
    identity_confidence: float


class IdentityGuard:
    """Finds a conservative one-to-one assignment for athletes on one court side.

    It evaluates all permutations because each side has only two athletes in the
    current product scope. This is intentionally explicit and auditable, rather
    than relying on a greedy nearest-neighbour match that can silently swap IDs.
    """

    def __init__(self, projector: CourtProjector, maximum_distance_m: float) -> None:
        self.projector = projector
        self.maximum_distance_m = maximum_distance_m

    def assign_side(
        self,
        tracks: Iterable[AthleteTrack],
        detections: Iterable[Detection],
    ) -> list[CandidateAssignment]:
        track_list = [track for track in tracks if track.last_position_m is not None]
        projected = []
        for detection in detections:
            x_px, y_px = detection.bbox.foot_point
            point = self.projector.pixel_to_metric(x_px, y_px)
            projected.append((detection, point.x_m, point.y_m))

        if not track_list or not projected:
            return []

        best: tuple[float, list[CandidateAssignment]] | None = None
        for chosen in permutations(projected, min(len(track_list), len(projected))):
            assignments: list[CandidateAssignment] = []
            total_cost = 0.0
            valid = True
            for track, (detection, x_m, y_m) in zip(track_list, chosen, strict=False):
                previous_x, previous_y = track.last_position_m or (x_m, y_m)
                distance = hypot(x_m - previous_x, y_m - previous_y)
                if distance > self.maximum_distance_m:
                    valid = False
                    break
                distance_score = distance_confidence(distance, self.maximum_distance_m)
                id_seen = (
                    detection.detector_track_id is not None
                    and detection.detector_track_id in track.detector_ids
                )
                score = identity_confidence(
                    detector_id_seen_before=id_seen,
                    side_consistent=True,
                    distance_score=distance_score,
                    missed_frames=track.missed_frames,
                )
                # Penalize crossing the relative left/right order unless evidence is strong.
                lane_penalty = 0.0
                if track.initial_lane.value == "left" and x_m > 4.8:
                    lane_penalty = 0.08
                elif track.initial_lane.value == "right" and x_m < 3.2:
                    lane_penalty = 0.08
                total_cost += (1.0 - score) + lane_penalty
                assignments.append(
                    CandidateAssignment(
                        athlete_id=track.athlete_id,
                        detection=detection,
                        x_m=x_m,
                        y_m=y_m,
                        distance_m=distance,
                        identity_confidence=max(0.0, score - lane_penalty),
                    )
                )
            if valid and (best is None or total_cost < best[0]):
                best = (total_cost, assignments)

        if best is None:
            return []

        assignments = best[1]
        if len(assignments) == 2:
            first, second = assignments
            # Reject an ambiguous swap when both identities are weak.
            if first.identity_confidence < 0.55 and second.identity_confidence < 0.55:
                return []
        return assignments
