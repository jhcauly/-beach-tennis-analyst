from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from beach_tennis_analyst.ball.contact import detect_contacts
from beach_tennis_analyst.ball.detector import BallDetector
from beach_tennis_analyst.ball.tracking import BallDetection, BallTrackPoint, BallTracker
from beach_tennis_analyst.ball.trajectory import interpolate_short_gaps, smooth_track
from beach_tennis_analyst.calibration.projection import CourtProjector
from beach_tennis_analyst.domain.models import PlayerFrame
from beach_tennis_analyst.game.events import BallContact
from beach_tennis_analyst.game.rallies import RallyWindow, segment_rallies


@dataclass(frozen=True, slots=True)
class BallPipelineConfig:
    maximum_prediction_error_px: float = 90.0
    court_margin_m: float = 1.0


@dataclass(frozen=True, slots=True)
class BallPipelineOutputs:
    detections: tuple[BallDetection, ...]
    track: tuple[BallTrackPoint, ...]
    contacts: tuple[BallContact, ...]
    rallies: tuple[RallyWindow, ...]


class BallAnalysisPipeline:
    """Connects real video frames to one conservative ball trajectory and rally events."""

    def __init__(
        self,
        projector: CourtProjector,
        *,
        detector: BallDetector | None = None,
        tracker: BallTracker | None = None,
        config: BallPipelineConfig | None = None,
    ) -> None:
        self.projector = projector
        self.detector = detector or BallDetector()
        self.tracker = tracker or BallTracker()
        self.config = config or BallPipelineConfig()

    def run(
        self,
        frames: Iterable[tuple[int, float, object]],
        player_frames: dict[str, list[PlayerFrame]],
    ) -> BallPipelineOutputs:
        selected: list[BallDetection] = []
        for frame_index, timestamp_s, frame in frames:
            candidates = self.detector.detect(
                frame,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
            )
            candidate = self._select_candidate(candidates, selected)
            if candidate is not None:
                selected.append(candidate)

        raw_track = self.tracker.build(selected)
        continuous = smooth_track(interpolate_short_gaps(raw_track))
        contacts = detect_contacts(
            continuous,
            player_frames,
            pixel_to_metric=self.projector.pixel_to_metric,
        )
        rallies = segment_rallies(contacts)
        return BallPipelineOutputs(
            detections=tuple(selected),
            track=tuple(continuous),
            contacts=tuple(contacts),
            rallies=tuple(rallies),
        )

    def _select_candidate(
        self,
        candidates: list[BallDetection],
        selected: list[BallDetection],
    ) -> BallDetection | None:
        valid = [candidate for candidate in candidates if self._inside_extended_court(candidate)]
        if not valid:
            return None
        if not selected:
            return max(valid, key=lambda item: item.confidence)

        previous = selected[-1]
        predicted_x = previous.x_px
        predicted_y = previous.y_px
        if len(selected) >= 2:
            before = selected[-2]
            predicted_x += previous.x_px - before.x_px
            predicted_y += previous.y_px - before.y_px

        ranked = sorted(
            valid,
            key=lambda item: (
                hypot(item.x_px - predicted_x, item.y_px - predicted_y),
                -item.confidence,
            ),
        )
        best = ranked[0]
        error = hypot(best.x_px - predicted_x, best.y_px - predicted_y)
        if error > self.config.maximum_prediction_error_px:
            return None
        return best

    def _inside_extended_court(self, candidate: BallDetection) -> bool:
        point = self.projector.pixel_to_metric(candidate.x_px, candidate.y_px)
        margin = self.config.court_margin_m
        return (
            -margin <= point.x_m <= 8.0 + margin
            and -margin <= point.y_m <= 16.0 + margin
        )
