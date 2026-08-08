from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from beach_tennis_analyst.analytics.ball_report import build_ball_report
from beach_tennis_analyst.analytics.motion import MotionSummary, summarize_motion
from beach_tennis_analyst.analytics.pair import PairFrame, PairSummary, build_pair_frames, summarize_pair
from beach_tennis_analyst.ball.pipeline import BallAnalysisPipeline, BallPipelineOutputs
from beach_tennis_analyst.calibration.camera_standard import validate_rear_server_camera
from beach_tennis_analyst.calibration.court import BeachTennisCourt
from beach_tennis_analyst.calibration.depth_confidence import projection_confidence_for_y
from beach_tennis_analyst.calibration.manual import ManualCalibration
from beach_tennis_analyst.calibration.projection import CourtProjector
from beach_tennis_analyst.detection.player_detector import PlayerDetector, PlayerDetectorConfig
from beach_tennis_analyst.domain.models import Confidence, IdentityStatus, ObservationStatus, PlayerFrame
from beach_tennis_analyst.export.ball_events import export_ball_track, export_rallies, export_shots
from beach_tennis_analyst.export.trajectory import TrajectoryRecord, export_csv, export_jsonl
from beach_tennis_analyst.game.event_builder import build_shot_events
from beach_tennis_analyst.game.events import ShotEvent
from beach_tennis_analyst.ingestion.video_reader import VideoMetadata, VideoReader
from beach_tennis_analyst.render.video_outputs import render_annotated_match, render_movement_2d
from beach_tennis_analyst.review.clip_exporter import export_review_clips
from beach_tennis_analyst.tracking.confidence import FrameConfidence
from beach_tennis_analyst.tracking.four_player_tracker import (
    FourPlayerTracker,
    TrackingInitializationError,
)
from beach_tennis_analyst.tracking.trajectory import TrajectoryConfig, TrajectoryProcessor


@dataclass(frozen=True, slots=True)
class SessionOutputs:
    metadata: VideoMetadata
    trajectories: dict[str, list[PlayerFrame]]
    athlete_summaries: dict[str, MotionSummary]
    pair_frames: dict[str, list[PairFrame]]
    pair_summaries: dict[str, PairSummary]
    ball: BallPipelineOutputs
    shots: list[ShotEvent]
    output_dir: Path


class BeachTennisAnalysisPipeline:
    """Runs the rear-camera athlete-tracking homologation pipeline."""

    def __init__(
        self,
        *,
        detector_config: PlayerDetectorConfig | None = None,
        trajectory_config: TrajectoryConfig | None = None,
        max_initialization_frames: int = 300,
        track_near_only: bool = True,
        analyze_ball: bool = False,
    ) -> None:
        if max_initialization_frames <= 0:
            raise ValueError("max_initialization_frames must be positive")
        self.detector_config = detector_config or PlayerDetectorConfig()
        self.trajectory_config = trajectory_config or TrajectoryConfig()
        self.max_initialization_frames = max_initialization_frames
        self.track_near_only = track_near_only
        self.analyze_ball = analyze_ball

    def run(
        self,
        video_path: str | Path,
        calibration_path: str | Path,
        output_dir: str | Path,
    ) -> SessionOutputs:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        reader = VideoReader(video_path)
        metadata = reader.inspect()
        calibration = ManualCalibration.load(calibration_path)
        if (
            calibration.source_width_px != metadata.width_px
            or calibration.source_height_px != metadata.height_px
        ):
            raise ValueError("Calibration dimensions do not match video dimensions")

        camera_report = validate_rear_server_camera(calibration)
        if not camera_report.valid:
            raise ValueError(
                "Calibration does not match the rear-camera-behind-server standard: "
                + "; ".join(camera_report.warnings)
            )

        projector = CourtProjector(calibration, BeachTennisCourt())
        raw = self._track_players(reader, projector)
        required_ids = {"near_left", "near_right"}
        if not self.track_near_only:
            required_ids |= {"far_left", "far_right"}
        missing = required_ids - raw.keys()
        if missing:
            raise RuntimeError(f"Could not build required trajectories: {sorted(missing)}")

        processor = TrajectoryProcessor(self.trajectory_config)
        trajectories = {athlete_id: processor.process(frames) for athlete_id, frames in raw.items()}
        athlete_summaries = {
            athlete_id: summarize_motion(frames) for athlete_id, frames in trajectories.items()
        }
        pair_frames = {
            "near": build_pair_frames(
                trajectories["near_left"], trajectories["near_right"], team_side="near"
            )
        }
        if not self.track_near_only:
            pair_frames["far"] = build_pair_frames(
                trajectories["far_left"], trajectories["far_right"], team_side="far"
            )
        pair_summaries = {
            side: summarize_pair(frames) for side, frames in pair_frames.items() if frames
        }

        if self.analyze_ball:
            ball = BallAnalysisPipeline(projector).run(
                VideoReader(video_path).frames(),
                trajectories,
            )
            shots = build_shot_events(
                list(ball.contacts),
                list(ball.rallies),
                list(ball.track),
                projector,
            )
        else:
            ball = BallPipelineOutputs(
                detections=(),
                track=(),
                contacts=(),
                rallies=(),
            )
            shots = []

        self._export_outputs(
            output=output,
            video_path=video_path,
            metadata=metadata,
            projector=projector,
            trajectories=trajectories,
            athlete_summaries=athlete_summaries,
            pair_frames=pair_frames,
            pair_summaries=pair_summaries,
            ball=ball,
            shots=shots,
        )
        return SessionOutputs(
            metadata=metadata,
            trajectories=trajectories,
            athlete_summaries=athlete_summaries,
            pair_frames=pair_frames,
            pair_summaries=pair_summaries,
            ball=ball,
            shots=shots,
            output_dir=output,
        )

    def _track_players(
        self,
        reader: VideoReader,
        projector: CourtProjector,
    ) -> dict[str, list[PlayerFrame]]:
        detector = PlayerDetector(self.detector_config)
        tracker = FourPlayerTracker(projector, near_only=self.track_near_only)
        raw: dict[str, list[PlayerFrame]] = {}
        frames_seen = 0
        max_detector_candidates = 0
        last_initialization_error: str | None = None
        initialized = False

        for frame_index, timestamp_s, frame in reader.frames():
            frames_seen += 1
            detections = detector.detect(frame)

            if not initialized:
                max_detector_candidates = max(max_detector_candidates, len(detections))

            try:
                assignments = tracker.update(detections, frame_index)
            except TrackingInitializationError as exc:
                last_initialization_error = str(exc)
                if frames_seen >= self.max_initialization_frames:
                    target = "near-side two-player" if self.track_near_only else "four-player"
                    raise TrackingInitializationError(
                        f"Could not initialize {target} tracking within "
                        f"{self.max_initialization_frames} frames; "
                        f"max detector candidates in one frame={max_detector_candidates}; "
                        f"last initialization error: {last_initialization_error}"
                    ) from exc
                continue

            if len(tracker.tracks) == tracker.expected_track_count:
                initialized = True

            for assignment in assignments:
                identity_confidence = assignment.assignment_confidence
                projection_confidence = projection_confidence_for_y(assignment.y_m)
                confidence = Confidence(
                    detection=assignment.detection.confidence,
                    identity=identity_confidence,
                    projection=projection_confidence,
                    trajectory=identity_confidence,
                )
                identity_status = (
                    IdentityStatus.CONFIRMED
                    if identity_confidence >= 0.80
                    else IdentityStatus.PROBABLE
                    if identity_confidence >= 0.55
                    else IdentityStatus.UNCERTAIN
                )
                raw.setdefault(assignment.athlete_id, []).append(
                    PlayerFrame(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        athlete_id=assignment.athlete_id,
                        x_m=assignment.x_m,
                        y_m=assignment.y_m,
                        speed_mps=None,
                        acceleration_mps2=None,
                        observation_status=ObservationStatus.OBSERVED,
                        identity_status=identity_status,
                        confidence=confidence,
                    )
                )

        if not initialized:
            detail = last_initialization_error or "no valid initialization was observed"
            target = "near-side two-player" if self.track_near_only else "four-player"
            raise TrackingInitializationError(
                f"Video ended before {target} tracking could initialize; "
                f"frames inspected={frames_seen}; "
                f"max detector candidates in one frame={max_detector_candidates}; "
                f"last initialization error: {detail}"
            )

        return raw

    def _export_outputs(
        self,
        *,
        output: Path,
        video_path: str | Path,
        metadata: VideoMetadata,
        projector: CourtProjector,
        trajectories: dict[str, list[PlayerFrame]],
        athlete_summaries: dict[str, MotionSummary],
        pair_frames: dict[str, list[PairFrame]],
        pair_summaries: dict[str, PairSummary],
        ball: BallPipelineOutputs,
        shots: list[ShotEvent],
    ) -> None:
        records: list[TrajectoryRecord] = []
        for athlete_frames in trajectories.values():
            for frame in athlete_frames:
                records.append(
                    TrajectoryRecord(
                        frame_index=frame.frame_index,
                        timestamp_s=frame.timestamp_s,
                        athlete_id=frame.athlete_id,
                        x_m=frame.x_m,
                        y_m=frame.y_m,
                        detector_track_id=None,
                        observation_status=frame.observation_status.value,
                        identity_status=frame.identity_status.value,
                        confidence=FrameConfidence(
                            detection=frame.confidence.detection,
                            identity=frame.confidence.identity,
                            projection=frame.confidence.projection,
                            trajectory=frame.confidence.trajectory,
                        ),
                    )
                )
        records.sort(key=lambda item: (item.frame_index, item.athlete_id))
        export_csv(records, output / "trajectories.csv")
        export_jsonl(records, output / "trajectories.jsonl")

        if self.analyze_ball:
            export_ball_track(ball.track, output / "ball_track.jsonl")
            export_rallies(ball.rallies, output / "rallies.json")
            export_shots(shots, output / "shots.jsonl")
            ball_report: dict[str, object] = build_ball_report(shots)
            (output / "ball_report.json").write_text(
                json.dumps(ball_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            ball_report = {"enabled": False}
            for filename in (
                "ball_track.jsonl",
                "rallies.json",
                "shots.jsonl",
                "ball_report.json",
            ):
                (output / filename).unlink(missing_ok=True)
            review_dir = output / "review_clips"
            if review_dir.exists():
                shutil.rmtree(review_dir)

        payload = {
            "video": {
                "path": str(metadata.path),
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "width_px": metadata.width_px,
                "height_px": metadata.height_px,
                "duration_s": metadata.duration_s,
                "codec": metadata.codec,
            },
            "tracking_scope": "near_pair" if self.track_near_only else "all_four_players",
            "court_scope": "camera_side_of_net" if self.track_near_only else "full_court",
            "athletes": {
                athlete_id: asdict(summary) for athlete_id, summary in athlete_summaries.items()
            },
            "pairs": {side: asdict(summary) for side, summary in pair_summaries.items()},
            "ball_analysis": ball_report,
        }
        (output / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for side, frames in pair_frames.items():
            path = output / f"pair_{side}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for frame in frames:
                    handle.write(json.dumps(asdict(frame), ensure_ascii=False) + "\n")

        render_movement_2d(
            output_path=output / "movement_2d.mp4",
            fps=metadata.fps,
            frame_count=metadata.frame_count,
            trajectories=trajectories,
            ball_track=list(ball.track) if self.analyze_ball else [],
            projector=projector,
        )
        render_annotated_match(
            video_path=video_path,
            output_path=output / "annotated_match.mp4",
            trajectories=trajectories,
            ball_track=list(ball.track) if self.analyze_ball else [],
            shots=shots if self.analyze_ball else [],
            projector=projector,
        )
        if self.analyze_ball:
            export_review_clips(
                video_path=video_path,
                shots=shots,
                output_dir=output / "review_clips",
            )
