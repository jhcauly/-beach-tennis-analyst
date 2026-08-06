from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from beach_tennis_analyst.analytics.motion import MotionSummary, summarize_motion
from beach_tennis_analyst.analytics.pair import PairFrame, PairSummary, build_pair_frames, summarize_pair
from beach_tennis_analyst.calibration.camera_standard import validate_rear_server_camera
from beach_tennis_analyst.calibration.court import BeachTennisCourt
from beach_tennis_analyst.calibration.depth_confidence import projection_confidence_for_y
from beach_tennis_analyst.calibration.manual import ManualCalibration
from beach_tennis_analyst.calibration.projection import CourtProjector
from beach_tennis_analyst.detection.player_detector import PlayerDetector, PlayerDetectorConfig
from beach_tennis_analyst.domain.models import Confidence, IdentityStatus, ObservationStatus, PlayerFrame
from beach_tennis_analyst.export.trajectory import TrajectoryRecord, export_csv, export_jsonl
from beach_tennis_analyst.ingestion.video_reader import VideoMetadata, VideoReader
from beach_tennis_analyst.tracking.confidence import FrameConfidence
from beach_tennis_analyst.tracking.four_player_tracker import FourPlayerTracker
from beach_tennis_analyst.tracking.trajectory import TrajectoryConfig, TrajectoryProcessor


@dataclass(frozen=True, slots=True)
class SessionOutputs:
    metadata: VideoMetadata
    trajectories: dict[str, list[PlayerFrame]]
    athlete_summaries: dict[str, MotionSummary]
    pair_frames: dict[str, list[PairFrame]]
    pair_summaries: dict[str, PairSummary]
    output_dir: Path


class BeachTennisAnalysisPipeline:
    """Runs the rear-camera M1 pipeline from video to spatial summaries."""

    def __init__(
        self,
        *,
        detector_config: PlayerDetectorConfig | None = None,
        trajectory_config: TrajectoryConfig | None = None,
    ) -> None:
        self.detector_config = detector_config or PlayerDetectorConfig()
        self.trajectory_config = trajectory_config or TrajectoryConfig()

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
        detector = PlayerDetector(self.detector_config)
        tracker = FourPlayerTracker(projector)
        raw: dict[str, list[PlayerFrame]] = {}

        for frame_index, timestamp_s, frame in reader.frames():
            detections = detector.detect(frame)
            assignments = tracker.update(detections, frame_index)
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

        required_ids = {"near_left", "near_right", "far_left", "far_right"}
        missing = required_ids - raw.keys()
        if missing:
            raise RuntimeError(f"Could not build all four trajectories: {sorted(missing)}")

        processor = TrajectoryProcessor(self.trajectory_config)
        trajectories = {athlete_id: processor.process(frames) for athlete_id, frames in raw.items()}
        athlete_summaries = {
            athlete_id: summarize_motion(frames) for athlete_id, frames in trajectories.items()
        }

        pair_frames = {
            "near": build_pair_frames(
                trajectories["near_left"], trajectories["near_right"], team_side="near"
            ),
            "far": build_pair_frames(
                trajectories["far_left"], trajectories["far_right"], team_side="far"
            ),
        }
        pair_summaries = {
            side: summarize_pair(frames) for side, frames in pair_frames.items() if frames
        }

        self._export_outputs(
            output,
            metadata,
            trajectories,
            athlete_summaries,
            pair_frames,
            pair_summaries,
        )
        return SessionOutputs(
            metadata=metadata,
            trajectories=trajectories,
            athlete_summaries=athlete_summaries,
            pair_frames=pair_frames,
            pair_summaries=pair_summaries,
            output_dir=output,
        )

    def _export_outputs(
        self,
        output: Path,
        metadata: VideoMetadata,
        trajectories: dict[str, list[PlayerFrame]],
        athlete_summaries: dict[str, MotionSummary],
        pair_frames: dict[str, list[PairFrame]],
        pair_summaries: dict[str, PairSummary],
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
            "athletes": {
                athlete_id: asdict(summary) for athlete_id, summary in athlete_summaries.items()
            },
            "pairs": {side: asdict(summary) for side, summary in pair_summaries.items()},
        }
        (output / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for side, frames in pair_frames.items():
            path = output / f"pair_{side}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for frame in frames:
                    handle.write(json.dumps(asdict(frame), ensure_ascii=False) + "\n")
