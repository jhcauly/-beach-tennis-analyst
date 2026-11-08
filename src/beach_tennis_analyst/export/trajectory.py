from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from beach_tennis_analyst.tracking.confidence import FrameConfidence


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    frame_index: int
    timestamp_s: float
    athlete_id: str
    x_m: float
    y_m: float
    detector_track_id: int | None
    observation_status: str
    identity_status: str
    confidence: FrameConfidence

    def to_flat_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "athlete_id": self.athlete_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "detector_track_id": self.detector_track_id,
            "observation_status": self.observation_status,
            "identity_status": self.identity_status,
            "detection_confidence": self.confidence.detection,
            "identity_confidence": self.confidence.identity,
            "projection_confidence": self.confidence.projection,
            "trajectory_confidence": self.confidence.trajectory,
            "overall_confidence": self.confidence.overall,
        }


def export_csv(records: Iterable[TrajectoryRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_flat_dict() for record in records]
    if not rows:
        raise ValueError("cannot export an empty trajectory")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def export_jsonl(records: Iterable[TrajectoryRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            payload["confidence"]["overall"] = record.confidence.overall
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            wrote_any = True
    if not wrote_any:
        output.unlink(missing_ok=True)
        raise ValueError("cannot export an empty trajectory")
    return output
