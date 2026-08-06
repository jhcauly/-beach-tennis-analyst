from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from beach_tennis_analyst.tracking.models import BoundingBox, Detection


class DetectorUnavailableError(RuntimeError):
    """Raised when the optional Ultralytics dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class PlayerDetectorConfig:
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.50
    tracker_config: str = "botsort.yaml"
    person_class_id: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")


class PlayerDetector:
    """Adapts Ultralytics person detections to the tracker domain model.

    Detector IDs are intentionally treated as hints. Stable athlete identity is
    owned by FourPlayerTracker, not by YOLO/BoT-SORT.
    """

    def __init__(self, config: PlayerDetectorConfig | None = None) -> None:
        self.config = config or PlayerDetectorConfig()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise DetectorUnavailableError(
                "Install the 'vision' extra to use PlayerDetector"
            ) from exc
        self._model: Any = YOLO(self.config.model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape HxWx3")

        result = self._model.track(
            frame,
            persist=True,
            classes=[self.config.person_class_id],
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            tracker=self.config.tracker_config,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            confidence = float(box.conf.item())
            if confidence < self.config.confidence_threshold:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            raw_track_id = None if box.id is None else int(box.id.item())
            detections.append(
                Detection(
                    detector_track_id=raw_track_id,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=confidence,
                )
            )
        return detections
