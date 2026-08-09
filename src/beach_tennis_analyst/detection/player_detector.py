from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from beach_tennis_analyst.tracking.models import BoundingBox, Detection


class DetectorUnavailableError(RuntimeError):
    """Raised when the optional Ultralytics dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class PlayerDetectorConfig:
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.50
    tracker_config: str = "botsort.yaml"
    person_class_id: int = 0
    image_size: int = 1280

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")


class PlayerDetector:
    """Detects people and attaches a clothing-based appearance signature."""

    def __init__(self, config: PlayerDetectorConfig | None = None) -> None:
        self.config = config or PlayerDetectorConfig()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise DetectorUnavailableError(
                "Install the 'vision' extra to use PlayerDetector"
            ) from exc
        self._model: Any = YOLO(self.config.model_path)

    @staticmethod
    def _appearance_vector(frame: np.ndarray, bbox: BoundingBox) -> tuple[float, ...] | None:
        """Build a compact HSV histogram from the torso/clothing area."""
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(round(bbox.x1))))
        x2 = max(0, min(w, int(round(bbox.x2))))
        y1 = max(0, min(h - 1, int(round(bbox.y1))))
        y2 = max(0, min(h, int(round(bbox.y2))))
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None

        box_w = x2 - x1
        box_h = y2 - y1
        # Ignore most of the head and legs; clothing on the torso is more stable.
        tx1 = x1 + int(box_w * 0.12)
        tx2 = x2 - int(box_w * 0.12)
        ty1 = y1 + int(box_h * 0.18)
        ty2 = y1 + int(box_h * 0.62)
        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
        vector = hist.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return None
        vector /= norm
        return tuple(float(value) for value in vector)

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
            imgsz=self.config.image_size,
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
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            detections.append(
                Detection(
                    detector_track_id=raw_track_id,
                    bbox=bbox,
                    confidence=confidence,
                    appearance_vector=self._appearance_vector(frame, bbox),
                )
            )
        return detections
