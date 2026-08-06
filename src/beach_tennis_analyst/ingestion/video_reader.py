from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoOpenError(RuntimeError):
    """Raised when OpenCV cannot open or inspect a video."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width_px: int
    height_px: int
    duration_s: float
    codec: str

    def timestamp_for_frame(self, frame_index: int) -> float:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(f"frame_index out of range: {frame_index}")
        return frame_index / self.fps


class VideoReader:
    """Safe sequential reader that keeps the source FPS instead of assuming 24/30 FPS."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def inspect(self) -> VideoMetadata:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise VideoOpenError(f"Could not open video: {self.path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        finally:
            capture.release()

        if fps <= 0:
            raise VideoOpenError(f"Invalid FPS reported for {self.path}: {fps}")
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise VideoOpenError(
                f"Invalid metadata for {self.path}: "
                f"frames={frame_count}, size={width}x{height}"
            )

        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")
        return VideoMetadata(
            path=self.path,
            fps=fps,
            frame_count=frame_count,
            width_px=width,
            height_px=height,
            duration_s=frame_count / fps,
            codec=codec or "unknown",
        )

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        metadata = self.inspect()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise VideoOpenError(f"Could not reopen video: {self.path}")

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame_index, frame_index / metadata.fps, frame
                frame_index += 1
        finally:
            capture.release()

        if frame_index == 0:
            raise VideoOpenError(f"No decodable frames found in {self.path}")
