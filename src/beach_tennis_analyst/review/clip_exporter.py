from __future__ import annotations

from pathlib import Path

import cv2

from beach_tennis_analyst.game.events import ShotEvent, ShotOutcome


REVIEW_OUTCOMES = {
    ShotOutcome.DEFENDED,
    ShotOutcome.OUT,
    ShotOutcome.NET,
    ShotOutcome.FORCED_ERROR,
    ShotOutcome.UNFORCED_ERROR,
    ShotOutcome.UNKNOWN,
}


def export_review_clips(
    *,
    video_path: str | Path,
    shots: list[ShotEvent],
    output_dir: str | Path,
    pre_s: float = 2.0,
    post_s: float = 2.5,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    paths: list[Path] = []
    try:
        for shot in shots:
            if shot.outcome not in REVIEW_OUTCOMES:
                continue
            start_frame = max(0, int((shot.contact.timestamp_s - pre_s) * fps))
            end_frame = min(frame_count - 1, int((shot.contact.timestamp_s + post_s) * fps))
            path = output / f"{shot.shot_id}_{shot.outcome.value}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(start_frame, end_frame + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
            writer.release()
            paths.append(path)
    finally:
        capture.release()
    return paths
