from __future__ import annotations

from dataclasses import replace

from .tracking import BallTrackPoint


def interpolate_short_gaps(
    points: list[BallTrackPoint],
    *,
    max_gap_frames: int = 4,
) -> list[BallTrackPoint]:
    if len(points) < 2:
        return list(points)

    ordered = sorted(points, key=lambda item: item.frame_index)
    output: list[BallTrackPoint] = []
    for left, right in zip(ordered, ordered[1:]):
        output.append(left)
        gap = right.frame_index - left.frame_index - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        frame_span = right.frame_index - left.frame_index
        time_span = right.timestamp_s - left.timestamp_s
        if frame_span <= 0 or time_span <= 0:
            continue
        for offset in range(1, gap + 1):
            ratio = offset / frame_span
            timestamp = left.timestamp_s + time_span * ratio
            x = left.x_px + (right.x_px - left.x_px) * ratio
            y = left.y_px + (right.y_px - left.y_px) * ratio
            vx = (right.x_px - left.x_px) / time_span
            vy = (right.y_px - left.y_px) / time_span
            output.append(
                BallTrackPoint(
                    frame_index=left.frame_index + offset,
                    timestamp_s=timestamp,
                    x_px=x,
                    y_px=y,
                    vx_px_s=vx,
                    vy_px_s=vy,
                    speed_px_s=(vx * vx + vy * vy) ** 0.5,
                    confidence=min(left.confidence, right.confidence) * 0.65,
                    interpolated=True,
                )
            )
    output.append(ordered[-1])
    return sorted(output, key=lambda item: item.frame_index)


def smooth_track(points: list[BallTrackPoint], *, window: int = 3) -> list[BallTrackPoint]:
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if len(points) < window:
        return list(points)
    radius = window // 2
    smoothed: list[BallTrackPoint] = []
    for index, point in enumerate(points):
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        sample = points[start:end]
        x = sum(item.x_px for item in sample) / len(sample)
        y = sum(item.y_px for item in sample) / len(sample)
        smoothed.append(replace(point, x_px=x, y_px=y))
    return smoothed
