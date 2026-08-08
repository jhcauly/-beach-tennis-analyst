from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from beach_tennis_analyst.ball.tracking import BallTrackPoint
from beach_tennis_analyst.calibration.projection import CourtProjector
from beach_tennis_analyst.domain.models import PlayerFrame
from beach_tennis_analyst.game.events import ShotEvent
from beach_tennis_analyst.ingestion.video_reader import VideoReader


def render_movement_2d(
    *,
    output_path: str | Path,
    fps: float,
    frame_count: int,
    trajectories: dict[str, list[PlayerFrame]],
    ball_track: list[BallTrackPoint],
    projector: CourtProjector,
    width_px: int = 480,
    height_px: int = 800,
) -> None:
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width_px, height_px)
    )
    players = _players_by_frame(trajectories)
    ball = {point.frame_index: point for point in ball_track}
    try:
        for frame_index in range(frame_count):
            canvas = np.full((height_px, width_px, 3), 238, dtype=np.uint8)
            _draw_court(canvas)
            for player in players.get(frame_index, []):
                x, y = _metric_to_canvas(player.x_m, player.y_m, width_px, height_px)
                cv2.circle(canvas, (x, y), 11, _athlete_color(player.athlete_id), -1)
                cv2.putText(canvas, player.athlete_id, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
            point = ball.get(frame_index)
            if point is not None:
                metric = projector.pixel_to_metric(point.x_px, point.y_px)
                x, y = _metric_to_canvas(metric.x_m, metric.y_m, width_px, height_px)
                cv2.circle(canvas, (x, y), 6, (0, 220, 255), -1)
            writer.write(canvas)
    finally:
        writer.release()


def render_annotated_match(
    *,
    video_path: str | Path,
    output_path: str | Path,
    trajectories: dict[str, list[PlayerFrame]],
    ball_track: list[BallTrackPoint],
    shots: list[ShotEvent],
    projector: CourtProjector,
) -> None:
    reader = VideoReader(video_path)
    metadata = reader.inspect()
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width_px, metadata.height_px),
    )
    players = _players_by_frame(trajectories)
    ball = {point.frame_index: point for point in ball_track}
    shot_by_frame = {shot.contact.frame_index: shot for shot in shots}
    try:
        for frame_index, _, frame in reader.frames():
            _draw_projected_court(frame, projector)
            for player in players.get(frame_index, []):
                if player.athlete_id not in {"near_left", "near_right"}:
                    continue
                _draw_player_identity_box(frame, player, projector)

            point = ball.get(frame_index)
            if point is not None:
                cv2.circle(frame, (round(point.x_px), round(point.y_px)), 7, (0, 255, 255), 2)
            shot = shot_by_frame.get(frame_index)
            if shot is not None:
                label = f"{shot.contact.stroke_type.value}: {shot.outcome.value}"
                cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def _draw_player_identity_box(
    frame: np.ndarray,
    player: PlayerFrame,
    projector: CourtProjector,
) -> None:
    foot_x, foot_y = projector.metric_to_pixel(player.x_m, player.y_m)
    frame_h, frame_w = frame.shape[:2]

    depth_ratio = min(max(player.y_m / 16.0, 0.0), 1.0)
    box_height = round(210 - 115 * depth_ratio)
    box_height = max(85, min(230, box_height))
    box_width = round(box_height * 0.42)

    x1 = max(0, round(foot_x - box_width / 2))
    x2 = min(frame_w - 1, round(foot_x + box_width / 2))
    y2 = min(frame_h - 1, round(foot_y))
    y1 = max(0, y2 - box_height)

    red = (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), red, 3)

    label = f"{player.athlete_id} | {player.identity_status.value}"
    text_y = max(22, y1 - 8)
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        red,
        2,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (round(foot_x), round(foot_y)), 5, red, -1)


def _draw_projected_court(frame: np.ndarray, projector: CourtProjector) -> None:
    corners_metric = [(0.0, 0.0), (8.0, 0.0), (8.0, 16.0), (0.0, 16.0)]
    corners_px = np.asarray(
        [[round(x), round(y)] for x, y in (projector.metric_to_pixel(x_m, y_m) for x_m, y_m in corners_metric)],
        dtype=np.int32,
    )
    court_color = (0, 255, 255)
    cv2.polylines(frame, [corners_px], isClosed=True, color=court_color, thickness=3, lineType=cv2.LINE_AA)

    net_left = projector.metric_to_pixel(0.0, 8.0)
    net_right = projector.metric_to_pixel(8.0, 8.0)
    cv2.line(
        frame,
        (round(net_left[0]), round(net_left[1])),
        (round(net_right[0]), round(net_right[1])),
        court_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "COURT",
        (round(corners_px[0][0]), max(22, round(corners_px[0][1]) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        court_color,
        2,
        cv2.LINE_AA,
    )


def _players_by_frame(trajectories: dict[str, list[PlayerFrame]]) -> dict[int, list[PlayerFrame]]:
    result: dict[int, list[PlayerFrame]] = defaultdict(list)
    for frames in trajectories.values():
        for frame in frames:
            result[frame.frame_index].append(frame)
    return result


def _draw_court(canvas: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    margin_x, margin_y = 50, 40
    cv2.rectangle(canvas, (margin_x, margin_y), (width - margin_x, height - margin_y), (255, 255, 255), 3)
    net_y = height // 2
    cv2.line(canvas, (margin_x, net_y), (width - margin_x, net_y), (255, 255, 255), 3)
    cv2.line(canvas, (width // 2, margin_y), (width // 2, height - margin_y), (255, 255, 255), 1)


def _metric_to_canvas(x_m: float, y_m: float, width: int, height: int) -> tuple[int, int]:
    margin_x, margin_y = 50, 40
    x = margin_x + x_m / 8.0 * (width - 2 * margin_x)
    y = height - margin_y - y_m / 16.0 * (height - 2 * margin_y)
    return round(x), round(y)


def _athlete_color(athlete_id: str) -> tuple[int, int, int]:
    return (255, 120, 40) if athlete_id.startswith("near_") else (40, 160, 255)
