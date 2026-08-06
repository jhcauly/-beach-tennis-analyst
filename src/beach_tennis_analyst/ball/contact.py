from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from beach_tennis_analyst.ball.tracking import BallTrackPoint
from beach_tennis_analyst.domain.models import PlayerFrame
from beach_tennis_analyst.game.events import BallContact, StrokeType


@dataclass(frozen=True, slots=True)
class ContactDetectorConfig:
    maximum_player_distance_m: float = 1.8
    minimum_ball_speed_change_px_s: float = 120.0
    minimum_confidence: float = 0.35


def detect_contacts(
    ball_track: Iterable[BallTrackPoint],
    player_frames: dict[str, list[PlayerFrame]],
    *,
    pixel_to_metric: callable,
    config: ContactDetectorConfig | None = None,
) -> list[BallContact]:
    """Detect candidate contacts using proximity plus a ball direction/speed change.

    This is intentionally conservative and should emit UNKNOWN stroke type until
    pose or stroke classification is added.
    """
    cfg = config or ContactDetectorConfig()
    ball = list(ball_track)
    players_by_frame: dict[int, list[PlayerFrame]] = {}
    for frames in player_frames.values():
        for frame in frames:
            players_by_frame.setdefault(frame.frame_index, []).append(frame)

    contacts: list[BallContact] = []
    for index in range(1, len(ball) - 1):
        previous, current, following = ball[index - 1], ball[index], ball[index + 1]
        if current.confidence < cfg.minimum_confidence:
            continue
        if previous.speed_px_s is None or following.speed_px_s is None:
            continue
        speed_change = abs(following.speed_px_s - previous.speed_px_s)
        direction_change = 0.0
        if previous.vx_px_s is not None and previous.vy_px_s is not None and following.vx_px_s is not None and following.vy_px_s is not None:
            direction_change = hypot(following.vx_px_s - previous.vx_px_s, following.vy_px_s - previous.vy_px_s)
        if max(speed_change, direction_change) < cfg.minimum_ball_speed_change_px_s:
            continue

        metric = pixel_to_metric(current.x_px, current.y_px)
        nearest: tuple[float, PlayerFrame] | None = None
        for player in players_by_frame.get(current.frame_index, []):
            distance = hypot(metric.x_m - player.x_m, metric.y_m - player.y_m)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, player)
        if nearest is None or nearest[0] > cfg.maximum_player_distance_m:
            continue

        distance, player = nearest
        contacts.append(
            BallContact(
                frame_index=current.frame_index,
                timestamp_s=current.timestamp_s,
                athlete_id=player.athlete_id,
                stroke_type=StrokeType.UNKNOWN,
                contact_x_m=metric.x_m,
                contact_y_m=metric.y_m,
                confidence=max(0.0, min(1.0, current.confidence * (1.0 - distance / cfg.maximum_player_distance_m))),
            )
        )
    return contacts
