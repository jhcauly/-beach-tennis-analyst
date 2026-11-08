from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StrokeType(StrEnum):
    SERVE = "serve"
    SMASH = "smash"
    VOLLEY = "volley"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    LOB = "lob"
    UNKNOWN = "unknown"


class ShotOutcome(StrEnum):
    WINNER = "winner"
    DEFENDED = "defended"
    OUT = "out"
    NET = "net"
    FORCED_ERROR = "forced_error"
    UNFORCED_ERROR = "unforced_error"
    IN_PLAY = "in_play"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BallContact:
    frame_index: int
    timestamp_s: float
    athlete_id: str
    stroke_type: StrokeType
    contact_x_m: float | None
    contact_y_m: float | None
    confidence: float


@dataclass(frozen=True, slots=True)
class ShotEvent:
    shot_id: str
    rally_id: str
    contact: BallContact
    landing_x_m: float | None
    landing_y_m: float | None
    outcome: ShotOutcome
    opponent_touched_ball: bool
    point_ended: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class RallyEvent:
    rally_id: str
    start_s: float
    end_s: float
    serving_athlete_id: str | None
    winning_team_side: str | None
    shots: tuple[ShotEvent, ...]
