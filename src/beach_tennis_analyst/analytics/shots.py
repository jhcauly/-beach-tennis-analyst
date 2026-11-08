from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from beach_tennis_analyst.game.events import ShotEvent, ShotOutcome, StrokeType


@dataclass(frozen=True, slots=True)
class StrokeSummary:
    stroke_type: StrokeType
    attempts: int
    winners: int
    defended: int
    out: int
    net: int
    forced_errors: int
    unforced_errors: int
    in_play: int
    success_rate: float


@dataclass(frozen=True, slots=True)
class LandingCell:
    row: int
    column: int
    samples: int
    ratio: float


@dataclass(frozen=True, slots=True)
class PlayerShotReport:
    athlete_id: str
    total_shots: int
    serve_errors: int
    smash_errors: int
    winners: int
    defended_shots: int
    shots_out: int
    shots_into_net: int
    landing_grid: tuple[LandingCell, ...]
    by_stroke: tuple[StrokeSummary, ...]


def _stroke_summary(stroke_type: StrokeType, events: list[ShotEvent]) -> StrokeSummary:
    attempts = len(events)
    count = lambda outcome: sum(event.outcome == outcome for event in events)
    winners = count(ShotOutcome.WINNER)
    defended = count(ShotOutcome.DEFENDED)
    out = count(ShotOutcome.OUT)
    net = count(ShotOutcome.NET)
    forced_errors = count(ShotOutcome.FORCED_ERROR)
    unforced_errors = count(ShotOutcome.UNFORCED_ERROR)
    in_play = count(ShotOutcome.IN_PLAY)
    successful = winners + defended + forced_errors + in_play
    return StrokeSummary(
        stroke_type=stroke_type,
        attempts=attempts,
        winners=winners,
        defended=defended,
        out=out,
        net=net,
        forced_errors=forced_errors,
        unforced_errors=unforced_errors,
        in_play=in_play,
        success_rate=0.0 if attempts == 0 else successful / attempts,
    )


def _landing_grid(
    events: list[ShotEvent],
    *,
    court_width_m: float = 8.0,
    opponent_half_length_m: float = 8.0,
    columns: int = 3,
    rows: int = 3,
) -> tuple[LandingCell, ...]:
    counts = [[0 for _ in range(columns)] for _ in range(rows)]
    valid = [
        event
        for event in events
        if event.landing_x_m is not None
        and event.landing_y_m is not None
        and 0.0 <= event.landing_x_m <= court_width_m
        and 0.0 <= event.landing_y_m <= opponent_half_length_m
    ]
    for event in valid:
        column = min(columns - 1, int(event.landing_x_m / court_width_m * columns))
        row = min(rows - 1, int(event.landing_y_m / opponent_half_length_m * rows))
        counts[row][column] += 1
    total = sum(sum(row) for row in counts)
    return tuple(
        LandingCell(
            row=row,
            column=column,
            samples=counts[row][column],
            ratio=0.0 if total == 0 else counts[row][column] / total,
        )
        for row in range(rows)
        for column in range(columns)
    )


def build_player_shot_report(
    events: Iterable[ShotEvent],
    *,
    athlete_id: str,
) -> PlayerShotReport:
    player_events = [event for event in events if event.contact.athlete_id == athlete_id]
    grouped = {
        stroke_type: [event for event in player_events if event.contact.stroke_type == stroke_type]
        for stroke_type in StrokeType
    }
    return PlayerShotReport(
        athlete_id=athlete_id,
        total_shots=len(player_events),
        serve_errors=sum(
            event.contact.stroke_type == StrokeType.SERVE
            and event.outcome in {ShotOutcome.OUT, ShotOutcome.NET, ShotOutcome.UNFORCED_ERROR}
            for event in player_events
        ),
        smash_errors=sum(
            event.contact.stroke_type == StrokeType.SMASH
            and event.outcome in {ShotOutcome.OUT, ShotOutcome.NET, ShotOutcome.UNFORCED_ERROR}
            for event in player_events
        ),
        winners=sum(event.outcome == ShotOutcome.WINNER for event in player_events),
        defended_shots=sum(event.outcome == ShotOutcome.DEFENDED for event in player_events),
        shots_out=sum(event.outcome == ShotOutcome.OUT for event in player_events),
        shots_into_net=sum(event.outcome == ShotOutcome.NET for event in player_events),
        landing_grid=_landing_grid(player_events),
        by_stroke=tuple(
            _stroke_summary(stroke_type, grouped[stroke_type])
            for stroke_type in StrokeType
            if grouped[stroke_type]
        ),
    )
