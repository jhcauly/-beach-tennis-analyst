from __future__ import annotations

from collections import Counter, defaultdict

from beach_tennis_analyst.game.events import ShotEvent, ShotOutcome, StrokeType


def build_ball_report(shots: list[ShotEvent]) -> dict[str, object]:
    by_athlete: dict[str, list[ShotEvent]] = defaultdict(list)
    for shot in shots:
        by_athlete[shot.contact.athlete_id].append(shot)

    athletes: dict[str, object] = {}
    for athlete_id, athlete_shots in by_athlete.items():
        outcomes = Counter(shot.outcome.value for shot in athlete_shots)
        strokes = Counter(shot.contact.stroke_type.value for shot in athlete_shots)
        serve_errors = sum(
            shot.contact.stroke_type == StrokeType.SERVE
            and shot.outcome in {ShotOutcome.OUT, ShotOutcome.NET}
            for shot in athlete_shots
        )
        smash_errors = sum(
            shot.contact.stroke_type == StrokeType.SMASH
            and shot.outcome in {ShotOutcome.OUT, ShotOutcome.NET, ShotOutcome.UNFORCED_ERROR}
            for shot in athlete_shots
        )
        landing_grid = _landing_grid(athlete_shots)
        classified = sum(shot.outcome != ShotOutcome.UNKNOWN for shot in athlete_shots)
        projected = sum(
            shot.landing_x_m is not None and shot.landing_y_m is not None for shot in athlete_shots
        )
        athletes[athlete_id] = {
            "total_shots": len(athlete_shots),
            "outcomes": dict(outcomes),
            "strokes": dict(strokes),
            "serve_errors": serve_errors,
            "smash_errors": smash_errors,
            "landing_grid_3x3": landing_grid,
            "classification_coverage": 0.0 if not athlete_shots else classified / len(athlete_shots),
            "landing_projection_coverage": 0.0 if not athlete_shots else projected / len(athlete_shots),
            "pending_review": outcomes.get(ShotOutcome.UNKNOWN.value, 0),
        }

    return {
        "total_shots": len(shots),
        "athletes": athletes,
    }


def _landing_grid(shots: list[ShotEvent]) -> list[dict[str, object]]:
    counts = [[0 for _ in range(3)] for _ in range(3)]
    for shot in shots:
        if shot.landing_x_m is None or shot.landing_y_m is None:
            continue
        x = min(7.999999, max(0.0, shot.landing_x_m))
        y = min(15.999999, max(0.0, shot.landing_y_m))
        column = min(2, int(x / 8.0 * 3))
        row = min(2, int(y / 16.0 * 3))
        counts[row][column] += 1
    total = sum(sum(row) for row in counts)
    return [
        {
            "row": row,
            "column": column,
            "shots": counts[row][column],
            "ratio": 0.0 if total == 0 else counts[row][column] / total,
        }
        for row in range(3)
        for column in range(3)
    ]
