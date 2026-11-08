from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from beach_tennis_analyst.ball.tracking import BallTrackPoint
from beach_tennis_analyst.game.events import ShotEvent
from beach_tennis_analyst.game.rallies import RallyWindow


def export_ball_track(points: Iterable[BallTrackPoint], path: str | Path) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(json.dumps(asdict(point), ensure_ascii=False) + "\n")


def export_rallies(rallies: Iterable[RallyWindow], path: str | Path) -> None:
    payload = [asdict(rally) for rally in rallies]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_shots(shots: Iterable[ShotEvent], path: str | Path) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for shot in shots:
            payload = asdict(shot)
            payload["contact"]["stroke_type"] = shot.contact.stroke_type.value
            payload["outcome"] = shot.outcome.value
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
