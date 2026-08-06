from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from beach_tennis_analyst.game.events import ShotEvent, ShotOutcome


@dataclass(frozen=True, slots=True)
class ReviewClip:
    clip_id: str
    athlete_id: str
    shot_id: str
    rally_id: str
    start_s: float
    end_s: float
    reason: str
    outcome: ShotOutcome
    confidence: float


def select_review_clips(
    events: Iterable[ShotEvent],
    *,
    athlete_id: str,
    seconds_before: float = 2.0,
    seconds_after: float = 2.5,
    minimum_confidence: float = 0.55,
) -> list[ReviewClip]:
    """Select non-winning actions that are useful for technical review."""
    selected: list[ReviewClip] = []
    relevant = {
        ShotOutcome.DEFENDED: "adversario_defendeu",
        ShotOutcome.OUT: "bola_fora",
        ShotOutcome.NET: "bola_na_rede",
        ShotOutcome.UNFORCED_ERROR: "erro_nao_forcado",
        ShotOutcome.FORCED_ERROR: "erro_forcado",
    }
    for event in events:
        if event.contact.athlete_id != athlete_id:
            continue
        if event.confidence < minimum_confidence or event.outcome not in relevant:
            continue
        timestamp = event.contact.timestamp_s
        selected.append(
            ReviewClip(
                clip_id=f"clip_{event.shot_id}",
                athlete_id=athlete_id,
                shot_id=event.shot_id,
                rally_id=event.rally_id,
                start_s=max(0.0, timestamp - seconds_before),
                end_s=timestamp + seconds_after,
                reason=relevant[event.outcome],
                outcome=event.outcome,
                confidence=event.confidence,
            )
        )
    return selected
