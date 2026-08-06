from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from beach_tennis_analyst.game.events import BallContact


@dataclass(frozen=True, slots=True)
class RallyWindow:
    rally_id: str
    start_s: float
    end_s: float
    contact_count: int


@dataclass(frozen=True, slots=True)
class RallySegmenterConfig:
    inactivity_gap_s: float = 3.0
    pre_roll_s: float = 0.8
    post_roll_s: float = 1.2


def segment_rallies(
    contacts: Iterable[BallContact],
    *,
    config: RallySegmenterConfig | None = None,
) -> list[RallyWindow]:
    cfg = config or RallySegmenterConfig()
    items = sorted(contacts, key=lambda item: item.timestamp_s)
    if not items:
        return []

    groups: list[list[BallContact]] = [[items[0]]]
    for contact in items[1:]:
        if contact.timestamp_s - groups[-1][-1].timestamp_s > cfg.inactivity_gap_s:
            groups.append([contact])
        else:
            groups[-1].append(contact)

    return [
        RallyWindow(
            rally_id=f"rally_{index:04d}",
            start_s=max(0.0, group[0].timestamp_s - cfg.pre_roll_s),
            end_s=group[-1].timestamp_s + cfg.post_roll_s,
            contact_count=len(group),
        )
        for index, group in enumerate(groups, start=1)
    ]
