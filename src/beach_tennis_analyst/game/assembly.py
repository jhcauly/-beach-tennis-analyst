from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from beach_tennis_analyst.ball.tracking import BallTrackPoint
from beach_tennis_analyst.calibration.projection import CourtProjector
from beach_tennis_analyst.game.events import BallContact, ShotEvent, StrokeType
from beach_tennis_analyst.game.outcomes import build_shot_event, classify_landing
from beach_tennis_analyst.game.rallies import RallyWindow


def infer_landing_after_contact(
    contact: BallContact,
    track: Iterable[BallTrackPoint],
    projector: CourtProjector,
    *,
    search_window_s: float = 2.5,
):
    points = [
        item
        for item in track
        if contact.timestamp_s < item.timestamp_s <= contact.timestamp_s + search_window_s
    ]
    if len(points) < 3:
        return classify_landing(
            landing_x_m=None,
            landing_y_m=None,
            hit_net=False,
            confidence=0.0,
        )

    metric = [projector.pixel_to_metric(item.x_px, item.y_px) for item in points]
    candidate_index = None
    for index in range(1, len(metric) - 1):
        before = metric[index - 1]
        current = metric[index]
        after = metric[index + 1]
        before_dy = current.y_m - before.y_m
        after_dy = after.y_m - current.y_m
        if before_dy * after_dy < 0:
            candidate_index = index
            break

    if candidate_index is None:
        candidate_index = len(metric) - 1

    landing = metric[candidate_index]
    confidence = points[candidate_index].confidence * (0.7 if candidate_index == len(metric) - 1 else 1.0)
    hit_net = abs(landing.y_m - 8.0) <= 0.35 and candidate_index < 2
    return classify_landing(
        landing_x_m=landing.x_m,
        landing_y_m=landing.y_m,
        hit_net=hit_net,
        confidence=confidence,
    )


def assemble_shot_events(
    contacts: Iterable[BallContact],
    rallies: Iterable[RallyWindow],
    track: Iterable[BallTrackPoint],
    projector: CourtProjector,
) -> list[ShotEvent]:
    contact_items = sorted(contacts, key=lambda item: item.timestamp_s)
    windows = list(rallies)
    events: list[ShotEvent] = []

    for rally in windows:
        rally_contacts = [
            item for item in contact_items if rally.start_s <= item.timestamp_s <= rally.end_s
        ]
        for index, contact in enumerate(rally_contacts):
            next_contact = rally_contacts[index + 1] if index + 1 < len(rally_contacts) else None
            opponent_contact_after_s = None
            if next_contact is not None and next_contact.athlete_id != contact.athlete_id:
                opponent_contact_after_s = next_contact.timestamp_s
            point_ended = next_contact is None
            assessment = infer_landing_after_contact(contact, track, projector)
            event = build_shot_event(
                shot_id=f"{rally.rally_id}_shot_{index + 1:03d}",
                rally_id=rally.rally_id,
                contact=contact,
                assessment=assessment,
                opponent_contact_after_s=opponent_contact_after_s,
                next_rally_contact_after_s=next_contact.timestamp_s if next_contact else None,
                point_ended=point_ended,
            )
            if index == 0 and contact.stroke_type == StrokeType.UNKNOWN:
                event = replace(event, contact=replace(contact, stroke_type=StrokeType.SERVE))
            events.append(event)
    return events
