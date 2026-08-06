from __future__ import annotations

from dataclasses import replace

from beach_tennis_analyst.ball.tracking import BallTrackPoint
from beach_tennis_analyst.calibration.projection import CourtProjector
from beach_tennis_analyst.game.events import BallContact, ShotEvent, ShotOutcome, StrokeType
from beach_tennis_analyst.game.outcomes import build_shot_event, classify_landing
from beach_tennis_analyst.game.rallies import RallyWindow


def build_shot_events(
    contacts: list[BallContact],
    rallies: list[RallyWindow],
    track: list[BallTrackPoint],
    projector: CourtProjector,
) -> list[ShotEvent]:
    shots: list[ShotEvent] = []
    ordered_contacts = sorted(contacts, key=lambda item: item.timestamp_s)
    ordered_track = sorted(track, key=lambda item: item.timestamp_s)

    for rally in rallies:
        rally_contacts = [
            contact for contact in ordered_contacts if rally.start_s <= contact.timestamp_s <= rally.end_s
        ]
        for index, contact in enumerate(rally_contacts):
            next_contact = rally_contacts[index + 1] if index + 1 < len(rally_contacts) else None
            stroke = contact.stroke_type
            if index == 0 and stroke == StrokeType.UNKNOWN:
                stroke = StrokeType.SERVE
            normalized_contact = replace(contact, stroke_type=stroke)

            segment_end = next_contact.timestamp_s if next_contact else rally.end_s
            segment = [
                point
                for point in ordered_track
                if normalized_contact.timestamp_s <= point.timestamp_s <= segment_end
            ]
            landing_x = landing_y = None
            landing_confidence = 0.0
            if segment:
                candidate = segment[-1]
                metric = projector.pixel_to_metric(candidate.x_px, candidate.y_px)
                landing_x = metric.x_m
                landing_y = metric.y_m
                landing_confidence = candidate.confidence

            assessment = classify_landing(
                landing_x_m=landing_x,
                landing_y_m=landing_y,
                hit_net=False,
                confidence=landing_confidence,
            )
            opponent_contact_after = None
            if next_contact and _opposite_sides(normalized_contact.athlete_id, next_contact.athlete_id):
                opponent_contact_after = next_contact.timestamp_s
            point_ended = next_contact is None
            shot = build_shot_event(
                shot_id=f"{rally.rally_id}_shot_{index + 1:03d}",
                rally_id=rally.rally_id,
                contact=normalized_contact,
                assessment=assessment,
                opponent_contact_after_s=opponent_contact_after,
                next_rally_contact_after_s=next_contact.timestamp_s if next_contact else None,
                point_ended=point_ended,
            )
            shots.append(shot)
    return shots


def _opposite_sides(first: str, second: str) -> bool:
    return first.startswith("near_") != second.startswith("near_")
