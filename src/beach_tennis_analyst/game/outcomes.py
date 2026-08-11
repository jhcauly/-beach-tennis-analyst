from __future__ import annotations

from dataclasses import dataclass

from beach_tennis_analyst.game.events import BallContact, ShotEvent, ShotOutcome, StrokeType


@dataclass(frozen=True, slots=True)
class LandingAssessment:
    landing_x_m: float | None
    landing_y_m: float | None
    inside_court: bool | None
    hit_net: bool
    confidence: float


def classify_landing(
    *,
    landing_x_m: float | None,
    landing_y_m: float | None,
    hit_net: bool,
    court_width_m: float = 8.0,
    court_length_m: float = 16.0,
    margin_m: float = 0.08,
    confidence: float = 1.0,
) -> LandingAssessment:
    if hit_net:
        return LandingAssessment(landing_x_m, landing_y_m, None, True, confidence)
    if landing_x_m is None or landing_y_m is None:
        return LandingAssessment(None, None, None, False, confidence)
    inside = (
        -margin_m <= landing_x_m <= court_width_m + margin_m
        and -margin_m <= landing_y_m <= court_length_m + margin_m
    )
    return LandingAssessment(landing_x_m, landing_y_m, inside, False, confidence)


def classify_shot_outcome(
    *,
    assessment: LandingAssessment,
    opponent_contact_after_s: float | None,
    next_rally_contact_after_s: float | None,
    point_ended: bool,
) -> ShotOutcome:
    if assessment.hit_net:
        return ShotOutcome.NET
    if assessment.inside_court is False:
        return ShotOutcome.OUT
    if opponent_contact_after_s is not None:
        return ShotOutcome.DEFENDED
    if assessment.inside_court is True and point_ended and next_rally_contact_after_s is None:
        return ShotOutcome.WINNER
    if assessment.inside_court is True:
        return ShotOutcome.IN_PLAY
    return ShotOutcome.UNKNOWN


def build_shot_event(
    *,
    shot_id: str,
    rally_id: str,
    contact: BallContact,
    assessment: LandingAssessment,
    opponent_contact_after_s: float | None,
    next_rally_contact_after_s: float | None,
    point_ended: bool,
) -> ShotEvent:
    outcome = classify_shot_outcome(
        assessment=assessment,
        opponent_contact_after_s=opponent_contact_after_s,
        next_rally_contact_after_s=next_rally_contact_after_s,
        point_ended=point_ended,
    )
    return ShotEvent(
        shot_id=shot_id,
        rally_id=rally_id,
        contact=contact,
        landing_x_m=assessment.landing_x_m,
        landing_y_m=assessment.landing_y_m,
        outcome=outcome,
        opponent_touched_ball=opponent_contact_after_s is not None,
        point_ended=point_ended,
        confidence=min(contact.confidence, assessment.confidence),
    )


def classify_serve_or_smash(
    *,
    contact: BallContact,
    athlete_y_m: float,
    ball_height_px_relative_to_player: float | None,
    rally_shot_index: int,
    speed_px_s: float | None,
) -> StrokeType:
    if rally_shot_index == 0 and athlete_y_m <= 2.2:
        return StrokeType.SERVE
    if (
        ball_height_px_relative_to_player is not None
        and ball_height_px_relative_to_player < -0.18
        and (speed_px_s or 0.0) >= 500.0
    ):
        return StrokeType.SMASH
    return StrokeType.UNKNOWN
