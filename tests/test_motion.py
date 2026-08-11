from beach_tennis_analyst.analytics.motion import summarize_motion
from beach_tennis_analyst.domain.models import (
    Confidence,
    IdentityStatus,
    ObservationStatus,
    PlayerFrame,
)


def make_frame(index: int, x_m: float, speed: float, acceleration: float) -> PlayerFrame:
    return PlayerFrame(
        frame_index=index,
        timestamp_s=float(index),
        athlete_id="near_left",
        x_m=x_m,
        y_m=0.0,
        speed_mps=speed,
        acceleration_mps2=acceleration,
        observation_status=ObservationStatus.SMOOTHED,
        identity_status=IdentityStatus.CONFIRMED,
        confidence=Confidence(0.9, 0.9, 0.9, 0.9),
    )


def test_motion_summary_uses_valid_trajectory() -> None:
    summary = summarize_motion(
        [
            make_frame(0, 0.0, 0.0, 0.0),
            make_frame(1, 1.0, 1.0, 1.0),
            make_frame(2, 3.0, 2.0, -0.5),
        ]
    )
    assert summary.total_distance_m == 3.0
    assert summary.duration_s == 2.0
    assert summary.average_speed_mps == 1.5
    assert summary.maximum_speed_mps == 2.0
    assert summary.maximum_acceleration_mps2 == 1.0
    assert summary.maximum_deceleration_mps2 == -0.5
