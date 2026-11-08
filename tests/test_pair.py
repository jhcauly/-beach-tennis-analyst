from beach_tennis_analyst.analytics.pair import build_pair_frames, summarize_pair
from beach_tennis_analyst.domain.models import (
    Confidence,
    IdentityStatus,
    ObservationStatus,
    PlayerFrame,
)


def frame(index: int, athlete_id: str, x_m: float, y_m: float) -> PlayerFrame:
    return PlayerFrame(
        frame_index=index,
        timestamp_s=index / 10.0,
        athlete_id=athlete_id,
        x_m=x_m,
        y_m=y_m,
        speed_mps=0.0,
        acceleration_mps2=0.0,
        observation_status=ObservationStatus.OBSERVED,
        identity_status=IdentityStatus.CONFIRMED,
        confidence=Confidence(1.0, 1.0, 1.0, 1.0),
    )


def test_pair_centroid_opening_and_misalignment() -> None:
    left = [frame(0, "near_left", 1.0, 3.0), frame(1, "near_left", 1.5, 3.0)]
    right = [frame(0, "near_right", 7.0, 3.5), frame(1, "near_right", 6.5, 6.0)]

    pair_frames = build_pair_frames(left, right, team_side="near")
    assert pair_frames[0].centroid_x_m == 4.0
    assert pair_frames[0].lateral_opening_m == 6.0
    assert pair_frames[0].is_too_open is True
    assert pair_frames[1].depth_misalignment_m == 3.0
    assert pair_frames[1].is_depth_misaligned is True

    summary = summarize_pair(pair_frames)
    assert summary.valid_samples == 2
    assert summary.too_open_ratio == 0.5
    assert summary.depth_misaligned_ratio == 0.5
