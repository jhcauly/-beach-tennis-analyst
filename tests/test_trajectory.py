from beach_tennis_analyst.domain.models import (
    Confidence,
    IdentityStatus,
    ObservationStatus,
    PlayerFrame,
)
from beach_tennis_analyst.tracking.trajectory import TrajectoryConfig, TrajectoryProcessor


def frame(index: int, x_m: float, status: ObservationStatus = ObservationStatus.OBSERVED) -> PlayerFrame:
    return PlayerFrame(
        frame_index=index,
        timestamp_s=index / 10.0,
        athlete_id="near_left",
        x_m=x_m,
        y_m=2.0,
        speed_mps=None,
        acceleration_mps2=None,
        observation_status=status,
        identity_status=IdentityStatus.CONFIRMED,
        confidence=Confidence(0.9, 0.9, 0.9, 0.9),
    )


def test_short_gap_is_interpolated() -> None:
    processor = TrajectoryProcessor(
        TrajectoryConfig(smoothing_window=1, max_interpolation_gap_frames=2)
    )
    result = processor.process(
        [
            frame(0, 0.0),
            frame(1, 0.0, ObservationStatus.SUSPECT),
            frame(2, 2.0),
        ]
    )
    assert result[1].observation_status == ObservationStatus.INTERPOLATED
    assert result[1].x_m == 1.0


def test_impossible_jump_is_marked_suspect() -> None:
    processor = TrajectoryProcessor(
        TrajectoryConfig(smoothing_window=1, max_speed_mps=5.0, max_acceleration_mps2=100.0)
    )
    result = processor.process([frame(0, 0.0), frame(1, 5.0)])
    assert result[1].observation_status == ObservationStatus.SUSPECT
    assert result[1].speed_mps is None


def test_motion_is_derived_after_processing() -> None:
    processor = TrajectoryProcessor(TrajectoryConfig(smoothing_window=1))
    result = processor.process([frame(0, 0.0), frame(1, 0.1), frame(2, 0.2)])
    assert result[1].speed_mps == 1.0
    assert result[2].speed_mps == 1.0
