from beach_tennis_analyst.ball.pipeline import BallAnalysisPipeline
from beach_tennis_analyst.ball.tracking import BallDetection


class _Point:
    def __init__(self, x_m: float, y_m: float) -> None:
        self.x_m = x_m
        self.y_m = y_m


class _Projector:
    def pixel_to_metric(self, x_px: float, y_px: float):
        return _Point(x_px / 10.0, y_px / 10.0)


def test_candidate_selection_prefers_predicted_position() -> None:
    pipeline = BallAnalysisPipeline(_Projector())
    selected = [
        BallDetection(1, 0.0, 20.0, 20.0, 0.8),
        BallDetection(2, 0.1, 30.0, 20.0, 0.8),
    ]
    candidates = [
        BallDetection(3, 0.2, 40.0, 20.0, 0.7),
        BallDetection(3, 0.2, 25.0, 25.0, 0.95),
    ]

    chosen = pipeline._select_candidate(candidates, selected)

    assert chosen is not None
    assert chosen.x_px == 40.0
    assert chosen.y_px == 20.0


def test_candidate_outside_extended_court_is_rejected() -> None:
    pipeline = BallAnalysisPipeline(_Projector())
    candidate = BallDetection(1, 0.0, 200.0, 200.0, 0.99)

    chosen = pipeline._select_candidate([candidate], [])

    assert chosen is None
