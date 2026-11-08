from beach_tennis_analyst.calibration.court import BeachTennisCourt
from beach_tennis_analyst.calibration.manual import ManualCalibration
from beach_tennis_analyst.calibration.projection import CourtProjector


def test_court_corners_round_trip() -> None:
    calibration = ManualCalibration.from_points(
        [(100, 700), (900, 700), (700, 100), (300, 100)],
        frame_width_px=1000,
        frame_height_px=800,
    )
    projector = CourtProjector(calibration, BeachTennisCourt())

    expected = [(0.0, 0.0), (8.0, 0.0), (8.0, 16.0), (0.0, 16.0)]
    pixels = [(100, 700), (900, 700), (700, 100), (300, 100)]

    for pixel, metric in zip(pixels, expected, strict=True):
        projected = projector.pixel_to_metric(*pixel)
        assert abs(projected.x_m - metric[0]) < 1e-4
        assert abs(projected.y_m - metric[1]) < 1e-4

    assert projector.reprojection_error_px() < 1e-3
