from beach_tennis_analyst.calibration.manual import ManualCalibration, PixelPoint
from beach_tennis_analyst.calibration.visual import calibration_from_points


def test_calibration_from_points_preserves_required_order() -> None:
    calibration = calibration_from_points(
        [(80.0, 330.0), (560.0, 330.0), (410.0, 120.0), (230.0, 120.0)],
        width_px=640,
        height_px=360,
    )

    assert calibration == ManualCalibration(
        near_left=PixelPoint(80.0, 330.0),
        near_right=PixelPoint(560.0, 330.0),
        far_right=PixelPoint(410.0, 120.0),
        far_left=PixelPoint(230.0, 120.0),
        source_width_px=640,
        source_height_px=360,
    )
