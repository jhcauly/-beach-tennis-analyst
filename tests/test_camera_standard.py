from beach_tennis_analyst.calibration.camera_standard import validate_rear_server_camera
from beach_tennis_analyst.calibration.depth_confidence import projection_confidence_for_y
from beach_tennis_analyst.calibration.manual import ManualCalibration


def test_rear_server_camera_standard_accepts_expected_perspective() -> None:
    calibration = ManualCalibration.from_points(
        [(80.0, 330.0), (560.0, 330.0), (430.0, 80.0), (210.0, 80.0)],
        frame_width_px=640,
        frame_height_px=360,
    )
    report = validate_rear_server_camera(calibration)
    assert report.valid is True
    assert report.near_width_px > report.far_width_px


def test_projection_confidence_decreases_with_depth() -> None:
    assert projection_confidence_for_y(0.0) == 1.0
    assert projection_confidence_for_y(16.0) == 0.65
    assert projection_confidence_for_y(8.0) == 0.825
