from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .manual import ManualCalibration


@dataclass(frozen=True, slots=True)
class CameraStandardReport:
    valid: bool
    near_width_px: float
    far_width_px: float
    center_offset_ratio: float
    perspective_ratio: float
    warnings: tuple[str, ...]


def validate_rear_server_camera(calibration: ManualCalibration) -> CameraStandardReport:
    """Validate the product standard: camera behind the server, facing court lengthwise."""
    calibration.validate()
    near_left = calibration.near_left
    near_right = calibration.near_right
    far_left = calibration.far_left
    far_right = calibration.far_right

    near_width = hypot(near_right.x - near_left.x, near_right.y - near_left.y)
    far_width = hypot(far_right.x - far_left.x, far_right.y - far_left.y)
    near_center_x = (near_left.x + near_right.x) / 2.0
    frame_center_x = calibration.source_width_px / 2.0
    center_offset_ratio = abs(near_center_x - frame_center_x) / calibration.source_width_px
    perspective_ratio = 0.0 if near_width == 0.0 else far_width / near_width

    warnings: list[str] = []
    if near_width <= far_width:
        warnings.append("near baseline should appear wider than far baseline")
    if not 0.20 <= perspective_ratio <= 0.95:
        warnings.append("court perspective ratio is outside the expected rear-camera range")
    if center_offset_ratio > 0.20:
        warnings.append("camera appears too far from the court centerline")
    # Image coordinates grow downward. For a rear camera, the near corners must
    # therefore have a larger y coordinate (appear lower) than the far corners.
    if near_left.y <= far_left.y or near_right.y <= far_right.y:
        warnings.append("near corners must appear lower in the image than far corners")

    return CameraStandardReport(
        valid=not warnings,
        near_width_px=near_width,
        far_width_px=far_width,
        center_offset_ratio=center_offset_ratio,
        perspective_ratio=perspective_ratio,
        warnings=tuple(warnings),
    )
