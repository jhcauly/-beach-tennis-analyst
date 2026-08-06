from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from beach_tennis_analyst.calibration.camera_standard import validate_rear_server_camera
from beach_tennis_analyst.calibration.court import BeachTennisCourt
from beach_tennis_analyst.calibration.manual import ManualCalibration, PixelPoint
from beach_tennis_analyst.calibration.projection import CourtProjector

POINT_LABELS = (
    "near_left",
    "near_right",
    "far_right",
    "far_left",
)


def first_frame(video_path: str | Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode first frame: {video_path}")
    return frame


def calibration_from_points(frame: np.ndarray, points: list[tuple[int, int]]) -> ManualCalibration:
    if len(points) != 4:
        raise ValueError("Exactly four points are required")
    height, width = frame.shape[:2]
    return ManualCalibration(
        near_left=PixelPoint(*points[0]),
        near_right=PixelPoint(*points[1]),
        far_right=PixelPoint(*points[2]),
        far_left=PixelPoint(*points[3]),
        source_width_px=width,
        source_height_px=height,
    )


def draw_calibration_preview(frame: np.ndarray, calibration: ManualCalibration) -> np.ndarray:
    preview = frame.copy()
    points = calibration.source_points.astype(np.int32)
    cv2.polylines(preview, [points.reshape((-1, 1, 2))], True, (0, 255, 255), 2)
    for index, (label, point) in enumerate(zip(POINT_LABELS, points, strict=True), start=1):
        x, y = int(point[0]), int(point[1])
        cv2.circle(preview, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(
            preview,
            f"{index}:{label}",
            (x + 8, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    report = validate_rear_server_camera(calibration)
    status = "VALID" if report.valid else "INVALID"
    cv2.putText(
        preview,
        f"rear-camera standard: {status}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if report.valid else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def top_down_preview(frame: np.ndarray, calibration: ManualCalibration, scale: int = 50) -> np.ndarray:
    court = BeachTennisCourt()
    width = int(court.width_m * scale)
    height = int(court.length_m * scale)
    destination = np.asarray(
        [[0, height - 1], [width - 1, height - 1], [width - 1, 0], [0, 0]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(calibration.source_points, destination)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    cv2.line(warped, (0, height // 2), (width - 1, height // 2), (255, 255, 255), 2)
    return warped


def run_visual_calibration(
    video_path: str | Path,
    output_json: str | Path,
    preview_path: str | Path | None = None,
    top_down_path: str | Path | None = None,
) -> ManualCalibration:
    frame = first_frame(video_path)
    points: list[tuple[int, int]] = []
    window = "Beach Tennis Calibration"

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = frame.copy()
        for index, point in enumerate(points):
            cv2.circle(canvas, point, 6, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                f"{index + 1}:{POINT_LABELS[index]}",
                (point[0] + 8, max(20, point[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        instruction = "Left click 1-4 | right click undo | ENTER save | ESC cancel"
        cv2.putText(canvas, instruction, (12, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled")
        if key in (10, 13) and len(points) == 4:
            calibration = calibration_from_points(frame, points)
            report = validate_rear_server_camera(calibration)
            if not report.valid:
                print(json.dumps({"valid": False, "warnings": report.warnings}, ensure_ascii=False, indent=2))
                continue
            calibration.save(output_json)
            if preview_path is not None:
                cv2.imwrite(str(preview_path), draw_calibration_preview(frame, calibration))
            if top_down_path is not None:
                cv2.imwrite(str(top_down_path), top_down_preview(frame, calibration))
            cv2.destroyAllWindows()
            return calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Click four court corners and save calibration.json")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--top-down", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calibration = run_visual_calibration(args.video, args.output, args.preview, args.top_down)
    report = validate_rear_server_camera(calibration)
    projector = CourtProjector(calibration, BeachTennisCourt())
    print(json.dumps({
        "saved": str(args.output),
        "valid": report.valid,
        "perspective_ratio": report.perspective_ratio,
        "center_offset_ratio": report.center_offset_ratio,
        "reprojection_error_px": projector.reprojection_error_px(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
