from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from beach_tennis_analyst.calibration.camera_standard import validate_rear_server_camera
from beach_tennis_analyst.calibration.court import BeachTennisCourt
from beach_tennis_analyst.calibration.manual import ManualCalibration
from beach_tennis_analyst.calibration.projection import CourtProjector

POINT_LABELS = (
    "near_left",
    "near_right",
    "far_right",
    "far_left",
)

SUPPORTED_REFERENCE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def calibration_from_points(
    points: list[tuple[float, float]],
    *,
    width_px: int,
    height_px: int,
) -> ManualCalibration:
    """Build calibration while preserving near-left to far-left point order."""
    return ManualCalibration.from_points(points, width_px, height_px)


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


def top_down_preview(
    frame: np.ndarray,
    calibration: ManualCalibration,
    scale: int = 50,
) -> np.ndarray:
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


def reference_dirs_for(video_path: str | Path) -> list[Path]:
    video = Path(video_path).resolve()
    candidates = [
        video.parent / "referencia",
        video.parent.parent / "referencia",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def find_reference_image(video_path: str | Path) -> Path:
    video = Path(video_path)
    searched: list[Path] = []

    for reference_dir in reference_dirs_for(video):
        reference_dir.mkdir(parents=True, exist_ok=True)
        for extension in SUPPORTED_REFERENCE_EXTENSIONS:
            candidate = reference_dir / f"{video.stem}{extension}"
            searched.append(candidate)
            if candidate.exists():
                return candidate

    expected = ", ".join(str(path) for path in searched)
    raise FileNotFoundError(
        f"Reference image not found for video '{video.name}'. "
        f"Place an image with the same file name in a 'referencia' folder. "
        f"Searched: {expected}"
    )


def load_reference_image(video_path: str | Path) -> tuple[np.ndarray, Path]:
    reference_path = find_reference_image(video_path)
    frame = cv2.imread(str(reference_path))
    if frame is None:
        raise RuntimeError(f"Could not open reference image: {reference_path}")
    return frame, reference_path


def run_visual_calibration(
    video_path: str | Path,
    output_json: str | Path,
    preview_path: str | Path | None = None,
    top_down_path: str | Path | None = None,
) -> ManualCalibration:
    frame, reference_path = load_reference_image(video_path)
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
        cv2.putText(
            canvas,
            instruction,
            (12, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled")
        if key in (10, 13) and len(points) == 4:
            height, width = frame.shape[:2]
            calibration = calibration_from_points(
                points,
                width_px=width,
                height_px=height,
            )
            report = validate_rear_server_camera(calibration)
            if not report.valid:
                print(
                    json.dumps(
                        {"valid": False, "warnings": report.warnings},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                continue
            calibration.save(output_json)
            if preview_path is not None:
                cv2.imwrite(str(preview_path), draw_calibration_preview(frame, calibration))
            if top_down_path is not None:
                cv2.imwrite(str(top_down_path), top_down_preview(frame, calibration))
            cv2.destroyAllWindows()
            print(json.dumps({"reference_image": str(reference_path)}, ensure_ascii=False))
            return calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a same-name image from a referencia folder and click four court corners"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--top-down", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calibration = run_visual_calibration(
        args.video,
        args.output,
        args.preview,
        args.top_down,
    )
    report = validate_rear_server_camera(calibration)
    projector = CourtProjector(calibration, BeachTennisCourt())
    ref_path = find_reference_image(args.video)
    print(
        json.dumps(
            {
                "saved": str(args.output),
                "reference_image": str(ref_path),
                "valid": report.valid,
                "perspective_ratio": report.perspective_ratio,
                "center_offset_ratio": report.center_offset_ratio,
                "reprojection_error_px": projector.reprojection_error_px(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
