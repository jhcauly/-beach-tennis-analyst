"""Accept only frames whose static scene agrees with the calibrated camera."""
from __future__ import annotations

import cv2
import numpy as np

from beach_tennis_analyst.calibration.manual import ManualCalibration


class ReferenceViewGate:
    def __init__(self, reference: np.ndarray, calibration: ManualCalibration) -> None:
        height, width = reference.shape[:2]
        if (width, height) != (calibration.source_width_px, calibration.source_height_px):
            raise ValueError("Reference image dimensions do not match calibration")
        self.size = (min(width, 960), round(height * min(width, 960) / width))
        self.scale = self.size[0] / width
        self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
        self.mask = np.full((self.size[1], self.size[0]), 255, np.uint8)
        # Broadcast graphics are fixed on screen even when the camera changes.
        self.mask[:round(self.size[1] * .23), :round(self.size[0] * .32)] = 0
        self.points, self.descriptors = self.orb.detectAndCompute(self._gray(reference), self.mask)
        if self.descriptors is None or len(self.points) < 20:
            raise ValueError("Reference image lacks enough detail to recognize the camera view")
        self.anchors = (calibration.source_points * self.scale).reshape(-1, 1, 2)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.resize(frame, self.size), cv2.COLOR_BGR2GRAY)

    def matches(self, frame: np.ndarray) -> bool:
        points, descriptors = self.orb.detectAndCompute(self._gray(frame), self.mask)
        if descriptors is None or len(points) < 20:
            return False
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
        if len(good) < 18:
            return False
        source = np.float32([self.points[m.queryIdx].pt for m in good])
        target = np.float32([points[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return False
        stable = source[inliers.ravel().astype(bool)]
        if len(stable) < 18 or len(stable) / len(good) < .45:
            return False
        # A shared logo, one athlete, or a small patch cannot validate a whole camera.
        spread = np.ptp(stable, axis=0)
        if spread[0] < self.size[0] * .35 or spread[1] < self.size[1] * .25:
            return False
        projected = cv2.perspectiveTransform(self.anchors, matrix)
        error = np.linalg.norm(projected - self.anchors, axis=2)
        # Similar scenery under zoom/pan still invalidates the saved court projection.
        return bool(np.isfinite(error).all() and error.max() <= self.size[0] * .012)
