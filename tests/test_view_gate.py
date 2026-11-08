from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from beach_tennis_analyst.calibration.manual import ManualCalibration
from beach_tennis_analyst.calibration.view_gate import ReferenceViewGate
from beach_tennis_analyst.pipeline import session


@pytest.fixture
def scene():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    calibration = ManualCalibration.from_points(
        [(40, 340), (600, 340), (460, 120), (180, 120)], 640, 360
    )
    return image, calibration


def test_view_recognition_rejects_zoom_pan_and_different_camera(scene):
    image, calibration = scene
    gate = ReferenceViewGate(image, calibration)
    assert gate.matches(image)
    assert gate.matches(cv2.convertScaleAbs(image, alpha=.85, beta=10))
    athletes_moved = image.copy()
    cv2.rectangle(athletes_moved, (220, 160), (380, 340), (50, 60, 90), -1)
    assert gate.matches(athletes_moved)
    for scale in (1.05, 1.2, 1.5):
        zoom = cv2.warpAffine(image, cv2.getRotationMatrix2D((320, 180), 0, scale), (640, 360))
        assert not gate.matches(zoom)
    assert not gate.matches(cv2.warpAffine(image, np.float32([[1, 0, 35], [0, 1, 0]]), (640, 360)))
    assert not gate.matches(np.flip(image, axis=1))
    assert not gate.matches(np.zeros_like(image))
    scoreboard_only = np.zeros_like(image)
    scoreboard_only[:80, :200] = image[:80, :200]
    assert not gate.matches(scoreboard_only)


def test_reference_dimensions_must_match_calibration(scene):
    image, calibration = scene
    with pytest.raises(ValueError, match='dimensions'):
        ReferenceViewGate(image[:200], calibration)


def test_waits_past_initialization_limit_and_suspends_on_later_camera_cut(monkeypatch):
    detected = []
    updated = []
    class Detector:
        def __init__(self, config):
            pass
        def detect(self, frame):
            detected.append(frame)
            return []
    class Tracker:
        expected_track_count = 2
        tracks = {'near_left': 1, 'near_right': 2}
        def __init__(self, *args, **kwargs):
            pass
        def update(self, detections, frame_index):
            updated.append(frame_index)
            return []
    monkeypatch.setattr(session, 'PlayerDetector', Detector)
    monkeypatch.setattr(session, 'FourPlayerTracker', Tracker)
    frames = [False] * 400 + [True] * 5 + [False] * 20 + [True] * 4
    reader = SimpleNamespace(frames=lambda: ((i, i / 30, view) for i, view in enumerate(frames)))
    pipeline = session.BeachTennisAnalysisPipeline(max_initialization_frames=3)
    events = []
    pipeline._track_players(reader, None, total_frames=len(frames),
                            progress_callback=events.append,
                            view_gate=SimpleNamespace(matches=lambda view: view))
    assert updated == [402, 403, 404, 427, 428]
    assert all(detected)
    assert pipeline._view_segment_starts == {402, 427}
    assert pipeline._accepted_view_frames == set(updated)
    assert any(event['phase'] == 'waiting_for_reference_view' for event in events)


def test_no_matching_camera_reports_clear_error(monkeypatch):
    monkeypatch.setattr(session, 'PlayerDetector', lambda config: None)
    pipeline = session.BeachTennisAnalysisPipeline()
    reader = SimpleNamespace(frames=lambda: ((i, float(i), False) for i in range(12)))
    with pytest.raises(session.TrackingInitializationError, match='calibrated camera view'):
        pipeline._track_players(reader, None, total_frames=12,
                                view_gate=SimpleNamespace(matches=lambda view: view))


def test_calibration_reference_survives_save_and_load(tmp_path, scene):
    from dataclasses import replace
    _, calibration = scene
    calibration = replace(calibration, reference_image='match.reference.png')
    path = tmp_path / 'calibration.json'
    calibration.save(path)
    assert ManualCalibration.load(path) == calibration
