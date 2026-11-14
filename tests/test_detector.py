"""Integration tests for the real YOLO11n + ByteTrack detector (need yolo11n.pt and bus.jpg)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from config.profiles import PROFILES
from occhio.tracks import TrackManager, TrackStatus

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (ROOT / "yolo11n.pt").exists() or not (ROOT / "bus.jpg").exists(),
    reason="needs yolo11n.pt and bus.jpg",
)


@pytest.fixture(scope="module")
def bus() -> np.ndarray:
    return cv2.imread(str(ROOT / "bus.jpg"))


@pytest.fixture(scope="module")
def detector():
    from occhio.detector import Detector

    return Detector(PROFILES["street"]["classes"])


def test_detects_people_and_bus(detector, bus: np.ndarray) -> None:
    detector.reset()
    dets = detector.detect(bus)
    names = [d.cls_name for d in dets]
    assert names.count("person") >= 3
    assert "bus" in names
    for d in dets:
        assert isinstance(d.track_id, int)
        assert 0.0 < d.conf <= 1.0
        x1, y1, x2, y2 = d.xyxy
        assert 0 <= x1 < x2 <= bus.shape[1] and 0 <= y1 < y2 <= bus.shape[0]


def test_filters_by_profile_classes(detector, bus: np.ndarray) -> None:
    detector.set_classes(PROFILES["industrial"]["classes"])
    try:
        detector.reset()
        names = {d.cls_name for d in detector.detect(bus)}
        assert "bus" not in names and "person" in names
        assert names <= set(PROFILES["industrial"]["classes"])
    finally:
        detector.set_classes(PROFILES["street"]["classes"])


def test_unknown_class_rejected(detector) -> None:
    with pytest.raises(ValueError):
        detector.set_classes(["unicorn"])


def test_same_id_after_walking_behind_a_chair(detector, bus: np.ndarray) -> None:
    """Phase 1 'done when': a person passes behind an occluder and keeps the same ID."""
    detector.reset()
    person = max((d for d in detector.detect(bus) if d.cls_name == "person"), key=lambda d: d.conf)
    x1, y1, x2, y2 = (int(c) for c in person.xyxy)
    crop = bus[y1:y2, x1:x2]
    ph, pw = crop.shape[:2]

    W, H = 1280, 720
    background = np.full((H, W, 3), (90, 110, 120), dtype=np.uint8)
    occluder = (560, 760)  # the "chair": a wall the person walks behind
    detector.reset()
    mgr = TrackManager()
    ids_before: set[int] = set()
    ids_after: set[int] = set()
    saw_hidden = False
    step = 10  # px per frame
    t = 0.0
    for i in range(80):
        frame = background.copy()
        px = 100 + i * step
        top = H - ph - 40
        if px + pw <= W:
            frame[top:top + ph, px:px + pw] = crop
        cv2.rectangle(frame, (occluder[0], 0), (occluder[1], H), (40, 40, 40), -1)
        tracks = mgr.update(detector.detect(frame), t)
        t += 1 / 30
        persons = [tr for tr in tracks if tr.cls_name == "person"]
        if px + pw < occluder[0]:
            ids_before |= {tr.track_id for tr in persons if tr.visible}
        elif px > occluder[1] and px + pw <= W:
            ids_after |= {tr.track_id for tr in persons if tr.visible}
        saw_hidden |= any(tr.status is TrackStatus.HIDDEN for tr in persons)

    assert len(ids_before) == 1, f"ID switched before occlusion: {ids_before}"
    assert saw_hidden, "track never went hidden behind the occluder"
    assert ids_after == ids_before, f"ID changed after occlusion: {ids_before} -> {ids_after}"



@pytest.mark.skipif(not (ROOT / "yolo11s.pt").exists(), reason="needs yolo11s.pt")
def test_more_accurate_model_finds_the_same_people() -> None:
    from occhio.detector import Detector
    bus = cv2.imread(str(ROOT / "bus.jpg"))
    small = Detector(PROFILES["street"]["classes"], {"model": str(ROOT / "yolo11s.pt")})
    names = [d.cls_name for d in small.detect(bus)]
    assert names.count("person") >= 4 and "bus" in names


def test_model_choice() -> None:
    import main
    from occhio.app import App  # noqa: F401
    assert main.parse_args(["--source", "phone"]).model == "auto"
    assert main.parse_args(["--model", "s"]).model == "s"
