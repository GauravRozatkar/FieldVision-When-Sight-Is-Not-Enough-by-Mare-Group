"""Walkway obstacles: bags, phones, chairs... detection rules, distance and wording."""
from __future__ import annotations

import pytest

from config.profiles import CATEGORY, OBJECTS, OBJECTS_PLURAL, PROFILES, detect_classes, render
from occhio.detector import Detection
from occhio.priority import PriorityPlanner
from occhio.reliability import ReliabilityState
from occhio.spatial import Calibration, SpatialReasoner
from occhio.tracks import TrackManager, TrackStatus

STREET, INDUSTRIAL = PROFILES["street"], PROFILES["industrial"]
OK = ReliabilityState(1.0, 1.0, 150, 500, 60, 0.9, False, True, "ok")
FRAME = (720, 1280)


def test_every_class_has_names_and_the_model_knows_it() -> None:
    from ultralytics.utils import YAML  # noqa: F401  (only to make sure ultralytics imports)
    from occhio.detector import Detector
    known = set(Detector(["person"])._names.values())
    for profile in PROFILES.values():
        for cls in detect_classes(profile):
            assert cls in known, cls
            for lang in ("en", "it"):
                assert cls in OBJECTS[lang], (lang, cls)
                if cls in profile["classes"]:
                    assert cls in OBJECTS_PLURAL[lang], (lang, cls)


def test_walkway_obstacles_are_in_street_mode() -> None:
    for cls in ("backpack", "handbag", "suitcase", "cell phone", "bottle", "chair", "bench"):
        assert cls in STREET["classes"]
    assert "stop sign" in STREET["signal_classes"] and "stop sign" not in STREET["classes"]
    assert CATEGORY.get("handbag", "obstacle") == "obstacle" and CATEGORY["car"] == "vehicle"


def test_small_objects_confirm_with_lower_confidence_but_more_frames() -> None:
    mgr = TrackManager()
    for i in range(7):
        mgr.update([Detection(1, "cell phone", 0.45, (600, 600, 640, 620))], i / 30)
    assert mgr.tracks[1].status is TrackStatus.UNCERTAIN  # 7 frames: not yet
    mgr.update([Detection(1, "cell phone", 0.45, (600, 600, 640, 620))], 8 / 30)
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED  # 8 frames at 0.45 is enough for a phone
    person = TrackManager()
    for i in range(10):
        person.update([Detection(1, "person", 0.45, (600, 200, 700, 600))], i / 30)
    assert person.tracks[1].status is TrackStatus.UNCERTAIN  # people still need 0.6


def test_phone_on_the_floor_distance_from_ground_contact() -> None:
    r = SpatialReasoner(STREET, calibration=Calibration(focal_px=900, frame_width=1280, fixed=True))
    mgr = TrackManager()
    # floor point 180 px below the horizon (row 360), camera 1.2 m high: 1.2 * 900 / 180 = 6 m
    tracks = mgr.update([Detection(1, "cell phone", 0.9, (620, 520, 660, 540))], 0.0)
    info = r.update(tracks, None, FRAME, 0.0)[0]
    assert info.distance_source == "ground" and info.distance_m == pytest.approx(6.0)
    # same phone box much lower in the image (closer): 1.2 * 900 / 340 = 3.2 m
    tracks = TrackManager().update([Detection(2, "cell phone", 0.9, (620, 680, 660, 700))], 1.0)
    assert r.update(tracks, None, FRAME, 1.0)[0].distance_m == pytest.approx(1.2 * 900 / 340, rel=0.01)


def test_bag_warning_wording() -> None:
    from occhio.spatial import SpatialInfo
    bag = SpatialInfo(3, "handbag", True, 0.5, "ahead", 12, 2.0, "ground", 0.0, 0.0, 0.0, False, None)
    assert PriorityPlanner(STREET, "en").plan([bag], OK, 0.0).text == "Bag ahead, 2 metres"
    assert render(STREET, "it", "ahead", "cell phone", 1.5) == "Telefono davanti, 1,5 metri"


def test_industrial_approach_names_the_object() -> None:
    assert render(INDUSTRIAL, "en", "approach", "person", side="left") == "Person coming from the left. Stop."
    assert render(INDUSTRIAL, "it", "approach", "truck", side="right") == "Camion in arrivo da destra. Stop."
