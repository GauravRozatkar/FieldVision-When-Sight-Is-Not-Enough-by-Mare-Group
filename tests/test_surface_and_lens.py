"""Real iPhone frames: a plain wall right in front is 'very close', not 'poor visibility';
the ultra-wide lens changes the field of view used for distances."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from config.profiles import PROFILES
from config.settings import RELIABILITY_PHONE
from occhio.phone_link import PhoneCamera
from occhio.reliability import ReliabilityMonitor
from occhio.spatial import SpatialReasoner

FIX = Path(__file__).parent / "fixtures"


def settle(monitor: ReliabilityMonitor, image: np.ndarray, frames: int = 30):
    state = None
    for _ in range(frames):
        state = monitor.update(image)
    return state


def test_plain_wall_from_the_iphone_is_a_close_surface() -> None:
    wall = cv2.imread(str(FIX / "iphone_wall.png"))
    state = settle(ReliabilityMonitor(RELIABILITY_PHONE), wall)
    assert state.reason == "surface" and state.reliable  # not "poor visibility"


def test_table_from_the_iphone_is_a_normal_view() -> None:
    table = cv2.imread(str(FIX / "iphone_table.png"))
    state = settle(ReliabilityMonitor(RELIABILITY_PHONE), table)
    assert state.reason == "ok" and state.reliable


def test_covered_or_dark_lens_is_still_unreliable() -> None:
    wall = cv2.imread(str(FIX / "iphone_wall.png"))
    for img, reason in ((np.full_like(wall, 4), "blocked"), ((wall * 0.12).astype(np.uint8), None)):
        state = settle(ReliabilityMonitor(RELIABILITY_PHONE), img)
        assert not state.reliable and state.reason != "surface"
        if reason:
            assert state.reason == reason


def test_phone_reports_its_lens() -> None:
    phone = PhoneCamera(use_https=False)
    phone._read_query("/frame?fov=108&voice=1")
    assert phone.fov_deg == 108
    phone._read_query("/frame?fov=500")
    assert phone.fov_deg == 130  # clamped
    page = PhoneCamera(use_https=False, strings={"wide_on": "Wide angle: on"}).page().decode()
    assert "enumerateDevices" in page and "/ultra/i" in page and '"fov=" + fov()' in page and "Wide angle: on" in page


def test_ultra_wide_lens_changes_distances() -> None:
    r = SpatialReasoner(PROFILES["street"], hfov_deg=69)
    box = (600, 300, 680, 560)  # a person 260 px tall in a 1280x720 frame
    normal = r.size_distance("person", box, (720, 1280))
    r.hfov_deg = 108  # same pixels through the ultra-wide lens: the person is much closer
    wide = r.size_distance("person", box, (720, 1280))
    assert wide < normal * 0.6
