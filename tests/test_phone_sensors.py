"""Phone sensors (tilt, walking speed), walls from depth, and the phone page's voice/motion code."""
from __future__ import annotations

import math

import numpy as np
import pytest

from config.profiles import PROFILES, render
from occhio.depth import DepthFrame
from occhio.detector import Detection
from occhio.free_space import FreeSpace
from occhio.phone_link import PAGE, PhoneCamera
from occhio.priority import PriorityPlanner
from occhio.reliability import ReliabilityState
from occhio.spatial import SpatialInfo, SpatialReasoner
from occhio.tracks import TrackManager

STREET = PROFILES["street"]
OK = ReliabilityState(1.0, 1.0, 150, 500, 60, 0.9, False, True, "ok")
FH, FW = 720, 1280


def test_floor_distance_uses_the_phone_tilt() -> None:
    r = SpatialReasoner(STREET, hfov_deg=70)
    f = r.focal_px(FW, FH)
    level = r.floor_distance_at_row(FH / 2 + 200, (FH, FW))
    assert level == pytest.approx(1.2 * f / 200, rel=1e-3)
    r.pitch_deg = 20  # tilted down: the same row is closer
    tilted = r.floor_distance_at_row(FH / 2 + 200, (FH, FW))
    assert tilted == pytest.approx(1.2 / math.tan(math.atan2(200, f) + math.radians(20)), rel=1e-3)
    assert tilted < level
    r.pitch_deg = -10  # tilted up: the horizon moves down
    assert r.floor_distance_at_row(FH / 2 + 20, (FH, FW)) is None


def test_walking_speed_makes_still_obstacles_approach() -> None:
    r = SpatialReasoner(STREET, hfov_deg=70)
    r.user_speed_mps = 1.2
    mgr = TrackManager({"class_confirm": {}})
    bench = (560, 380, 720, 560)  # still bench in the path
    infos = []
    for i in range(6):
        infos = r.update(mgr.update([Detection(1, "bench", 0.9, bench), Detection(2, "person", 0.9, (100, 200, 200, 700))],
                                    i / 30), None, (FH, FW), i / 30)
    by_cls = {i.cls_name: i for i in infos}
    assert by_cls["bench"].approach_mps == pytest.approx(0.96)  # 0.8 x walking speed
    assert by_cls["person"].approach_mps == pytest.approx(0.0, abs=0.05)  # people: vision only


def test_walking_warns_earlier() -> None:
    far = SpatialInfo(1, "bench", True, 0.9, "ahead", 12, 3.6, "size", 0.0, 0.0, 0.0, False, None)
    p = PriorityPlanner(STREET, "en")
    assert p.plan([far], OK, 0.0) is None  # 3.6 m: beyond the 3 m warning distance
    p.extra_warn_m = 1.0  # walking at 1 m/s
    msg = p.plan([far], OK, 0.1)
    assert msg is not None and msg.text == "Bench ahead, 4 metres"


def test_phone_reports_sensors_in_the_query() -> None:
    phone = PhoneCamera(use_https=False)
    phone._read_query("/frame?voice=1&speed=1.25&turn=40&pitch=12.5")
    assert phone.phone_voice_ok and phone.walk_speed == 1.25 and phone.pitch_deg == 12.5 and phone.turn_dps == 40
    phone._read_query("/frame?voice=0&speed=abc")  # garbage is ignored, not a crash
    assert not phone.phone_voice_ok and phone.walk_speed == 1.25


def test_page_unlocks_voice_and_reads_motion() -> None:
    page = PhoneCamera(use_https=False, strings={"test_voice": "Test voice", "voice_on": "Voice on"}).page().decode()
    for needle in ("devicemotion", "requestPermission", "Test voice", "Voice on", "voice=", "setTimeout(() => utter(text), 250)"):
        assert needle in page, needle
    assert "__" not in page.split("<script>")[0]  # every placeholder filled


# ------------------------------------------------------------------ walls
def depth_map(wall_m, pitch=0.0, seed=0, scale=6.0) -> np.ndarray:
    r = SpatialReasoner(STREET, hfov_deg=70)
    f = r.focal_px(FW, FH)
    rows = (np.arange(252) + 0.5) * FH / 252
    angle = np.arctan2(rows - FH / 2, f) + math.radians(pitch)
    z = np.where(angle > math.radians(0.5), 1.2 / np.tan(np.maximum(angle, 1e-3)), 50.0)
    if wall_m:
        z = np.minimum(z, wall_m)
    noise = 1 + 0.03 * np.random.default_rng(seed).standard_normal((252, 448))
    return ((scale / z)[:, None] * np.ones((1, 448)) * noise).astype(np.float32)


def run_walls(walls, pitch=0.0):
    r = SpatialReasoner(STREET, hfov_deg=70)
    r.pitch_deg = pitch
    fs, t, found = FreeSpace(), 0.0, []
    for wall in walls:
        for _ in range(3):
            d = fs.update(DepthFrame(depth_map(wall, pitch, seed=int(t)), (FH, FW), t, 1.0), (FH, FW),
                          r.floor_distance_at_row)
            t += 1
        found.append(d)
    return found


def test_walking_toward_a_wall() -> None:
    walls = [4.0, 3.2, 2.4, 1.6, 1.0]
    for true, found in zip(walls, run_walls(walls)):
        assert found == pytest.approx(true, rel=0.08), (true, found)  # learned the scale from the floor


def test_wall_with_a_tilted_phone() -> None:
    assert run_walls([2.0], pitch=15)[0] == pytest.approx(2.0, rel=0.08)


def test_open_floor_is_not_a_wall() -> None:
    assert run_walls([None, None]) == [None, None]
    assert run_walls([None], pitch=15) == [None]


def test_close_wall_without_any_scale_says_nothing() -> None:
    assert run_walls([1.5]) == [None]  # no floor ever seen: no number rather than a wrong one


def test_wall_is_spoken_like_any_obstacle() -> None:
    wall = SpatialInfo(-1, "obstacle", True, 0.8, "ahead", 12, 1.2, "depth", 0.0, 0.0, 0.0, False, None)
    msg = PriorityPlanner(STREET, "en").plan([wall], OK, 0.0)
    assert msg.urgent and msg.text == "Obstacle ahead, 1 metre"
    assert render(STREET, "it", "ahead", "obstacle", 2.0) == "Ostacolo davanti, 2 metri"



def test_urgent_flag_is_a_plain_bool_even_with_numpy_distances() -> None:
    wall = SpatialInfo(-1, "obstacle", True, 0.8, "ahead", 12, np.float64(1.2), "depth", 0.0, 0.0, 0.0, False, None)
    msg = PriorityPlanner(STREET, "en").plan([wall], OK, 0.0)
    assert type(msg.urgent) is bool and msg.urgent
