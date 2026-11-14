"""iPhone as the camera: finding it by name, rotation, portrait display and focal length."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.profiles import PROFILES
from occhio.cameras import find_camera, parse_avfoundation
from occhio.capture import CameraStream
from occhio.hud import CAM_H, CAM_W, fit_geometry
from occhio.spatial import SpatialReasoner

FFMPEG_OUTPUT = """
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] MacBook Air Camera
[AVFoundation indev @ 0x1] [1] MacBook Air Desk View Camera
[AVFoundation indev @ 0x1] [2] Mario's iPhone Camera
[AVFoundation indev @ 0x1] [3] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Air Microphone
"""


def test_parse_camera_list_skips_screens_and_audio() -> None:
    assert parse_avfoundation(FFMPEG_OUTPUT) == [(0, "MacBook Air Camera"), (1, "MacBook Air Desk View Camera"),
                                                 (2, "Mario's iPhone Camera")]


def test_find_iphone_by_name_not_by_number() -> None:
    cams = parse_avfoundation(FFMPEG_OUTPUT)
    assert find_camera("iphone", cams) == 2  # not 1: that is the Desk View camera
    assert find_camera("desk view", cams) == 1
    assert find_camera("android", cams) is None


def test_resolve_source(monkeypatch) -> None:
    import main
    import occhio.cameras as cameras
    monkeypatch.setattr(cameras, "list_cameras", lambda: parse_avfoundation(FFMPEG_OUTPUT))
    assert main.resolve_source("iphone") == 2
    assert main.resolve_source("cam:desk") == 1
    assert main.resolve_source(0) == 0 and main.resolve_source("walk.mp4") == "walk.mp4"
    monkeypatch.setattr(cameras, "list_cameras", lambda: [(0, "MacBook Air Camera")])
    assert main.resolve_source("iphone") is None


def test_rotate_frames(tmp_path: Path) -> None:
    path = str(tmp_path / "land.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 90))
    for _ in range(10):
        w.write(np.zeros((90, 160, 3), np.uint8))
    w.release()
    stream = CameraStream(path, rotate=90).start()
    frame = stream.read(timeout=2.0)
    stream.stop()
    assert frame is not None and frame.image.shape[:2] == (160, 90)  # now portrait


def test_portrait_fits_without_stretching() -> None:
    s, ox, oy, w, h = fit_geometry(1920, 1080)  # upright iPhone
    assert h == CAM_H and w == pytest.approx(CAM_H * 1080 / 1920, abs=1) and ox > 0 and oy == 0
    assert fit_geometry(720, 1280) == (CAM_W / 1280, 0, 0, CAM_W, CAM_H)  # landscape fills it


def test_focal_length_uses_the_long_side() -> None:
    r = SpatialReasoner(PROFILES["street"], hfov_deg=70)
    assert r.focal_px(1080, 1920) == pytest.approx(r.focal_px(1920, 1080))  # same lens, turned
    # same person, same distance, landscape or portrait -> same distance estimate
    h = r.focal_px(1920, 1080) * 1.70 / 3.0
    box_l = (900, 900 - h, 1000, 900)
    box_p = (500, 1700 - h, 600, 1700)
    assert r.size_distance("person", box_l, (1080, 1920)) == pytest.approx(3.0)
    assert r.size_distance("person", box_p, (1920, 1080)) == pytest.approx(3.0)
