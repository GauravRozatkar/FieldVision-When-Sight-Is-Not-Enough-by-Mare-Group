"""Tests for the logic modules (tracks, spatial, priority, reliability) and the skeleton."""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.profiles import PROFILES, get_profile
from main import parse_args, parse_source
from occhio.capture import CameraStream
from occhio.metrics import FpsCounter

MODULES = ["capture", "detector", "depth", "tracks", "spatial", "reliability",
           "priority", "speech", "scene_llm", "overlay", "metrics"]


# ---------------------------------------------------------------- skeleton
@pytest.mark.parametrize("name", MODULES)
def test_module_importable_with_docstring(name: str) -> None:
    module = importlib.import_module(f"occhio.{name}")
    assert module.__doc__ and module.__doc__.strip()


# ---------------------------------------------------------------- profiles
@pytest.mark.parametrize("mode", ["street", "industrial"])
def test_profile_shape(mode: str) -> None:
    profile = get_profile(mode)
    assert profile["classes"]
    assert 0 < profile["danger_distance_m"] < profile["warn_distance_m"]
    assert set(profile["lines"]) == set(profile["lines_en"]) == {
        "ahead", "side", "approach", "ttc", "ttc_one", "clear", "unreliable", "mode", "describe_failed",
        "light_red", "light_green", "step", "read_failed", "listening", "command_unknown",
        "voice_unavailable", "zone", "sound_horn", "sound_siren", "sound_approach", "scene_none",
        "read_none", "read_result", "stop_sign"}


def test_profile_lines_format() -> None:
    for profile in PROFILES.values():
        assert profile["lines"]["ahead"].format(obj="Persona", dist=3, unit="metri")
        assert profile["lines"]["approach"].format(obj="Bici", side="sinistra")


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError):
        get_profile("underwater")


# --------------------------------------------------------------------- CLI
def test_parse_source() -> None:
    assert parse_source("0") == 0
    assert parse_source("1") == 1
    assert parse_source("clips/walk.mp4") == "clips/walk.mp4"


def test_parse_args_defaults_and_flags() -> None:
    args = parse_args([])
    assert (args.mode, args.source, args.no_voice) == ("street", 0, False)
    args = parse_args(["--mode", "industrial", "--source", "demo.mp4", "--no-voice"])
    assert (args.mode, args.source, args.no_voice) == ("industrial", "demo.mp4", True)


# ----------------------------------------------------------------- metrics
def test_fps_counter() -> None:
    counter = FpsCounter(window_s=1.0)
    assert counter.tick(now=0.0) == 0.0
    fps = 0.0
    for i in range(1, 61):
        fps = counter.tick(now=i / 30.0)
    assert fps == pytest.approx(30.0, rel=0.02)


# ----------------------------------------------------------------- capture
@pytest.fixture()
def video_file(tmp_path: Path) -> str:
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for i in range(30):
        writer.write(np.full((120, 160, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


def test_camera_stream_returns_only_new_frames(video_file: str) -> None:
    stream = CameraStream(video_file).start()
    try:
        indices: list[int] = []
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            frame = stream.read(timeout=0.5)
            if frame is None:
                if not stream.running:
                    break
                continue
            assert frame.image.shape == (120, 160, 3)
            indices.append(frame.index)
        assert indices, "no frames read"
        assert indices == sorted(set(indices)), "a frame was returned twice or out of order"
        assert stream.frames_grabbed == 30
    finally:
        stream.stop()


def test_camera_stream_drops_old_frames(video_file: str) -> None:
    stream = CameraStream(video_file).start()
    try:
        first = stream.read(timeout=2.0)
        assert first is not None
        time.sleep(0.4)  # a slow consumer: ~12 frames go by
        second = stream.read(timeout=2.0)
        assert second is not None
        assert second.index - first.index > 5, "stale frames were queued instead of dropped"
    finally:
        stream.stop()


def test_camera_stream_bad_source() -> None:
    with pytest.raises(RuntimeError):
        CameraStream("/no/such/file.mp4")


# ------------------------------------------------------------------ tracks
from occhio.detector import Detection  # noqa: E402
from occhio.tracks import TrackManager, TrackStatus  # noqa: E402

DT = 1 / 30  # 30 FPS


def det(track_id: int = 1, conf: float = 0.8, x: float = 100.0, cls_name: str = "person") -> Detection:
    """A fake 50x100 px detection whose left edge is at x."""
    return Detection(track_id, cls_name, conf, (x, 200.0, x + 50.0, 300.0))


def run(mgr: TrackManager, frames: list[list[Detection]], t0: float = 0.0) -> float:
    """Feed frames at 30 FPS starting at t0; return the time of the next frame."""
    t = t0
    for dets in frames:
        mgr.update(dets, t)
        t += DT
    return t


def test_new_track_is_uncertain_then_confirmed_after_5_frames() -> None:
    mgr = TrackManager()
    for i in range(4):
        mgr.update([det()], i * DT)
        assert mgr.tracks[1].status is TrackStatus.UNCERTAIN
    mgr.update([det()], 4 * DT)
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED
    assert mgr.tracks[1].frames_seen == 5


def test_low_confidence_track_stays_uncertain() -> None:
    mgr = TrackManager()
    run(mgr, [[det(conf=0.45)]] * 20)
    track = mgr.tracks[1]
    assert track.frames_seen == 20
    assert track.mean_conf == pytest.approx(0.45)
    assert track.status is TrackStatus.UNCERTAIN


def test_mean_conf_gate_uses_mean_not_last() -> None:
    mgr = TrackManager()
    run(mgr, [[det(conf=0.3)]] * 4 + [[det(conf=0.95)]])  # mean 0.43
    assert mgr.tracks[1].status is TrackStatus.UNCERTAIN


def test_flicker_never_confirmed_and_is_removed() -> None:
    mgr = TrackManager()
    t = run(mgr, [[det(track_id=7, conf=0.9)]] * 2)  # false detection: 2 frames only
    assert mgr.tracks[7].status is TrackStatus.UNCERTAIN
    t = run(mgr, [[]] * 3, t)  # briefly gone: kept invisibly
    assert 7 in mgr.tracks and not mgr.tracks[7].visible
    assert mgr.tracks[7].status is TrackStatus.UNCERTAIN  # never hidden: was never confirmed
    run(mgr, [[]] * 20, t)  # > uncertain_ttl_s
    assert 7 not in mgr.tracks


def test_uncertain_track_survives_one_missed_frame() -> None:
    mgr = TrackManager()
    run(mgr, [[det()], [det()], [], [det()], [det()], [det()]])
    assert mgr.tracks[1].frames_seen == 5
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED


def test_occlusion_20_frames_keeps_id_predicts_and_reconfirms() -> None:
    mgr = TrackManager()
    speed = 3.0  # px per frame -> 90 px/s to the right
    x = 100.0
    t = 0.0
    for _ in range(10):  # walking, confirmed
        mgr.update([det(x=x)], t)
        x += speed
        t += DT
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED
    last_x = x - speed
    frames_before = mgr.tracks[1].frames_seen

    for i in range(1, 21):  # behind the chair for 20 frames (0.67 s)
        tracks = mgr.update([], t)
        assert [tr.track_id for tr in tracks] == [1]
        track = mgr.tracks[1]
        assert track.status is TrackStatus.HIDDEN
        assert not track.visible
        assert track.box[0] == pytest.approx(last_x + speed * i, abs=0.5)  # constant velocity
        assert track.box[2] - track.box[0] == pytest.approx(50.0)  # size kept
        x += speed
        t += DT

    mgr.update([det(x=x, conf=0.5)], t)  # comes back with the same id, even with low conf
    track = mgr.tracks[1]
    assert track.status is TrackStatus.CONFIRMED
    assert track.visible and track.predicted_box is None
    assert track.box[0] == pytest.approx(x)
    assert track.frames_seen == frames_before + 1


def test_hidden_track_deleted_after_ttl() -> None:
    mgr = TrackManager()
    t = run(mgr, [[det()]] * 10)
    t = run(mgr, [[]] * 44, t)  # up to ~1.47 s lost
    assert mgr.tracks[1].status is TrackStatus.HIDDEN
    run(mgr, [[]] * 3, t)  # past 1.5 s
    assert 1 not in mgr.tracks


def test_back_after_deletion_starts_uncertain() -> None:
    mgr = TrackManager()
    t = run(mgr, [[det()]] * 10)
    t = run(mgr, [[]] * 50, t)
    mgr.update([det()], t)
    assert mgr.tracks[1].status is TrackStatus.UNCERTAIN
    assert mgr.tracks[1].frames_seen == 1


def test_thresholds_come_from_settings() -> None:
    mgr = TrackManager({"confirm_frames": 2, "confirm_mean_conf": 0.4, "hidden_ttl_s": 0.1})
    run(mgr, [[det(conf=0.45)]] * 2)
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED
    run(mgr, [[]] * 4, 2 * DT)  # 0.13 s > 0.1 s
    assert 1 not in mgr.tracks


def test_history_is_capped() -> None:
    mgr = TrackManager()
    run(mgr, [[det(x=float(i))] for i in range(100)])
    track = mgr.tracks[1]
    assert len(track.boxes) == 30 and len(track.confs) == 30
    assert track.frames_seen == 100


def test_several_tracks_independent() -> None:
    mgr = TrackManager()
    t = run(mgr, [[det(1), det(2, conf=0.3, cls_name="dog")]] * 6)
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED
    assert mgr.tracks[2].status is TrackStatus.UNCERTAIN
    run(mgr, [[det(2, conf=0.3, cls_name="dog")]], t)
    assert mgr.tracks[1].status is TrackStatus.HIDDEN
    assert mgr.tracks[2].visible


# ----------------------------------------------------------------- overlay
from occhio.metrics import StageTimes  # noqa: E402
from occhio.overlay import GREEN, GREY, YELLOW, draw_stats, draw_tracks  # noqa: E402


def test_overlay_colours_by_status() -> None:
    mgr = TrackManager()
    t = run(mgr, [[det(1, x=50), det(2, conf=0.3, x=300)]] * 6)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_tracks(image, mgr.tracks.values())
    assert tuple(image[250, 50]) == GREEN  # left edge of confirmed box
    assert tuple(image[250, 300]) == YELLOW  # left edge of uncertain box

    run(mgr, [[det(2, conf=0.3, x=300)]], t)  # 1 is now hidden
    image[:] = 0
    draw_tracks(image, mgr.tracks.values())
    column = image[200:301, 50]
    assert (column == GREY).all(axis=1).any() and not (column == GREY).all(), "should be dashed"


def test_overlay_skips_boxes_outside_frame() -> None:
    mgr = TrackManager()
    mgr.update([det(x=5000)], 0.0)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_tracks(image, mgr.tracks.values())
    draw_stats(image, 30.0, {"detect": 12.3})
    assert image.any()


def test_stage_times() -> None:
    st = StageTimes(alpha=0.5)
    st.add("detect", 10.0)
    st.add("detect", 20.0)
    assert st.smoothed["detect"] == pytest.approx(15.0)
    assert st.pop_window_means() == {"detect": pytest.approx(15.0)}
    assert st.pop_window_means() == {}


# ------------------------------------------------- re-identification (F3)
def confirmed_then_hidden(mgr: TrackManager, frames_hidden: int = 10, x0: float = 100.0,
                          speed: float = 3.0) -> tuple[float, float]:
    """Track 1 walks right for 10 frames, then hides. Returns (t, x where it would be now)."""
    x, t = x0, 0.0
    for _ in range(10):
        mgr.update([det(1, x=x)], t)
        x += speed
        t += DT
    for _ in range(frames_hidden):
        mgr.update([], t)
        x += speed
        t += DT
    assert mgr.tracks[1].status is TrackStatus.HIDDEN
    return t, x


def test_new_bytetrack_id_near_hidden_track_takes_its_id() -> None:
    mgr = TrackManager()
    t, x = confirmed_then_hidden(mgr)
    tracks = mgr.update([det(21, x=x + 20, conf=0.6)], t)  # ByteTrack gave a new id
    assert [tr.track_id for tr in tracks] == [1]
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED and mgr.tracks[1].visible
    mgr.update([det(21, x=x + 23)], t + DT)  # the alias sticks on later frames
    assert list(mgr.tracks) == [1] and mgr.tracks[1].box[0] == pytest.approx(x + 23)


def test_far_away_new_id_is_a_new_track() -> None:
    mgr = TrackManager()
    t, x = confirmed_then_hidden(mgr)
    mgr.update([det(21, x=x + 400)], t)  # gate = 1 x height (100 px)
    assert mgr.tracks[1].status is TrackStatus.HIDDEN
    assert mgr.tracks[21].status is TrackStatus.UNCERTAIN


def test_reid_needs_same_class_similar_height_and_conf() -> None:
    for new in (det(21, cls_name="dog"), det(21, conf=0.3),
                Detection(21, "person", 0.9, (100.0, 280.0, 150.0, 300.0))):  # 20 px tall
        mgr = TrackManager()
        t, x = confirmed_then_hidden(mgr)
        new = Detection(new.track_id, new.cls_name, new.conf,
                        (x, new.xyxy[1], x + 50.0, new.xyxy[3]))
        mgr.update([new], t)
        assert 21 in mgr.tracks, f"wrongly re-identified {new}"
        assert mgr.tracks[1].status is TrackStatus.HIDDEN


def test_reid_picks_nearest_hidden_track() -> None:
    mgr = TrackManager()
    t = 0.0
    for _ in range(10):
        mgr.update([det(1, x=100), det(2, x=200)], t)
        t += DT
    t = run(mgr, [[]] * 5, t)
    mgr.update([det(30, x=190)], t)
    assert mgr.tracks[2].visible and not mgr.tracks[1].visible
    assert 30 not in mgr.tracks


def test_uncertain_tracks_are_never_reidentified() -> None:
    mgr = TrackManager()
    t = run(mgr, [[det(1, conf=0.3)]] * 10)
    t = run(mgr, [[]] * 3, t)
    mgr.update([det(21)], t)
    assert 21 in mgr.tracks and 1 in mgr.tracks


def test_duplicate_ids_for_one_track_in_one_frame() -> None:
    mgr = TrackManager()
    t, x = confirmed_then_hidden(mgr)
    mgr.update([det(21, x=x)], t)  # 21 -> 1
    mgr.update([det(1, x=x + 3, conf=0.7), det(21, x=x + 3, conf=0.9)], t + DT)  # both at once
    assert list(mgr.tracks) == [1]
    assert mgr.tracks[1].conf == pytest.approx(0.9)  # the stronger box wins
    assert mgr.tracks[1].frames_seen == 12


def test_alias_dropped_when_track_deleted() -> None:
    mgr = TrackManager()
    t, x = confirmed_then_hidden(mgr)
    mgr.update([det(21, x=x)], t)
    t = run(mgr, [[]] * 50, t + DT)  # gone for good
    assert not mgr.tracks
    mgr.update([det(21, x=x)], t)
    assert list(mgr.tracks) == [21] and mgr.tracks[21].status is TrackStatus.UNCERTAIN


def test_hidden_prediction_uses_centre_velocity_and_keeps_size() -> None:
    mgr = TrackManager()
    t = 0.0
    for i in range(5):  # box shrinking on the right while the centre moves +5 px/frame
        x1, x2 = 100.0 + 10 * i, 200.0
        mgr.update([Detection(1, "person", 0.9, (x1, 200.0, x2, 300.0))], t)
        t += DT
    mgr.update([], t)
    x1, _, x2, _ = mgr.tracks[1].box
    assert x2 - x1 == pytest.approx(60.0)  # last size kept
    assert (x1 + x2) / 2 == pytest.approx(170.0 + 5.0)  # centre moved one more frame


def test_partly_occluded_boxes_ignored_for_prediction() -> None:
    mgr = TrackManager()
    t = 0.0
    for i in range(10):  # full 100 px tall box moving +3 px/frame
        mgr.update([det(1, x=100.0 + 3 * i)], t)
        t += DT
    # last two frames: only the legs are visible (short box, lower down)
    mgr.update([Detection(1, "person", 0.5, (130.0, 270.0, 180.0, 300.0))], t)
    t += DT
    mgr.update([Detection(1, "person", 0.4, (133.0, 280.0, 183.0, 300.0))], t)
    t += DT
    mgr.update([], t)  # hidden, 3 frames after the last full box (x=127 at frame 9)
    x1, y1, x2, y2 = mgr.tracks[1].box
    assert (y1, y2) == pytest.approx((200.0, 300.0))  # full height, no fake downward motion
    assert x1 == pytest.approx(127.0 + 3 * 3, abs=0.5)


def test_reid_when_person_stopped_behind_obstacle() -> None:
    mgr = TrackManager()
    t, x = confirmed_then_hidden(mgr, frames_hidden=40)  # predicted far to the right by now
    stopped_x = 100.0 + 3 * 9  # came out where they went in
    mgr.update([det(21, x=stopped_x)], t)
    assert list(mgr.tracks) == [1] and mgr.tracks[1].visible


def test_velocity_ignores_one_jumpy_box() -> None:
    mgr = TrackManager()
    t = 0.0
    for i in range(6):
        x = 100.0 + 3 * i + (60.0 if i == 4 else 0.0)  # one wrong box
        mgr.update([det(1, x=x)], t)
        t += DT
    vx, vy = mgr.tracks[1].centre_velocity(5)
    assert vx == pytest.approx(90.0) and vy == pytest.approx(0.0)


def test_reid_retried_while_new_track_is_unconfirmed() -> None:
    mgr = TrackManager()
    t, x = confirmed_then_hidden(mgr)
    mgr.update([det(21, x=x, conf=0.3)], t)  # too weak to re-identify yet
    assert 21 in mgr.tracks and mgr.tracks[1].status is TrackStatus.HIDDEN
    mgr.update([det(21, x=x + 3, conf=0.7)], t + DT)  # now strong enough
    assert list(mgr.tracks) == [1]
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED


def test_confirmed_track_is_never_merged_into_hidden_one() -> None:
    mgr = TrackManager()
    t = 0.0
    for _ in range(10):
        mgr.update([det(1, x=100), det(2, x=110)], t)  # two people side by side
        t += DT
    mgr.update([det(2, x=110)], t)  # 1 hides behind 2
    assert mgr.tracks[1].status is TrackStatus.HIDDEN
    assert mgr.tracks[2].status is TrackStatus.CONFIRMED and 2 in mgr.tracks


# ------------------------------------- lost memory + appearance re-identification
from occhio.tracks import appearance_distance, appearance_signature  # noqa: E402

RED, BLUE = (40, 40, 220), (220, 60, 40)  # BGR shirts


def scene(*people: tuple[float, tuple[int, int, int]]) -> np.ndarray:
    """Grey 640x480 image with one coloured 'person' (50x100 box at x, y=200) per entry."""
    image = np.full((480, 640, 3), 120, dtype=np.uint8)
    for x, colour in people:
        cv2.rectangle(image, (int(x), 200), (int(x) + 50, 300), colour, -1)
        image[260:300, int(x):int(x) + 50] = (30, 30, 30)  # dark trousers
    return image


def lose(mgr: TrackManager, colour=RED, x: float = 100.0, gone_frames: int = 60) -> float:
    """Confirmed person walks a bit, then is gone for `gone_frames`. Returns next time."""
    t = 0.0
    for _ in range(10):
        mgr.update([det(1, x=x)], t, scene((x, colour)))
        t += DT
    for _ in range(gone_frames):
        mgr.update([], t, scene())
        t += DT
    return t


def test_appearance_signature() -> None:
    red = appearance_signature(scene((100, RED)), (100, 200, 150, 300))
    red2 = appearance_signature(scene((300, RED)), (300, 200, 350, 300))
    blue = appearance_signature(scene((100, BLUE)), (100, 200, 150, 300))
    assert red is not None and red2 is not None and blue is not None
    assert appearance_distance(red, red2) < 0.05
    assert appearance_distance(red, blue) > 0.6
    assert appearance_signature(scene(), (10, 10, 14, 14)) is None  # too small
    assert appearance_signature(scene(), (700, 10, 900, 200)) is None  # outside the image


def test_lost_track_not_drawn_but_remembered() -> None:
    mgr = TrackManager()
    lose(mgr, gone_frames=60)  # 2 s > hidden_ttl_s
    assert 1 not in mgr.tracks
    assert 1 in mgr.lost


def test_same_person_back_after_5s_gets_old_id() -> None:
    mgr = TrackManager()
    t = lose(mgr, gone_frames=150)
    tracks = mgr.update([det(40, x=400)], t, scene((400, RED)))  # elsewhere, new ByteTrack id
    assert [tr.track_id for tr in tracks] == [1]
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED  # no 5-frame wait
    assert not mgr.lost
    assert any("back after" in e and "appearance" in e for e in mgr.events)
    mgr.update([det(40, x=403)], t + DT, scene((403, RED)))  # alias sticks
    assert list(mgr.tracks) == [1]


def test_different_person_gets_new_id() -> None:
    mgr = TrackManager()
    t = lose(mgr, colour=RED, gone_frames=90)
    mgr.update([det(40, x=100)], t, scene((100, BLUE)))  # same spot, other clothes
    assert 40 in mgr.tracks and mgr.tracks[40].status is TrackStatus.UNCERTAIN
    assert 1 in mgr.lost


def test_lost_memory_expires() -> None:
    mgr = TrackManager({"lost_memory_s": 3.0})
    t = lose(mgr, gone_frames=150)  # 5 s
    assert not mgr.lost
    mgr.update([det(40)], t, scene((100, RED)))
    assert list(mgr.tracks) == [40]


def test_match_only_in_first_second_of_new_track() -> None:
    mgr = TrackManager({"reid_window_s": 0.2})
    t = lose(mgr, gone_frames=60)
    for _ in range(10):  # a blurry/unknown version first: no match, becomes its own track
        mgr.update([det(40, x=300)], t, scene((300, BLUE)))
        t += DT
    mgr.update([det(40, x=300)], t, scene((300, RED)))  # now looks right, but too late
    assert 40 in mgr.tracks and 1 in mgr.lost


def test_bytetrack_reusing_lost_id_revives_it() -> None:
    mgr = TrackManager()
    t = lose(mgr, gone_frames=50)
    mgr.update([det(1, x=110)], t, scene((110, BLUE)))  # ByteTrack says same id: trust it
    assert mgr.tracks[1].status is TrackStatus.CONFIRMED and not mgr.lost


def test_no_image_means_no_lost_memory() -> None:
    mgr = TrackManager()
    t = run(mgr, [[det()]] * 10)
    run(mgr, [[]] * 60, t)
    assert not mgr.tracks and not mgr.lost


def test_partial_boxes_do_not_spoil_appearance() -> None:
    mgr = TrackManager()
    t = 0.0
    for _ in range(10):
        mgr.update([det(1, x=100)], t, scene((100, RED)))
        t += DT
    before = mgr.tracks[1].appearance.copy()
    for _ in range(10):  # only the dark trousers visible: short box
        mgr.update([Detection(1, "person", 0.9, (100.0, 260.0, 150.0, 300.0))], t, scene((100, RED)))
        t += DT
    assert appearance_distance(before, mgr.tracks[1].appearance) < 1e-6


def test_events_are_reported() -> None:
    mgr = TrackManager()
    t = 0.0
    events: list[str] = []
    for _ in range(10):
        mgr.update([det(1)], t, scene((100, RED)))
        events += mgr.events
        t += DT
    for _ in range(60):
        mgr.update([], t, scene())
        events += mgr.events
        t += DT
    assert events == ["#1 person confirmed", "#1 person hidden", "#1 person lost (remembered)"]


def test_default_language_is_english() -> None:
    from config.profiles import render
    for profile in PROFILES.values():
        assert profile["lang"] == "en"
    assert parse_args([]).lang is None  # None = the profile's language, English
    assert render(PROFILES["street"], PROFILES["street"]["lang"], "ahead", "person", 2.0) == "Person ahead, 2 metres"
