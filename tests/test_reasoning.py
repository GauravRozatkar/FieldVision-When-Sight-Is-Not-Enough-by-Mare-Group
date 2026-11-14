"""Tests for Phase 2 logic: profiles text, spatial, reliability, priority, speech, depth plumbing."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from config.profiles import PROFILES, format_distance, render
from occhio.depth import DepthEstimator, DepthFrame
from occhio.detector import Detection
from occhio.priority import PriorityPlanner
from occhio.reliability import ReliabilityMonitor, ReliabilityState, ramp
from occhio.spatial import (Calibration, SpatialInfo, SpatialReasoner, clock_of, fit_slope,
                            robust_slope, side_of)
from occhio.speech import Speaker
from occhio.tracks import TrackManager

STREET = PROFILES["street"]
INDUSTRIAL = PROFILES["industrial"]
FRAME = (720, 1280)  # h, w
DT = 1 / 30


# ------------------------------------------------------------------ profile text
def test_format_distance() -> None:
    assert format_distance(2.04, "it") == "2"
    assert format_distance(1.7, "it") == "1,5"
    assert format_distance(1.7, "en") == "1.5"
    assert format_distance(3.6, "it") == "4"
    assert format_distance(0.1, "it") == "0,5"


def test_render_lines() -> None:
    assert render(STREET, "it", "ahead", "person", 2.0) == "Persona davanti, 2 metri"
    assert render(STREET, "it", "approach", "bicycle", side="left") == "Bici in arrivo da sinistra"
    assert render(STREET, "it", "side", "car", 2.6, "right") == "Auto a destra, 2,5 metri"
    assert render(INDUSTRIAL, "it", "ahead", "person", 4.2) == "Attenzione: persona a 4 metri"
    assert render(STREET, "en", "ahead", "person", 3.0) == "Person ahead, 3 metres"
    assert render(STREET, "it", "unreliable") == "Visibilità scarsa, guida non affidabile"


# ---------------------------------------------------------------------- spatial
def test_side_thirds() -> None:
    assert [side_of(x, 1200) for x in (100, 399, 600, 801, 1100)] == ["left", "left", "ahead", "right", "right"]


def test_clock_positions() -> None:
    f = 600.0
    assert clock_of(640, 1280, f) == 12
    assert clock_of(640 + f * np.tan(np.radians(30)), 1280, f) == 1
    assert clock_of(640 - f * np.tan(np.radians(30)), 1280, f) == 11
    assert clock_of(640 + f * 10, 1280, f) == 2  # clamped
    assert clock_of(640 - f * 10, 1280, f) == 10


def test_robust_slope_ignores_outliers() -> None:
    pts = [(t / 10, 5 - 1.5 * t / 10) for t in range(10)]
    pts[4] = (0.4, 9.0)  # one wild depth value
    pts[7] = (0.7, 0.5)
    assert robust_slope(pts) == pytest.approx(-1.5)
    assert fit_slope(pts) != pytest.approx(-1.5, abs=0.3)  # least squares would be thrown off
    assert robust_slope([(0.0, 1.0)]) == 0.0


def test_fit_slope() -> None:
    assert fit_slope([(t / 10, 5 - 1.5 * t / 10) for t in range(10)]) == pytest.approx(-1.5)
    assert fit_slope([(0.0, 1.0)]) == 0.0


def person_box(dist_m: float, focal: float, cx: float = 640, bottom: float = 700) -> tuple:
    """Full-body person box (1.70 m tall) at dist_m."""
    h = focal * 1.70 / dist_m
    w = h / 3
    return (cx - w / 2, bottom - h, cx + w / 2, bottom)


def feed(reasoner: SpatialReasoner, mgr: TrackManager, boxes: list[tuple], t0: float = 0.0,
         depth_for=None, track_id: int = 1, cls: str = "person") -> list[SpatialInfo]:
    infos: list[SpatialInfo] = []
    t = t0
    for box in boxes:
        tracks = mgr.update([Detection(track_id, cls, 0.9, box)], t)
        infos = reasoner.update(tracks, depth_for(box, t) if depth_for else None, FRAME, t)
        t += DT
    return infos


def test_size_distance_full_body_and_cut_person() -> None:
    r = SpatialReasoner(STREET, calibration=Calibration(focal_px=900, frame_width=1280, fixed=True))
    assert r.size_distance("person", person_box(3.0, 900), FRAME) == pytest.approx(3.0)
    # sitting at a desk: box cut by the bottom edge -> shoulder width 0.45 m
    assert r.size_distance("person", (400, 100, 850, 720), FRAME) == pytest.approx(900 * 0.45 / 450)
    assert r.size_distance("unicorn", (0, 0, 10, 10), FRAME) is None


def test_focal_from_fov_scales_with_width() -> None:
    r = SpatialReasoner(STREET, hfov_deg=90)
    assert r.focal_px(1280) == pytest.approx(640)
    cal = SpatialReasoner(STREET, calibration=Calibration(focal_px=1000, frame_width=1280, fixed=True))
    assert cal.focal_px(640) == pytest.approx(500)


def test_walking_toward_camera_gives_approach_speed_and_ttc() -> None:
    focal = 900.0
    r = SpatialReasoner(STREET, calibration=Calibration(focal_px=focal, frame_width=1280, fixed=True))
    mgr = TrackManager()
    speed = 1.2  # m/s toward the camera
    boxes = [person_box(4.0 - speed * i * DT, focal) for i in range(20)]
    info = feed(r, mgr, boxes)[0]
    assert info.distance_source == "size"
    assert info.distance_m == pytest.approx(4.0 - speed * 18 * DT, rel=0.01)  # median of last 3
    assert info.approach_mps == pytest.approx(speed, rel=0.05)
    assert info.ttc_s == pytest.approx(info.distance_m / info.approach_mps)
    assert info.side == "ahead" and info.clock == 12 and not info.fast


def test_standing_still_is_not_approaching() -> None:
    r = SpatialReasoner(STREET, calibration=Calibration(focal_px=900, frame_width=1280, fixed=True))
    info = feed(r, TrackManager(), [person_box(2.0, 900)] * 15)[0]
    assert info.approach_mps == pytest.approx(0.0, abs=1e-6) and info.ttc_s is None


def test_fast_lateral_bike() -> None:
    focal = 900.0
    r = SpatialReasoner(STREET, calibration=Calibration(focal_px=focal, frame_width=1280, fixed=True))
    # bike 1.05 m tall at 3 m, crossing at 5 m/s = 1500 px/s at this distance
    h = focal * 1.05 / 3.0
    boxes = [(100 + 50 * i, 500 - h, 100 + 50 * i + 1.6 * h, 500) for i in range(10)]
    info = feed(r, TrackManager(), boxes, cls="bicycle")[0]
    assert info.lateral_mps == pytest.approx(5.0, rel=0.05)
    assert info.fast


def flat_depth(disparity: float):
    return lambda box, t: DepthFrame(np.full((90, 160), disparity, dtype=np.float32), FRAME, t, 5.0)


def test_whole_person_uses_size_cut_person_uses_depth() -> None:
    r = SpatialReasoner(STREET, calibration=Calibration(depth_k=6.0, focal_px=900, frame_width=1280, fixed=True))
    whole = feed(r, TrackManager(), [person_box(5.0, 900)] * 6, depth_for=flat_depth(3.0))[0]
    assert whole.distance_source == "size" and whole.distance_m == pytest.approx(5.0)
    cut = (400.0, 0.0, 850.0, 720.0)  # right in front of the camera: box cut top and bottom
    near = feed(SpatialReasoner(STREET, calibration=r.calibration), TrackManager(), [cut] * 6,
                depth_for=flat_depth(3.0))[0]
    assert near.distance_source == "size" and near.distance_m == pytest.approx(0.9)  # shoulders closer
    far_depth = feed(SpatialReasoner(STREET, calibration=r.calibration), TrackManager(), [cut] * 6,
                     depth_for=flat_depth(20.0))[0]
    assert far_depth.distance_source == "depth" and far_depth.distance_m == pytest.approx(0.3)


def test_cut_box_never_farther_than_its_visible_height_allows() -> None:
    cal = Calibration(depth_k=6.0, focal_px=900, frame_width=1280, fixed=True)
    r = SpatialReasoner(STREET, calibration=cal)
    cut = (600.0, 0.0, 640.0, 700.0)  # narrow (bad width), cut at the top; depth says 6 m
    info = feed(r, TrackManager(), [cut] * 6, depth_for=flat_depth(1.0))[0]
    assert info.distance_m == pytest.approx(900 * 1.70 / 700)  # 2.19 m upper bound wins


def test_cut_person_without_depth_uses_shoulder_width() -> None:
    r = SpatialReasoner(STREET, calibration=Calibration(focal_px=900, frame_width=1280, fixed=True))
    info = feed(r, TrackManager(), [(400.0, 0.0, 850.0, 720.0)] * 6)[0]
    assert info.distance_source == "size" and info.distance_m == pytest.approx(900 * 0.45 / 450)


def test_auto_calibration_learns_depth_scale_from_full_body_person() -> None:
    focal = 900.0
    cal = Calibration(focal_px=focal, frame_width=1280)
    r = SpatialReasoner(STREET, settings={"auto_calibrate_rate": 1.0}, calibration=cal)
    true_k = 8.0

    def depth_for(box, t):  # a perfect depth model with unknown scale true_k
        x1, y1, x2, y2 = box
        dist = focal * 1.70 / (y2 - y1) if y1 > 3 else 2.0
        return DepthFrame(np.full((90, 160), true_k / dist, dtype=np.float32), FRAME, t, 5.0)

    feed(r, TrackManager(), [person_box(3.0, focal)] * 8, depth_for=depth_for)
    assert cal.depth_k == pytest.approx(true_k, rel=0.01)
    near = feed(r, TrackManager(), [(600.0, 0.0, 700.0, 720.0)] * 4, depth_for=depth_for, track_id=2)[0]
    assert near.distance_source == "depth" and near.distance_m == pytest.approx(2.0, rel=0.01)


def test_distance_is_median_of_last_three() -> None:
    r = SpatialReasoner(STREET, calibration=Calibration(focal_px=900, frame_width=1280, fixed=True))
    boxes = [person_box(3.0, 900)] * 4 + [person_box(1.0, 900)]  # one glitch box
    assert feed(r, TrackManager(), boxes)[0].distance_m == pytest.approx(3.0)


def test_samples_dropped_for_vanished_tracks() -> None:
    r = SpatialReasoner(STREET)
    mgr = TrackManager()
    feed(r, mgr, [person_box(3, 900)] * 5)
    r.update([], None, FRAME, 10.0)
    assert not r._samples


def test_depth_median_in_box_maps_coordinates() -> None:
    disp = np.zeros((72, 128), dtype=np.float32)
    disp[36:72, 64:128] = 4.0  # bottom-right quarter
    df = DepthFrame(disp, (720, 1280), 0.0, 1.0)
    assert df.median_in_box((700, 400, 1200, 700)) == pytest.approx(4.0)
    assert df.median_in_box((0, 0, 500, 300)) is None  # disparity 0 = unknown
    assert df.median_in_box((5000, 5000, 5100, 5100)) is None


# ------------------------------------------------------------------ reliability
def webcam_like(mean: float = 150, contrast: float = 60, seed: int = 0) -> np.ndarray:
    """Room-like frame: big shapes (std ~contrast, like the real webcam ~60) plus fine texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((360, 640), dtype=np.float32)
    for _ in range(25):
        x, y = rng.integers(0, 640), rng.integers(0, 360)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(40, 250)), int(y + rng.integers(40, 200))),
                      float(rng.normal(0, 1)), -1)
    img += rng.normal(0, 0.15, img.shape).astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6) * contrast + mean
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_ramp() -> None:
    assert ramp(0, 10, 20) == 0 and ramp(15, 10, 20) == 0.5 and ramp(30, 10, 20) == 1


def test_good_image_is_reliable() -> None:
    mon = ReliabilityMonitor()
    for i in range(30):
        state = mon.update(webcam_like(seed=i))
    assert state.reliable and state.score > 0.9 and state.reason == "ok"


def test_covered_lens_unreliable_within_one_second_and_recovers() -> None:
    mon = ReliabilityMonitor()
    for i in range(30):
        mon.update(webcam_like(seed=i))
    covered = np.full((360, 640, 3), 4, dtype=np.uint8)
    frames = 0
    while mon.update(covered).reliable:
        frames += 1
        assert frames < 25, "should be unreliable within 1 s at 25 FPS"
    state = mon.update(covered)
    assert state.blocked and state.reason == "blocked"
    for i in range(30):
        state = mon.update(webcam_like(seed=i))
    assert state.reliable


@pytest.mark.parametrize("make, reason", [
    (lambda: webcam_like(mean=14, contrast=14), "dark"),
    (lambda: cv2.GaussianBlur(webcam_like(), (0, 0), 12), "blur"),
    (lambda: np.full((360, 640, 3), 255, dtype=np.uint8), "blocked"),
])
def test_bad_images(make, reason) -> None:
    mon = ReliabilityMonitor()
    for _ in range(20):
        state = mon.update(make())
    assert not state.reliable and state.reason == reason


def test_hysteresis() -> None:
    mon = ReliabilityMonitor({"ema_alpha": 1.0})
    mon.update(np.full((90, 160, 3), 4, dtype=np.uint8))
    assert not mon.reliable
    mon.score = 0.45  # between thresholds: stays unreliable
    mon.alpha = 0.0
    assert not mon.update(webcam_like()).reliable


# --------------------------------------------------------------------- priority
def info(track_id: int = 1, dist: float | None = 2.0, approach: float = 0.0, side: str = "ahead",
         confirmed: bool = True, conf: float = 0.9, cls: str = "person") -> SpatialInfo:
    ttc = dist / approach if (dist is not None and approach >= 0.3) else None
    return SpatialInfo(track_id, cls, confirmed, conf, side, 12, dist, "size", approach, 0.0,
                       abs(approach), False, ttc)


OK = ReliabilityState(1.0, 1.0, 150, 500, 60, 0.9, False, True, "ok")
BAD = ReliabilityState(0.1, 0.0, 3, 1, 1, None, True, False, "blocked")


def test_person_ahead_message() -> None:
    p = PriorityPlanner(STREET, "it")
    msg = p.plan([info(dist=2.0)], OK, 0.0)
    assert msg is not None and msg.text == "Persona davanti, 2 metri" and msg.kind == "ahead" and not msg.urgent


def test_only_confirmed_within_warn_distance() -> None:
    p = PriorityPlanner(STREET, "it")
    assert p.plan([info(confirmed=False), info(2, dist=3.5), info(3, dist=None)], OK, 0.0) is None


def test_most_dangerous_wins() -> None:
    p = PriorityPlanner(STREET, "it")
    msg = p.plan([info(1, dist=2.8), info(2, dist=2.5, approach=0.8, side="left", cls="bicycle")], OK, 0.0)
    assert msg is not None and msg.track_id == 2 and msg.text == "Bici in arrivo da sinistra"


def test_danger_close_or_low_ttc_is_urgent() -> None:
    p = PriorityPlanner(STREET, "it")
    close = p.plan([info(dist=1.2)], OK, 0.0)
    assert close is not None and close.kind == "danger" and close.urgent and close.text == "Persona davanti, 1 metro"
    p2 = PriorityPlanner(STREET, "it")
    fast = p2.plan([info(dist=2.8, approach=2.0)], OK, 0.0)  # ttc 1.4 s
    assert fast is not None and fast.kind == "danger"


def test_cooldown_and_same_text() -> None:
    p = PriorityPlanner(STREET, "it")
    assert p.plan([info(dist=2.0)], OK, 0.0) is not None
    assert p.plan([info(dist=2.0)], OK, 1.0) is None  # cooldown 3 s
    assert p.plan([info(dist=2.0)], OK, 4.0) is None  # same sentence within 10 s
    msg = p.plan([info(dist=3.0)], OK, 4.0)  # distance changed
    assert msg is not None and msg.text == "Persona davanti, 3 metri"
    assert p.plan([info(dist=2.0)], OK, 15.0) is not None


def test_danger_skips_cooldown_but_not_every_frame() -> None:
    p = PriorityPlanner(STREET, "it")
    assert p.plan([info(dist=2.0)], OK, 0.0) is not None
    assert p.plan([info(dist=1.2)], OK, 0.1) is None  # 1.2 s gap after the last sentence ...
    assert p.plan([info(dist=1.2)], OK, 1.3) is not None  # ... then the new danger at once
    assert p.plan([info(dist=1.1)], OK, 1.4) is None  # not repeated every frame
    assert p.plan([info(dist=0.6)], OK, 2.9) is not None  # 0.6 m closer: announced again


def test_unchanged_danger_repeats_only_every_6s() -> None:
    p = PriorityPlanner(STREET, "it")
    assert p.plan([info(dist=0.5)], OK, 0.0).text == "Persona davanti, 0,5 metri"
    for t in (1.6, 3.0, 5.9):  # someone standing still right in front
        assert p.plan([info(dist=0.5)], OK, t) is None
    assert p.plan([info(dist=0.5)], OK, 6.1) is not None
    assert p.plan([info(dist=1.0)], OK, 7.7) is None  # moved *away*: nothing new to say
    assert p.plan([info(dist=1.0)], OK, 12.2) is not None  # reminder after 6 s


def test_cooldown_is_per_track() -> None:
    p = PriorityPlanner(STREET, "it")
    assert p.plan([info(1, dist=2.0)], OK, 0.0) is not None
    assert p.plan([info(1, dist=2.0), info(2, dist=2.5, cls="dog", side="right")], OK, 0.5) is None  # pacing
    msg = p.plan([info(1, dist=2.0), info(2, dist=2.5, cls="dog", side="right")], OK, 2.6)
    assert msg is not None and msg.track_id == 2 and msg.text == "Cane a destra, 2,5 metri"


def test_unreliable_says_line_once_then_every_5s_and_blocks_warnings() -> None:
    p = PriorityPlanner(STREET, "it")
    msg = p.plan([info(dist=1.0)], BAD, 0.0)
    assert msg is not None and msg.kind == "unreliable" and msg.urgent
    assert msg.text == "Visibilità scarsa, guida non affidabile"
    assert p.plan([info(dist=1.0)], BAD, 1.0) is None
    assert p.plan([info(dist=1.0)], BAD, 5.1) is not None
    assert p.plan([info(dist=1.0)], OK, 5.2) is not None  # back to normal warnings


def test_industrial_profile_and_english() -> None:
    p = PriorityPlanner(INDUSTRIAL, "it")
    assert p.plan([info(dist=4.0)], OK, 0.0).text == "Attenzione: persona a 4 metri"
    e = PriorityPlanner(STREET, "en")
    assert e.plan([info(dist=2.0)], OK, 0.0).text == "Person ahead, 2 metres"


# ----------------------------------------------------------------------- speech
class FakeProcess:
    """Stands in for a `say` process: 'speaks' for `duration` seconds."""

    def __init__(self, text: str, duration: float, log: list) -> None:
        self.text = text
        self._done = threading.Event()
        self.killed = False
        log.append(self)
        threading.Timer(duration, self._done.set).start()

    def wait(self) -> int:
        self._done.wait()
        return 0

    def terminate(self) -> None:
        self.killed = True
        self._done.set()

    def poll(self) -> int | None:
        return 0 if self._done.is_set() else None


def fake_speaker(duration: float = 0.3, **kw) -> tuple[Speaker, list[FakeProcess]]:
    log: list[FakeProcess] = []
    return Speaker(runner=lambda text: FakeProcess(text, duration, log), **kw).start(), log


def test_say_never_blocks_and_speaks_in_order() -> None:
    sp, log = fake_speaker(0.1)
    t0 = time.perf_counter()
    sp.say("uno")
    sp.say("due")
    assert time.perf_counter() - t0 < 0.01
    assert sp.wait_idle(2)
    assert [p.text for p in log] == ["uno", "due"]
    sp.stop()


def test_queue_drops_oldest_normal_messages() -> None:
    sp, log = fake_speaker(0.3)
    sp.say("a")
    deadline = time.perf_counter() + 1
    while not log and time.perf_counter() < deadline:  # 'a' is playing
        time.sleep(0.005)
    for word in ("b", "c", "d"):
        sp.say(word)
    assert sp.wait_idle(3)
    assert [p.text for p in log] == ["a", "c", "d"]  # 'a' was playing; queue kept the latest 2
    sp.stop()


def test_urgent_interrupts_and_clears_queue() -> None:
    sp, log = fake_speaker(1.0)
    sp.say("lungo")
    time.sleep(0.05)
    sp.say("in coda")
    sp.say("Pericolo", urgent=True)
    assert sp.wait_idle(3)
    assert [p.text for p in log] == ["lungo", "Pericolo"]
    assert log[0].killed
    sp.stop()


def test_disabled_speaker_only_reports() -> None:
    heard: list[tuple[str, bool]] = []
    sp, log = fake_speaker(enabled=False, on_spoken=lambda t, u: heard.append((t, u)))
    sp.say("ciao", urgent=True)
    time.sleep(0.05)
    assert heard == [("ciao", True)] and not log
    sp.stop()


# ------------------------------------------------------------------------ depth
def test_depth_estimator_every_nth_frame_and_skips_when_busy() -> None:
    calls: list[tuple] = []
    gate = threading.Event()

    def slow_infer(small: np.ndarray) -> np.ndarray:
        calls.append(small.shape)
        gate.wait(2)
        return np.ones(small.shape[:2], dtype=np.float32)

    est = DepthEstimator(slow_infer, {"every_n": 3, "input_size": (64, 36)}).start()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    accepted = [est.submit(frame, i / 30) for i in range(7)]
    assert accepted == [True, False, False, False, False, False, False]  # frame 3, 6 skipped: busy
    assert est.skipped_busy == 2
    gate.set()
    deadline = time.perf_counter() + 2
    while est.latest is None and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert est.latest is not None and est.latest.frame_shape == (720, 1280)
    assert calls == [(36, 64, 3)]
    est.stop()


def test_depth_errors_do_not_kill_the_thread() -> None:
    def broken(small):
        raise RuntimeError("boom")

    est = DepthEstimator(broken, {"every_n": 1, "input_size": (32, 18)}).start()
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    est.submit(frame, 0.0)
    time.sleep(0.1)
    assert est.errors == 1 and est._thread.is_alive()
    assert est.submit(frame, 0.1)
    est.stop()


def test_one_glitch_frame_cannot_fake_a_danger() -> None:
    focal = 900.0
    r = SpatialReasoner(STREET, calibration=Calibration(focal_px=focal, frame_width=1280, fixed=True))
    p = PriorityPlanner(STREET, "it")
    boxes = [person_box(2.5, focal)] * 12 + [person_box(1.9, focal), person_box(2.5, focal)]
    mgr = TrackManager()
    t = 0.0
    for box in boxes:
        infos = r.update(mgr.update([Detection(1, "person", 0.9, box)], t), None, FRAME, t)
        msg = p.plan(infos, OK, t)
        assert msg is None or msg.kind != "danger", f"false danger at t={t:.2f}: {infos[0]}"
        t += DT


def test_singular_metre() -> None:
    assert render(STREET, "it", "ahead", "person", 1.1) == "Persona davanti, 1 metro"
    assert render(STREET, "en", "ahead", "person", 0.9) == "Person ahead, 1 metre"
    assert render(STREET, "it", "ahead", "person", 2.0) == "Persona davanti, 2 metri"
    assert render(INDUSTRIAL, "it", "ahead", "person", 1.0) == "Attenzione: persona a 1 metro"
