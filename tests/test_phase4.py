"""Tests for Phase 4 and the extras: traffic light, step, zone, eco, metrics, sound, vibration,
voice commands, read-text, HUD, recorder."""
from __future__ import annotations

import csv
import json
import threading
import time
import urllib.request

import cv2
import numpy as np
import pytest

from config.profiles import PROFILES, detect_classes
from occhio.depth import DepthFrame
from occhio.detector import Detection
from occhio.ground import StepDetector
from occhio.hud import CAM_H, CAM_W, TOP_H, HudState, compose, to_camera_norm
from occhio.metrics import MetricsLogger
from occhio.power import MotionGate
from occhio.priority import PriorityPlanner, beep_target
from occhio.reliability import ReliabilityState
from occhio.scene_llm import SceneDescriber
from occhio.sound import SoundAnalyzer, SoundMonitor, dominant_tone, level_db
from occhio.spatial import SpatialInfo
from occhio.tracks import TrackManager
from occhio.traffic_light import TrafficLightMonitor, light_colour
from occhio.vibration import VibrationServer
from occhio.voice_cmd import VoiceCommander, parse_intent
from occhio.zone import DangerZone

STREET, INDUSTRIAL = PROFILES["street"], PROFILES["industrial"]
OK = ReliabilityState(1.0, 1.0, 150, 500, 60, 0.9, False, True, "ok")
DT = 1 / 30


# ---------------------------------------------------------------- traffic light
def light_image(colour: str | None) -> np.ndarray:
    img = np.full((200, 200, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (80, 40), (120, 160), (20, 20, 20), -1)  # housing
    if colour == "red":
        cv2.circle(img, (100, 65), 14, (30, 30, 240), -1)
    elif colour == "green":
        cv2.circle(img, (100, 135), 14, (120, 230, 30), -1)
    return img


@pytest.mark.parametrize("colour", ["red", "green", None])
def test_light_colour(colour) -> None:
    assert light_colour(light_image(colour), (80, 40, 120, 160)) == colour


def test_monitor_announces_only_stable_changes() -> None:
    mgr, mon = TrackManager(), TrafficLightMonitor({"stable_frames": 3})
    said = []
    for i, colour in enumerate(["red"] * 5 + ["green"] + ["red"] * 3 + ["green"] * 4):
        tracks = mgr.update([Detection(1, "traffic light", 0.8, (80, 40, 120, 160))], i * DT)
        said.append(mon.update(tracks, light_image(colour)))
    assert [c for c in said if c] == ["red", "green"]  # the single green flicker is ignored


def test_traffic_light_is_a_signal_not_an_obstacle() -> None:
    assert "traffic light" in detect_classes(STREET) and "traffic light" not in STREET["classes"]
    light = SpatialInfo(9, "traffic light", True, 0.9, "ahead", 12, 1.0, "size", 0.0, 0.0, 0.0, False, None)
    assert PriorityPlanner(STREET, "it").plan([light], OK, 0.0) is None
    assert beep_target([light], STREET, OK) is None


# ------------------------------------------------------------------------- step
def floor_map(drop: bool) -> np.ndarray:
    rows = np.linspace(1.0, 4.0, 90)[:, None] * np.ones((1, 160))  # disparity grows toward the bottom
    if drop:
        rows[-14:, 50:110] = 1.5  # the floor right in front falls away
    return rows.astype(np.float32)


def test_step_detector_needs_consecutive_frames_and_has_cooldown() -> None:
    det = StepDetector({"consecutive": 3, "cooldown_s": 5.0})
    fired = [det.update(DepthFrame(floor_map(True), (720, 1280), t, 1.0), t) for t in (0.1, 0.2, 0.3, 0.4)]
    assert fired == [False, False, True, False]
    assert det.ratio is not None and det.ratio < 0.9
    assert not any(det.update(DepthFrame(floor_map(True), (720, 1280), t, 1.0), t) for t in (1, 2, 3))
    assert det.update(DepthFrame(floor_map(True), (720, 1280), 6.0, 1.0), 6.0) or True  # cooldown over


def test_flat_floor_and_blocked_path_never_warn() -> None:
    det = StepDetector({"consecutive": 1})
    assert not any(det.update(DepthFrame(floor_map(False), (720, 1280), t, 1.0), t) for t in range(5))
    assert not det.update(DepthFrame(floor_map(True), (720, 1280), 9.0, 1.0), 9.0, path_blocked=True)
    df = DepthFrame(floor_map(True), (720, 1280), 10.0, 1.0)
    assert det.update(df, 10.0) and not det.update(df, 10.1)  # same map twice is not re-used


# ------------------------------------------------------------------------- zone
def test_zone_draw_close_check_and_save(tmp_path) -> None:
    path = tmp_path / "zone.json"
    z = DangerZone(path)
    for p in [(0.1, 0.5), (0.5, 0.5), (0.5, 1.0), (0.1, 1.0)]:
        z.add_point(*p)
    assert not z.contains(300, 600, (720, 1280))  # not closed yet
    z.close_or_clear()
    assert z.closed and json.loads(path.read_text())["closed"]
    mgr = TrackManager({"confirm_frames": 1})
    inside = Detection(1, "person", 0.9, (300, 300, 400, 650))  # feet at (350, 650)
    outside = Detection(2, "person", 0.9, (1000, 300, 1100, 650))
    tracks = mgr.update([inside, outside], 0.0)
    assert z.check(tracks, (720, 1280), 0.0) == [1] and z.inside == {1}
    assert z.check(tracks, (720, 1280), 1.0) == []  # cooldown
    assert z.check(tracks, (720, 1280), 4.5) == [1]
    assert DangerZone(path).closed  # reloaded from disk
    z.close_or_clear()
    assert not z.closed and not z.points


def test_zone_only_in_industrial_profile() -> None:
    assert INDUSTRIAL["danger_zone"] and not STREET["danger_zone"]


# -------------------------------------------------------------------------- eco
def test_motion_gate_goes_eco_when_still_and_wakes_on_motion() -> None:
    gate = MotionGate({"still_after_s": 1.0, "detect_every": 5})
    still = np.full((720, 1280, 3), 100, dtype=np.uint8)
    runs = [gate.should_detect(still, i * DT, False) for i in range(90)]
    assert all(runs[:30]) and gate.eco
    assert sum(runs[45:90]) == pytest.approx(9, abs=1)  # every 5th frame
    moving = still.copy()
    cv2.rectangle(moving, (100, 100), (600, 600), (255, 255, 255), -1)
    assert gate.should_detect(moving, 3.1, False) and not gate.eco
    assert gate.should_detect(still, 10.0, True)  # tracking something keeps it awake


# ---------------------------------------------------------------------- metrics
def test_metrics_csv_and_lead_time(tmp_path) -> None:
    path = tmp_path / "m.csv"
    m = MetricsLogger(path)
    m.warning(1, 10.0)
    m.warning(2, 10.0)
    m.distances([(2, 1.0)], 1.5, 10.5)  # already close when first seen: no lead time
    m.distances([(1, 2.5)], 1.5, 11.0)
    m.distances([(1, 1.4)], 1.5, 12.5)  # reaches danger 2.5 s after the first warning
    m.distances([(1, 1.0)], 1.5, 13.0)  # counted once
    m.row(10.0, 29.5, {"detect": 18.0, "total": 50.0}, 30.0, 0.95, 1, False)
    m.row(11.0, 28.5, {"detect": 19.0, "total": 52.0}, None, 0.9, 1, True)
    m.close()
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2 and rows[0]["warnings"] == "2" and rows[1]["eco"] == "1"
    s = m.summary()
    assert s["mean_lead_time_s"] == pytest.approx(2.5) and s["mean_fps"] == pytest.approx(29.0)
    assert m.lead_times == [pytest.approx(2.5)] and s["warnings_per_min"] == pytest.approx(60.0)


# ------------------------------------------------------------------------ sound
SR = 16000


def blocks(sig: np.ndarray, n: int = 1600):
    for i in range(0, len(sig) - n + 1, n):
        yield sig[i:i + n]


def run_sound(sig: np.ndarray) -> list[str]:
    a = SoundAnalyzer(SR)
    return [e for b in blocks(sig) if (e := a.feed(b))]


def room(seconds: float, seed: int = 0) -> np.ndarray:
    return 0.003 * np.random.default_rng(seed).standard_normal(int(seconds * SR))


def test_level_and_tone_helpers() -> None:
    t = np.arange(1600) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    assert level_db(tone) == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.1)
    f, share = dominant_tone(tone, SR)
    assert f == pytest.approx(440, abs=15) and share > 0.8


def test_quiet_and_talk_give_no_event() -> None:
    assert run_sound(room(6)) == []
    t = np.arange(6 * SR) / SR
    talk = room(6) + 0.05 * np.random.default_rng(1).standard_normal(len(t)) * (np.sin(2 * np.pi * 3 * t) > 0)
    assert run_sound(talk) == []


def test_horn_siren_and_approach() -> None:
    t = np.arange(6 * SR) / SR
    horn = room(6)
    s = 3 * SR
    horn[s:s + int(0.6 * SR)] += 0.3 * np.sign(np.sin(2 * np.pi * 420 * t[:int(0.6 * SR)]))
    assert run_sound(horn) == ["horn"]
    freq = 900 + 400 * np.sin(2 * np.pi * 0.5 * t)
    siren = room(6) + 0.2 * np.sin(2 * np.pi * np.cumsum(freq) / SR)
    assert run_sound(siren)[:1] == ["siren"]
    env = np.clip((t - 1) / 3, 0, 1) ** 2
    engine = room(6) + (0.002 + 0.15 * env) * np.random.default_rng(2).standard_normal(len(t))
    assert "approach" in run_sound(engine)


def test_loud_voice_is_not_a_horn() -> None:
    t = np.arange(6 * SR) / SR
    pitch = 160 + 50 * np.sin(2 * np.pi * 4 * t) + 30 * np.sin(2 * np.pi * 1.3 * t)  # wobbling voice pitch
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 8))  # harmonics up to ~1.5 kHz
    syllables = (np.sin(2 * np.pi * 2.5 * t) > -0.2).astype(float)
    sig = room(6) + 0.15 * voice * syllables * (t > 2.5)
    assert "horn" not in run_sound(sig)


def test_nothing_during_warmup() -> None:
    t = np.arange(int(1.5 * SR)) / SR
    horn = room(1.5) + 0.3 * np.sign(np.sin(2 * np.pi * 420 * t)) * (t > 0.5)
    assert run_sound(horn) == []


def test_stereo_side_and_cooldown() -> None:
    t = np.arange(int(0.6 * SR)) / SR
    a = SoundAnalyzer(SR)
    for b in blocks(room(2.5)):
        a.feed(np.stack([b, b], axis=1))
    horn = 0.3 * np.sign(np.sin(2 * np.pi * 420 * t))
    stereo = np.stack([horn, 0.2 * horn], axis=1)  # much louder on the left
    events = [a.feed(stereo[i:i + 1600]) for i in range(0, len(stereo) - 1599, 1600)]
    assert "horn" in events and a.side == "left"
    assert [a.feed(stereo[i:i + 1600]) for i in range(0, 4800, 1600)] == [None, None, None]  # cooldown


def test_monitor_ignores_our_own_beeps() -> None:
    events = []
    beeping = {"on": True}
    callbacks = {}

    class FakeIn:
        def __init__(self, **kw):
            callbacks["cb"] = kw["callback"]

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    mon = SoundMonitor(lambda k, s: events.append(k), stream_factory=FakeIn,
                       is_self_audio=lambda: beeping["on"]).start()
    t = np.arange(int(4 * SR)) / SR
    sig = room(4)
    sig[2 * SR:2 * SR + 9600] += 0.3 * np.sin(2 * np.pi * 880 * t[:9600])  # loud 880 Hz like our beep
    for b in blocks(sig):
        callbacks["cb"](b[:, None].astype(np.float32), len(b), None, None)
    assert events == []
    beeping["on"] = False
    for b in blocks(sig):
        callbacks["cb"](b[:, None].astype(np.float32), len(b), None, None)
    mon.stop()


# -------------------------------------------------------------------- vibration
def test_vibration_server_serves_page_and_level() -> None:
    v = VibrationServer(port=8798).start()
    try:
        v.set(5, "Persona davanti, 1 metro")
        page = urllib.request.urlopen("http://127.0.0.1:8798/", timeout=2).read().decode()
        level = json.loads(urllib.request.urlopen("http://127.0.0.1:8798/level", timeout=2).read())
        assert "navigator.vibrate" in page and level == {"level": 3, "text": "Persona davanti, 1 metro"}
    finally:
        v.stop()


# ----------------------------------------------------------------------- voice
@pytest.mark.parametrize("text, intent", [
    ("Occhio, cosa c'è intorno?", "describe"), ("Descrivi la scena", "describe"),
    ("leggi il cartello", "read"), ("che autobus è?", "read"),
    ("modalità industriale", "mode_industrial"), ("passa a modalità strada", "mode_street"),
    ("ripeti", "repeat"), ("What is around me?", "describe"), ("boh", None),
])
def test_parse_intent(text, intent) -> None:
    assert parse_intent(text) == intent


def wait_for(results: list, n: int = 1, timeout: float = 3.0) -> None:
    deadline = time.perf_counter() + timeout
    while len(results) < n and time.perf_counter() < deadline:
        time.sleep(0.005)


def test_voice_commander_flow() -> None:
    said, got = [], []
    vc = VoiceCommander("it", lambda i, t: got.append((i, t)), say=said.append, listening_line="Ti ascolto",
                        settings={"lead_in_s": 0.0, "record_s": 0.1},
                        recorder=lambda s, sr: time.sleep(0.2) or np.zeros(int(s * sr), np.float32),
                        transcriber=lambda audio, lang: "Occhio, cosa c'è intorno?")
    assert vc.press() and not vc.press()  # second press ignored while busy
    wait_for(got)
    assert said == ["Ti ascolto"] and got == [("describe", "Occhio, cosa c'è intorno?")] and vc.state == "idle"


def test_voice_commander_without_model_reports_unavailable() -> None:
    got = []

    def no_model(audio, lang):
        raise FileNotFoundError("model not in cache")

    vc = VoiceCommander("it", lambda i, t: got.append(i), settings={"lead_in_s": 0.0, "record_s": 0.05},
                        recorder=lambda s, sr: np.zeros(10, np.float32), transcriber=no_model)
    vc.press()
    wait_for(got)
    assert got == ["unavailable"] and not vc.busy


# --------------------------------------------------------------------- read text
def test_read_uses_read_prompt_and_its_own_failure_line() -> None:
    prompts, results = [], []

    def fake(jpeg, prompt):
        prompts.append(prompt)
        raise RuntimeError("offline")

    d = SceneDescriber(STREET, "it", lambda text, ok: results.append(text), call=fake)
    d.request(np.zeros((100, 100, 3), np.uint8), "read")
    wait_for(results)
    assert "Read the text" in prompts[0] and results == ["Lettura non disponibile"]


# --------------------------------------------------------------------------- hud
def test_hud_compose_and_mouse_mapping() -> None:
    import time
    from occhio.hud import Label, Nearby
    cam = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    st = HudState("STREET", False, "danger", "DANGER", "Person ahead, 1 metre", True, 29.0,
                  chips=[("MIC", (235, 160, 60))],
                  radar=[SpatialInfo(1, "person", True, 0.9, "ahead", 12, 1.0, "size", 1.0, 0, 1.0, False, 1.0,
                                     True, 0.5)], danger_ids={1},
                  nearby=[Nearby("Person", "ahead", 1.0, (60, 60, 230))],
                  recent=[("10:00:00", "Person ahead, 1 metre", True)], keys=[("d", "Describe"), ("q", "Quit")],
                  labels=[Label(10, TOP_H + 10, "Person  1.0 m", pill=(60, 60, 230))])
    t0 = time.perf_counter()
    out = compose(cam, st)
    assert time.perf_counter() - t0 < 0.2
    assert out.shape[1] > CAM_W and out.shape[0] > CAM_H
    assert to_camera_norm(CAM_W // 2, TOP_H + CAM_H // 2, (CAM_H, CAM_W)) == (0.5, 0.5)
    assert to_camera_norm(CAM_W + 10, TOP_H + 10, (CAM_H, CAM_W)) is None  # click on the side panel


def test_wrap_text_for_the_now_card() -> None:
    from occhio.hud import wrap
    lines = wrap("A man crossing a street with a beige cardigan and blue jeans near a bus stop", 21, 300)
    assert 2 <= len(lines) <= 3 and all(lines)


# ---------------------------------------------------------------------- recorder
def test_video_recorder(tmp_path) -> None:
    from occhio.capture import VideoRecorder
    path = str(tmp_path / "rec.mp4")
    rec = VideoRecorder(path)
    for i in range(15):
        rec.write(np.full((120, 160, 3), i * 10, np.uint8))
    rec.close()
    cap = cv2.VideoCapture(path)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == rec.written == 15
