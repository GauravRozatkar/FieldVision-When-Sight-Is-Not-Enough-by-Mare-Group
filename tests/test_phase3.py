"""Tests for Phase 3: corridor, TTC / clear messages, beeps, describe-scene, mode switch helpers."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from config.profiles import PROFILES, next_mode, render
from main import parse_key_script
from occhio.priority import PriorityPlanner, beep_target
from occhio.reliability import ReliabilityState
from occhio.scene_llm import DescribeError, SceneDescriber, anthropic_call, describe_error, encode_jpeg
from occhio.spatial import SpatialInfo, corridor_polygon, in_corridor
from occhio.speech import Beeper, beep_rate, pan_gains

STREET = PROFILES["street"]
FRAME = (720, 1280)
OK = ReliabilityState(1.0, 1.0, 150, 500, 60, 0.9, False, True, "ok")
BAD = ReliabilityState(0.1, 0.0, 3, 1, 1, None, True, False, "blocked")


def info(track_id: int = 1, dist: float | None = 2.0, approach: float = 0.0, side: str = "ahead",
         in_corr: bool = True, x: float = 0.5, cls: str = "person", confirmed: bool = True) -> SpatialInfo:
    ttc = dist / approach if (dist is not None and approach >= 0.3) else None
    return SpatialInfo(track_id, cls, confirmed, 0.9, side, 12, dist, "size", approach, 0.0,
                       abs(approach), False, ttc, in_corr, x)


# ------------------------------------------------------------------- corridor
def test_corridor_shape_and_membership() -> None:
    c = STREET["corridor"]  # top_y 0.45, top_w 0.30, bottom_w 0.90
    poly = corridor_polygon(c, FRAME)
    assert poly[0] == (64, 719) and poly[1] == (1216, 719)  # bottom 90% wide
    assert poly[3][1] == round(0.45 * 720)
    assert in_corridor(640, 720, c, FRAME)  # feet straight ahead, close
    assert in_corridor(100, 720, c, FRAME)  # wide at the bottom
    assert not in_corridor(100, 400, c, FRAME)  # narrow further up
    assert in_corridor(640, 400, c, FRAME)
    assert not in_corridor(640, 200, c, FRAME)  # above the corridor (far / sky)
    assert not in_corridor(1270, 719, c, FRAME)


def test_outside_corridor_is_never_danger() -> None:
    p = PriorityPlanner(STREET, "it")
    msg = p.plan([info(dist=1.0, side="left", in_corr=False)], OK, 0.0)
    assert msg is not None and not msg.urgent and msg.kind == "side" and msg.text == "Persona a sinistra, 1 metro"


# ----------------------------------------------------------------- ttc / clear
def test_ttc_message_below_3s() -> None:
    p = PriorityPlanner(STREET, "it")
    msg = p.plan([info(dist=2.8, approach=1.1)], OK, 0.0)  # ttc 2.5 s
    assert msg is not None and msg.text == "Collisione tra 3 secondi" and msg.kind == "ttc" and not msg.urgent
    p2 = PriorityPlanner(STREET, "it")
    urgent = p2.plan([info(dist=1.6, approach=1.6)], OK, 0.0)  # ttc 1 s, just outside the danger distance
    assert urgent is not None and urgent.urgent and urgent.text == "Collisione tra un secondo"
    inside = PriorityPlanner(STREET, "it").plan([info(dist=1.2, approach=1.6)], OK, 0.0)
    assert inside.urgent and inside.text == "Persona davanti, 1 metro"  # inside 1.5 m: say where, not when


def test_clear_once_after_5s_without_danger() -> None:
    p = PriorityPlanner(STREET, "it")
    assert p.plan([], OK, 0.0) is None  # nothing announced yet: no "clear" at start-up
    assert p.plan([info(dist=2.0)], OK, 1.0) is not None
    assert p.plan([], OK, 3.0) is None
    msg = p.plan([], OK, 6.1)
    assert msg is not None and msg.text == "Strada libera" and msg.kind == "clear"
    assert p.plan([], OK, 20.0) is None  # only once


def test_object_outside_corridor_does_not_block_clear() -> None:
    p = PriorityPlanner(STREET, "it")
    p.plan([info(dist=2.0)], OK, 0.0)
    side = info(2, dist=2.5, side="right", in_corr=False)
    assert p.plan([side], OK, 3.0) is not None  # announced once (after the 2.5 s gap)
    assert p.plan([side], OK, 8.5).kind == "clear"


def test_industrial_lines() -> None:
    ind = PROFILES["industrial"]
    assert render(ind, "it", "ttc", ttc_s=2.2) == "Stop. Collisione tra 2 secondi"
    assert render(ind, "it", "mode") == "Modalità industriale"
    assert render(STREET, "it", "mode") == "Modalità strada"


# ----------------------------------------------------------------------- beeps
def test_beep_rate_mapping() -> None:
    assert beep_rate(3.5, 3.0, 1.5) is None
    assert beep_rate(None, 3.0, 1.5) is None
    assert beep_rate(3.0, 3.0, 1.5) == pytest.approx(0.5)  # calm: half a beep per second at 3 m
    assert beep_rate(2.25, 3.0, 1.5) == pytest.approx(1.75)
    assert beep_rate(1.5, 3.0, 1.5) == pytest.approx(3.0)  # at most 3 per second
    assert beep_rate(0.5, 3.0, 1.5) == pytest.approx(3.0)


def test_pan_gains_equal_power() -> None:
    left, right = pan_gains(-1.0)
    assert left == pytest.approx(1.0) and right == pytest.approx(0.0, abs=1e-9)
    c = pan_gains(0.0)
    assert c[0] == pytest.approx(c[1]) and c[0] ** 2 + c[1] ** 2 == pytest.approx(1.0)
    assert pan_gains(5.0) == pan_gains(1.0)


def count_beeps(audio: np.ndarray) -> int:
    on = np.convolve(np.abs(audio).sum(axis=1), np.ones(50), "same") > 1e-3
    return int(on[0]) + int(np.sum(on[1:] & ~on[:-1]))


def render_seconds(b: Beeper, seconds: float, block: int = 512) -> np.ndarray:
    return np.vstack([b.render(block) for _ in range(int(seconds * b.sr / block))])


@pytest.mark.parametrize("rate", [1.0, 3.0, 6.0])
def test_beeper_rate(rate: float) -> None:
    b = Beeper(stream_factory=lambda **kw: None)
    b.set_target(rate, 0.0)
    assert count_beeps(render_seconds(b, 3.0)) == pytest.approx(rate * 3, abs=1)


def test_beeper_pan_and_silence() -> None:
    b = Beeper(stream_factory=lambda **kw: None)
    b.set_target(4.0, 1.0)  # far right
    audio = render_seconds(b, 0.5)
    assert np.abs(audio[:, 0]).max() < 1e-6 and 0.08 < np.abs(audio[:, 1]).max() <= 0.12 + 1e-6  # quiet
    b.set_target(None)
    b.render(4096)  # finish a running beep
    assert not np.any(b.render(4096))


def test_beeper_speeds_up_without_waiting_for_old_gap() -> None:
    b = Beeper(stream_factory=lambda **kw: None)
    b.set_target(1.0, 0.0)
    b.render(int(0.1 * b.sr))  # first beep, then a ~0.94 s gap starts
    b.set_target(6.0, 0.0)
    assert count_beeps(render_seconds(b, 0.25)) >= 1  # next beep within 1/6 s, not after 0.9 s


def test_beeper_without_audio_device_disables_itself() -> None:
    def broken(**kw):
        raise OSError("no device")

    b = Beeper(stream_factory=broken).start()
    assert not b.enabled
    b.set_target(3.0, 0.0)
    b.stop()


def test_beeper_uses_stream_factory() -> None:
    made = {}

    class FakeStream:
        def __init__(self, **kw):
            made.update(kw)
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            pass

        def close(self):
            pass

    b = Beeper(stream_factory=FakeStream).start()
    assert made["channels"] == 2 and made["samplerate"] == 44100 and callable(made["callback"])
    out = np.zeros((256, 2), dtype=np.float32)
    b.set_target(6.0, -1.0)
    made["callback"](out, 256, None, None)
    assert np.abs(out[:, 0]).max() > 0 and np.abs(out[:, 1]).max() < 1e-6
    b.stop()


def test_beep_target_rules() -> None:
    assert beep_target([], STREET, OK) is None
    far = info(1, dist=4.0)
    side_still = info(2, dist=2.0, in_corr=False, x=0.05)
    assert beep_target([far, side_still], STREET, OK) is None  # beyond warn / outside and not approaching
    side_coming = info(3, dist=2.5, approach=0.8, in_corr=False, x=0.0)
    rate, pan, freq = beep_target([side_coming], STREET, OK)
    assert pan == pytest.approx(-1.0) and rate == pytest.approx(beep_rate(2.5, 3.0, 1.5)) and freq == 660.0
    near = info(4, dist=1.2, x=0.75)
    assert beep_target([side_coming, near], STREET, OK) == (pytest.approx(3.0), pytest.approx(0.5), 660.0)
    assert beep_target([info(5, dist=1.0, cls="car")], STREET, OK)[2] == 990.0  # vehicles: high
    assert beep_target([info(6, dist=1.0, cls="handbag")], STREET, OK)[2] == 440.0  # obstacles: low
    assert beep_target([near], STREET, BAD) is None
    assert beep_target([info(5, dist=1.0, confirmed=False)], STREET, OK) is None


# ---------------------------------------------------------------- describe scene
def frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_encode_jpeg_limits_size() -> None:
    import cv2
    jpeg = encode_jpeg(frame())
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 768 and img.shape[1] / img.shape[0] == pytest.approx(16 / 9, rel=0.01)


def wait_for(results: list, n: int = 1, timeout: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout
    while len(results) < n and time.perf_counter() < deadline:
        time.sleep(0.005)


def test_describe_success_runs_in_background() -> None:
    gate = threading.Event()
    seen: dict = {}

    def fake_call(jpeg: bytes, prompt: str) -> str:
        seen["jpeg"], seen["prompt"] = jpeg, prompt
        gate.wait(2)
        return "Una persona davanti, una sedia a destra."

    results: list = []
    d = SceneDescriber(STREET, "it", lambda text, ok: results.append((text, ok)), call=fake_call)
    t0 = time.perf_counter()
    assert d.request(frame())
    assert time.perf_counter() - t0 < 0.3  # only the JPEG encoding; the model call never blocks the loop
    assert d.busy and not d.request(frame())  # second press ignored while busy
    gate.set()
    wait_for(results)
    assert results == [("Una persona davanti, una sedia a destra.", True)]
    assert seen["jpeg"][:2] == b"\xff\xd8" and "blind person" in seen["prompt"] and "Italian" in seen["prompt"]
    assert not d.busy


def test_describe_failure_speaks_fallback_line() -> None:
    def broken(jpeg, prompt):
        raise DescribeError("refusal")

    results: list = []
    d = SceneDescriber(STREET, "it", lambda text, ok: results.append((text, ok)), call=broken)
    d.request(frame())
    wait_for(results)
    assert results == [("Descrizione non disponibile", False)] and d.last_error == "refusal"
    en: list = []
    SceneDescriber(STREET, "en", lambda text, ok: en.append(text), call=broken).request(frame())
    wait_for(en)
    assert en == ["Description not available"]


def test_describe_uses_mode_prompt() -> None:
    prompts: list = []
    d = SceneDescriber(STREET, "it", lambda t, ok: None, call=lambda j, p: prompts.append(p) or "ok")
    d.set_profile(PROFILES["industrial"])
    d.request(frame())
    wait_for(prompts)
    assert "worker" in prompts[0]


def test_real_call_without_credentials_fails_cleanly(monkeypatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")  # never reach the network
    monkeypatch.setenv("HOME", "/nonexistent")  # no CLI profile on disk
    monkeypatch.setenv("OCCHIO_LLM_MODEL", "claude-opus-5-5")
    call = anthropic_call({"timeout_s": 2.0})
    assert call.model == "claude-opus-5-5"
    results: list = []
    d = SceneDescriber(STREET, "it", lambda text, ok: results.append((text, ok)), call=call)
    d.request(frame())
    wait_for(results, timeout=10)
    assert results == [("Descrizione non disponibile", False)]
    assert d.last_error  # a short reason for the console


def test_describe_error_text() -> None:
    assert describe_error(DescribeError("refusal")) == "refusal"
    assert describe_error(ValueError("x")) == "ValueError"


# ------------------------------------------------------------------ mode / keys
def test_next_mode_cycles() -> None:
    assert next_mode("street") == "industrial" and next_mode("industrial") == "street"


def test_parse_key_script() -> None:
    assert parse_key_script("5:m, 2:d") == [(2.0, "d"), (5.0, "m")]
    assert parse_key_script("") == []



# ------------------------------------------------------------------ calm beeps
def test_still_object_goes_quiet_after_3s_and_rearms_when_closer() -> None:
    from occhio.priority import BeepPolicy
    pol = BeepPolicy(STREET)
    sitter = info(1, dist=0.8)  # someone sitting in front of the laptop, not moving
    assert pol.update([sitter], OK, 0.0) is not None  # first: beep
    assert pol.update([sitter], OK, 2.9) is not None
    assert pol.update([sitter], OK, 3.1) is None  # still there, not moving: quiet
    assert pol.update([sitter], OK, 30.0) is None
    assert pol.update([info(1, dist=0.25)], OK, 31.0) is not None  # came 0.55 m closer: beep again


def test_approaching_object_keeps_beeping() -> None:
    from occhio.priority import BeepPolicy
    pol = BeepPolicy(STREET)
    for t in range(10):
        assert pol.update([info(1, dist=2.8 - 0.2 * t, approach=0.8)], OK, float(t)) is not None


def test_new_object_beeps_even_if_another_is_quiet() -> None:
    from occhio.priority import BeepPolicy
    pol = BeepPolicy(STREET)
    pol.update([info(1, dist=0.8)], OK, 0.0)
    assert pol.update([info(1, dist=0.8)], OK, 5.0) is None
    target = pol.update([info(1, dist=0.8), info(2, dist=2.0, x=0.9)], OK, 5.1)
    assert target is not None and target[1] == pytest.approx(0.8)  # the new one, on the right
