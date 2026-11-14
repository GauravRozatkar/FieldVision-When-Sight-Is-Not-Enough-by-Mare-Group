"""Tests for describing the scene without an API key: instant summary (A) and local SmolVLM (B)."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.profiles import PROFILES
from occhio.scene_llm import SceneDescriber, encode_jpeg
from occhio.scene_local import describe_detections
from occhio.spatial import SpatialInfo
from occhio.vlm_local import LOCAL_PROMPTS, LocalVLM, choose_backend, is_cached

STREET, INDUSTRIAL = PROFILES["street"], PROFILES["industrial"]
ROOT = Path(__file__).resolve().parent.parent


def info(tid: int, cls: str, side: str, dist: float | None, confirmed: bool = True) -> SpatialInfo:
    return SpatialInfo(tid, cls, confirmed, 0.9, side, 12, dist, "size", 0.0, 0.0, 0.0, False, None)


# ------------------------------------------------------------ A: instant summary
def test_summary_groups_sorts_and_counts() -> None:
    infos = [info(1, "person", "ahead", 2.1), info(2, "person", "ahead", 3.4), info(3, "bicycle", "left", 4.2),
             info(4, "dog", "right", 1.0)]
    assert describe_detections(infos, STREET, "en") == \
        "One dog on the right, 1 metre. Two people ahead, 2 metres. One bike on the left, 4 metres"


def test_summary_skips_uncertain_signals_and_unknown_distance() -> None:
    infos = [info(1, "person", "ahead", 1.0, confirmed=False), info(2, "traffic light", "ahead", 5.0),
             info(3, "car", "left", None)]
    assert describe_detections(infos, STREET, "en") == "Nothing detected around you"


def test_summary_mentions_light_limits_items_and_speaks_italian() -> None:
    infos = [info(i, cls, "ahead", float(i)) for i, cls in enumerate(["person", "car", "bus", "truck"], 1)]
    text = describe_detections(infos, STREET, "en", light="green")
    assert text.count(". ") == 3 and text.endswith("Green light") and "truck" not in text
    assert describe_detections([info(1, "person", "left", 2.0)], STREET, "it") == "1 persona a sinistra, 2 metri"


def test_summary_follows_mode_classes() -> None:
    assert describe_detections([info(1, "dog", "ahead", 2.0)], INDUSTRIAL, "en") == "Nothing detected around you"


# ------------------------------------------------------------- backend choice
@pytest.mark.parametrize("pref, key, cached, want", [
    ("auto", True, True, "claude"), ("auto", False, True, "local"), ("auto", False, False, "none"),
    ("claude", False, False, "claude"), ("local", True, True, "local"), ("local", True, False, "none"),
    ("none", True, True, "none"),
])
def test_choose_backend(pref, key, cached, want) -> None:
    assert choose_backend(pref, key, cached) == want


def test_local_prompts_exist_for_both_languages() -> None:
    for kind in ("describe", "read"):
        assert set(LOCAL_PROMPTS[kind]) == {"en", "it"}


# ---------------------------------------------------------------- B: local model
def test_missing_local_model_fails_cleanly() -> None:
    vlm = LocalVLM({"model": "nobody/does-not-exist-vlm", "load_timeout_s": 30}).start_loading()
    results: list = []
    d = SceneDescriber(STREET, "en", lambda t, ok: results.append((t, ok)), call=vlm,
                       prompt_for=lambda k, lang: LOCAL_PROMPTS[k][lang])
    d.request(np.zeros((90, 160, 3), np.uint8), "read")
    deadline = time.perf_counter() + 30
    while not results and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert results == [("Reading not available", False)] and vlm.error


@pytest.mark.skipif(not is_cached(), reason="SmolVLM not downloaded")
def test_local_model_describes_bus_photo() -> None:
    vlm = LocalVLM().start_loading()
    jpeg = encode_jpeg(cv2.imread(str(ROOT / "bus.jpg")))
    t0 = time.perf_counter()
    answer = vlm(jpeg, LOCAL_PROMPTS["describe"]["en"])
    assert vlm.ready and answer and "bus" in answer.lower()
    t1 = time.perf_counter()
    vlm(jpeg, LOCAL_PROMPTS["read"]["en"])
    assert time.perf_counter() - t1 < 15.0  # about 1 s on an idle M5 Air; generous when the GPU is shared
    assert t1 - t0 < 90


@pytest.mark.parametrize("answer, none", [("No text.", True), ("no text", True), ("There is no text in the image", True),
                                          ("Nessun testo", True), ("", True), ("Bus 140", False), ("STOP", False)])
def test_is_no_text(answer, none) -> None:
    from occhio.app import is_no_text
    assert is_no_text(answer) is none


@pytest.mark.skipif(not is_cached(), reason="SmolVLM not downloaded")
def test_local_reader_does_not_invent_text_on_a_traffic_light() -> None:
    from occhio.app import is_no_text
    img = np.full((720, 1280, 3), (150, 140, 130), np.uint8)
    cv2.rectangle(img, (595, 60), (675, 260), (25, 25, 25), -1)
    cv2.circle(img, (635, 95), 25, (30, 30, 230), -1)
    vlm = LocalVLM().start_loading()
    assert is_no_text(vlm(encode_jpeg(img), LOCAL_PROMPTS["read"]["en"]))


@pytest.mark.skipif(not is_cached(), reason="SmolVLM not downloaded")
def test_local_reader_reads_a_clear_sign() -> None:
    img = np.full((720, 1280, 3), (120, 130, 140), np.uint8)
    cv2.rectangle(img, (340, 220), (940, 480), (240, 240, 240), -1)
    cv2.putText(img, "EXIT", (380, 390), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (20, 20, 20), 8, cv2.LINE_AA)
    vlm = LocalVLM().start_loading()
    assert "EXIT" in vlm(encode_jpeg(img), LOCAL_PROMPTS["read"]["en"]).upper()
