"""Busy scenes: speech must stay calm, complete and clear."""
from __future__ import annotations

import threading
import time

import pytest

from config.profiles import PROFILES
from occhio.priority import PriorityPlanner
from occhio.reliability import ReliabilityState
from occhio.spatial import SpatialInfo, side_of
from occhio.speech import Speaker

STREET = PROFILES["street"]
OK = ReliabilityState(1.0, 1.0, 150, 500, 60, 0.9, False, True, "ok")


def info(tid: int, dist: float, side: str = "ahead", approach: float = 0.0, in_corr: bool = True,
         cls: str = "person") -> SpatialInfo:
    ttc = dist / approach if approach >= 0.3 else None
    return SpatialInfo(tid, cls, True, 0.9, side, 12, dist, "size", approach, 0.0, abs(approach), False, ttc,
                       in_corr, 0.5)


def count_messages(p: PriorityPlanner, frames: list[list[SpatialInfo]], fps: float = 30.0) -> list:
    out = []
    for i, infos in enumerate(frames):
        msg = p.plan(infos, OK, i / fps)
        if msg:
            out.append((round(i / fps, 2), msg.text, msg.urgent))
    return out


def test_two_people_close_and_jittering_stay_calm() -> None:
    """The live log: someone leaning at 1 m, someone wandering on the right."""
    import math
    frames = []
    for i in range(30 * 60):
        t = i / 30
        a = info(1, 1.0 + 0.2 * math.sin(t * 2), side="ahead" if math.sin(t * 1.5) > 0 else "left")
        b = info(2, 2.1 + 0.9 * math.sin(2 * math.pi * t / 8), side="right", in_corr=False,
                 approach=-0.9 * 2 * math.pi / 8 * math.cos(2 * math.pi * t / 8))
        frames.append([a, b])
    said = count_messages(PriorityPlanner(STREET, "en"), frames)
    assert len(said) <= 25, said  # was ~75 per minute before
    gaps = [b[0] - a[0] for a, b in zip(said, said[1:])]
    assert min(gaps) >= 1.2 - 1e-6


def test_crowd_becomes_one_summary_sentence() -> None:
    people = [info(i, 2.0 + 0.2 * i, side=s, in_corr=(s == "ahead")) for i, s in
              enumerate(["ahead", "ahead", "right", "left"], 1)]
    p = PriorityPlanner(STREET, "en")
    msg = p.plan(people, OK, 0.0)
    assert msg.kind == "crowd" and msg.text.startswith("Two people ahead, 2 metres")
    assert p.plan(people, OK, 3.0) is None  # summaries every 8 s, not every gap
    assert p.plan(people, OK, 9.0) is None  # same picture: repeated only after 16 s
    assert p.plan(people, OK, 16.5).kind == "crowd"


def test_danger_still_breaks_through_a_crowd() -> None:
    people = [info(i, 2.5, side="right", in_corr=False) for i in range(1, 4)]
    p = PriorityPlanner(STREET, "en")
    assert p.plan(people, OK, 0.0).kind == "crowd"
    msg = p.plan(people + [info(9, 1.0)], OK, 1.5)
    assert msg.kind == "danger" and msg.urgent and msg.text == "Person ahead, 1 metre"


def test_no_collision_wording_inside_danger_zone() -> None:
    msg = PriorityPlanner(STREET, "en").plan([info(1, 0.5, approach=0.6)], OK, 0.0)
    assert msg.text == "Person ahead, 0.5 metres"


def test_side_hysteresis() -> None:
    w = 1200  # thirds at 400 / 800, margin 5% = 60 px
    assert side_of(390, w) == "left"
    assert side_of(390, w, previous="ahead", margin=0.05) == "ahead"  # leaning a little: no flip
    assert side_of(330, w, previous="ahead", margin=0.05) == "left"  # clearly past the line
    assert side_of(430, w, previous="left", margin=0.05) == "left"
    assert side_of(470, w, previous="left", margin=0.05) == "ahead"


class FakeProcess:
    def __init__(self, text, duration, log):
        self.text, self.killed = text, False
        self._done = threading.Event()
        log.append(self)
        threading.Timer(duration, self._done.set).start()

    def wait(self):
        self._done.wait()
        return 0

    def terminate(self):
        self.killed = True
        self._done.set()

    def poll(self):
        return 0 if self._done.is_set() else None


def test_urgent_never_cuts_urgent_but_cuts_normal() -> None:
    log: list = []
    sp = Speaker(runner=lambda text: FakeProcess(text, 0.4, log)).start()
    sp.say("normal")
    time.sleep(0.05)
    sp.say("danger one", urgent=True)  # cuts the normal sentence
    time.sleep(0.05)
    sp.say("danger two", urgent=True)  # waits for "danger one" to finish
    sp.say("info", urgent=False)  # dropped: an urgent sentence is waiting
    assert sp.wait_idle(3)
    assert [p.text for p in log] == ["normal", "danger one", "danger two"]
    assert log[0].killed and not log[1].killed
    sp.stop()
