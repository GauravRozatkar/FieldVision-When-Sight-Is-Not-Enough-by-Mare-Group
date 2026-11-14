"""Danger ranking and anti-spam cooldowns: decides the one sentence to speak, if any.

Rules (thresholds in config/settings.py PRIORITY and in the profile):
- Unreliable guidance: say the profile's "unreliable" line (urgent, then every
  unreliable_repeat_s) and nothing else.
- Only confirmed tracks with a distance within warn_distance_m are candidates.
- danger_score = closeness * max(approach_speed, 0.1) * mean_conf; the highest
  score that is allowed to speak wins. At most one message per cycle.
- Message type: "danger" (inside the walking corridor and distance <
  danger_distance_m or ttc < danger_ttc_s), "ttc" ("Collisione tra 2 secondi",
  in the corridor with ttc < ttc_warn_s), "approach" (coming closer from the
  left/right), "ahead" or "side". Outside the corridor nothing is "danger".
- "clear" ("Path clear") once, after clear_after_s without anything within
  warn distance in the corridor, if something was announced before.

Calm in busy scenes (a crowd must not turn into a stream of fragments):
- at most one sentence every min_gap_s; only a new danger, or a danger that came
  at least closer_m closer, may follow sooner (after urgent_gap_s);
- an unchanged danger is repeated only every danger_still_repeat_s (the beeps
  keep saying "still close"); no new warning for something walking away;
- "Collision in N seconds" only before the object is inside the danger distance;
- with crowd_count or more objects within warn distance, one summary sentence
  ("Two people ahead, 1 metre. One person on the right, 2 metres") replaces the
  per-person warnings, every crowd_repeat_s; dangers still come first.
- Cooldown: the same (track, type) at most once every cooldown_s, and the same
  sentence for the same track not again for repeat_same_text_s. Danger skips the
  cooldown (urgent, interrupts speech) when it is new or when its sentence
  changed (closer, fewer seconds), at most every danger_repeat_s; an unchanged
  danger (someone standing still in front) is repeated only every
  danger_still_repeat_s, because the beeps keep saying "still close".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config.profiles import CATEGORY, Profile, render
from config.settings import BEEPS, PRIORITY, TALK
from occhio.reliability import ReliabilityState
from occhio.scene_local import describe_detections
from occhio.spatial import SpatialInfo
from occhio.speech import beep_rate


@dataclass(frozen=True)
class Message:
    text: str
    kind: str  # "danger" | "ttc" | "approach" | "ahead" | "side" | "crowd" | "clear" | "unreliable"
    track_id: int | None
    urgent: bool
    score: float = 0.0


class PriorityPlanner:
    """Chooses at most one Message per frame."""

    def __init__(self, profile: Profile, lang: str, settings: dict | None = None) -> None:
        cfg = {**PRIORITY, **(settings or {})}
        self.profile = profile
        self.lang = lang
        self.cooldown_s = float(cfg["cooldown_s"])
        self.repeat_same_text_s = float(cfg["repeat_same_text_s"])
        self.danger_repeat_s = float(cfg["danger_repeat_s"])
        self.danger_still_repeat_s = float(cfg["danger_still_repeat_s"])
        self.danger_ttc_s = float(cfg["danger_ttc_s"])
        self.unreliable_repeat_s = float(cfg["unreliable_repeat_s"])
        self.min_closeness_m = float(cfg["min_closeness_m"])
        self.ttc_warn_s = float(cfg.get("ttc_warn_s", TALK["ttc_warn_s"]))
        self.clear_after_s = float(cfg.get("clear_after_s", TALK["clear_after_s"]))
        self._last_relevant: float | None = None  # last time something was within warn distance
        self._announced = False  # something was said since the last "clear"
        self._last_kind: dict[tuple[int, str], float] = {}
        self._last_text: dict[int, tuple[str, float]] = {}
        self._last_unreliable: float | None = None
        self._was_reliable = True
        self.min_gap_s = float(cfg["min_gap_s"])
        self.urgent_gap_s = float(cfg["urgent_gap_s"])
        self.closer_m = float(cfg["closer_m"])
        self.away_mps = float(cfg["away_mps"])
        self.crowd_count = int(cfg["crowd_count"])
        self.crowd_repeat_s = float(cfg["crowd_repeat_s"])
        self._last_any: float | None = None  # last time anything was said
        self._danger_dist: dict[int, float] = {}  # distance at the last danger message per track
        self._last_crowd: tuple[str, float] | None = None

    def set_profile(self, profile: Profile) -> None:
        self.profile = profile
        self.reset()

    def reset(self) -> None:
        self._last_kind.clear()
        self._last_text.clear()
        self._last_unreliable = None
        self._was_reliable = True
        self._last_relevant = None
        self._announced = False
        self._last_any = None
        self._danger_dist.clear()
        self._last_crowd = None

    # ---------------------------------------------------------------- scoring
    def danger_score(self, info: SpatialInfo) -> float:
        assert info.distance_m is not None
        closeness = float(self.profile["warn_distance_m"]) / max(info.distance_m, self.min_closeness_m)
        return closeness * max(info.approach_mps, 0.1) * info.mean_conf

    def is_danger(self, info: SpatialInfo) -> bool:
        assert info.distance_m is not None
        if not info.in_corridor:
            return False
        if info.distance_m < float(self.profile["danger_distance_m"]):
            return True
        return info.ttc_s is not None and info.ttc_s < self.danger_ttc_s

    def message_for(self, info: SpatialInfo) -> Message:
        danger = self.is_danger(info)
        approaching = info.approach_mps >= float(self.profile["approach_min_mps"])
        outside_danger = info.distance_m is not None and info.distance_m >= float(self.profile["danger_distance_m"])
        if info.in_corridor and info.ttc_s is not None and info.ttc_s < self.ttc_warn_s and outside_danger:
            key, kind = "ttc", "danger" if danger else "ttc"
        elif not danger and approaching and info.side != "ahead":
            key, kind = "approach", "approach"
        else:
            key = "ahead" if info.side == "ahead" else "side"
            kind = "danger" if danger else key
        text = render(self.profile, self.lang, key, info.cls_name, info.distance_m, info.side, info.ttc_s)
        return Message(text, kind, info.track_id, urgent=danger, score=self.danger_score(info))

    # ----------------------------------------------------------------- choose
    def plan(self, infos: Iterable[SpatialInfo], reliability: ReliabilityState | None,
             now: float) -> Message | None:
        if reliability is not None and not reliability.reliable:
            first = self._was_reliable
            self._was_reliable = False
            if first or self._last_unreliable is None or now - self._last_unreliable >= self.unreliable_repeat_s:
                self._last_unreliable = now
                return Message(render(self.profile, self.lang, "unreliable"), "unreliable", None, urgent=True)
            return None
        self._was_reliable = True

        warn = float(self.profile["warn_distance_m"])
        obstacles = set(self.profile["classes"])
        candidates = [i for i in infos if i.confirmed and i.distance_m is not None and i.distance_m <= warn
                      and i.cls_name in obstacles]
        if any(i.in_corridor for i in candidates):
            self._last_relevant = now
        by_id = {i.track_id: i for i in candidates}
        messages = sorted((self.message_for(i) for i in candidates), key=lambda m: m.score, reverse=True)
        crowd = len(candidates) >= self.crowd_count
        for msg in messages:
            if crowd and msg.kind != "danger":
                continue  # in a crowd only dangers are spoken one by one
            if self._allowed(msg, by_id[msg.track_id], now):
                self._remember(msg, by_id[msg.track_id], now)
                return msg
        if crowd and self._gap_ok(now, self.min_gap_s):
            text = describe_detections(candidates, self.profile, self.lang, max_items=2)
            last = self._last_crowd
            if last is None or (now - last[1] >= self.crowd_repeat_s and (text != last[0] or now - last[1] >= 2 * self.crowd_repeat_s)):
                self._last_crowd = (text, now)
                self._last_any = now
                self._announced = True
                return Message(text, "crowd", None, urgent=False)
        if (self._announced and self._last_relevant is not None
                and now - self._last_relevant >= self.clear_after_s and self._gap_ok(now, self.min_gap_s)):
            self._announced = False
            self._last_any = now
            return Message(render(self.profile, self.lang, "clear"), "clear", None, urgent=False)
        return None

    def _gap_ok(self, now: float, gap: float) -> bool:
        return self._last_any is None or now - self._last_any >= gap

    def _allowed(self, msg: Message, info: SpatialInfo, now: float) -> bool:
        assert msg.track_id is not None and info.distance_m is not None
        if msg.kind == "danger":
            last = self._last_kind.get((msg.track_id, "danger"))
            if last is None:  # a new danger: as soon as possible
                return self._gap_ok(now, self.urgent_gap_s)
            closer = self._danger_dist.get(msg.track_id, info.distance_m) - info.distance_m >= self.closer_m
            if closer and now - last >= self.danger_repeat_s:
                return self._gap_ok(now, self.urgent_gap_s)
            return now - last >= self.danger_still_repeat_s and self._gap_ok(now, self.min_gap_s)
        if not self._gap_ok(now, self.min_gap_s):
            return False
        last = self._last_kind.get((msg.track_id, msg.kind))
        if last is not None and now - last < self.cooldown_s:
            return False
        if info.approach_mps <= self.away_mps and self._last_text.get(msg.track_id) is not None:
            return False  # walking away: already announced, nothing new to say
        same = self._last_text.get(msg.track_id)
        return not (same is not None and same[0] == msg.text and now - same[1] < self.repeat_same_text_s)

    def _remember(self, msg: Message, info: SpatialInfo, now: float) -> None:
        assert msg.track_id is not None
        self._last_kind[(msg.track_id, msg.kind)] = now
        self._last_text[msg.track_id] = (msg.text, now)
        self._last_any = now
        self._announced = True
        if msg.kind == "danger" and info.distance_m is not None:
            self._danger_dist[msg.track_id] = info.distance_m


class BeepPolicy:
    """Decides when to beep, with memory: an object that stays put goes quiet.

    Beeps while the nearest relevant object is coming closer, or for quiet_after_s
    after it first becomes relevant; then silent until it comes rearm_closer_m
    closer. Beeping non-stop at someone who is just standing there is irritating
    and hides the sounds of the street.
    """

    def __init__(self, profile: Profile, settings: dict | None = None) -> None:
        cfg = {**BEEPS, **(settings or {})}
        self.profile = profile
        self.quiet_after_s = float(cfg["quiet_after_s"])
        self.rearm_closer_m = float(cfg["rearm_closer_m"])
        self._state: dict[int, tuple[float, float]] = {}  # track -> (since, distance then)

    def set_profile(self, profile: Profile) -> None:
        self.profile = profile
        self._state.clear()

    def update(self, infos: Iterable[SpatialInfo], reliability: ReliabilityState | None,
               now: float) -> tuple[float, float, float] | None:
        infos = list(infos)
        approach_min = float(self.profile["approach_min_mps"])
        alive: set[int] = set()
        active: list[SpatialInfo] = []
        for info in _beep_candidates(infos, self.profile):
            alive.add(info.track_id)
            assert info.distance_m is not None
            approaching = info.approach_mps >= approach_min
            since, ref = self._state.get(info.track_id, (now, info.distance_m))
            if approaching or ref - info.distance_m >= self.rearm_closer_m:
                since, ref = now, info.distance_m
            self._state[info.track_id] = (since, min(ref, info.distance_m) if not approaching else ref)
            if approaching or now - since < self.quiet_after_s:
                active.append(info)
        for track_id in [t for t in self._state if t not in alive]:
            del self._state[track_id]
        return beep_target(active, self.profile, reliability)


def _beep_candidates(infos: Iterable[SpatialInfo], profile: Profile) -> list[SpatialInfo]:
    warn = float(profile["warn_distance_m"])
    approach_min = float(profile["approach_min_mps"])
    obstacles = set(profile["classes"])
    return [i for i in infos if i.cls_name in obstacles and i.confirmed and i.distance_m is not None
            and i.distance_m <= warn and (i.in_corridor or i.approach_mps >= approach_min)]


def beep_target(infos: Iterable[SpatialInfo], profile: Profile,
                reliability: ReliabilityState | None = None) -> tuple[float, float, float] | None:
    """(beeps per second, pan -1..1, pitch Hz) for the nearest confirmed object worth beeping about.

    The pitch says what it is: high for vehicles, middle for people and animals, low for
    obstacles (bags, chairs...), so the user can tell a car from a bench without words.

    Worth it = within warn distance and either in the walking corridor or coming
    closer. None (silence) when nothing qualifies or the guidance is unreliable.
    """
    if reliability is not None and not reliability.reliable:
        return None
    warn = float(profile["warn_distance_m"])
    danger = float(profile["danger_distance_m"])
    candidates = _beep_candidates(infos, profile)
    if not candidates:
        return None
    nearest = min(candidates, key=lambda i: i.distance_m)  # type: ignore[arg-type,return-value]
    rate = beep_rate(nearest.distance_m, warn, danger)
    freq = float(BEEPS["category_freq_hz"][CATEGORY.get(nearest.cls_name, "obstacle")])
    return None if rate is None else (rate, nearest.x_norm * 2.0 - 1.0, freq)
