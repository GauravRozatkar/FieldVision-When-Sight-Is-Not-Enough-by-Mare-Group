"""Instant scene summary from what Occhio already detects (no internet, no model, ~0 ms).

Pressing 'd' speaks this first, so there is never silence; a vision model (Claude
or the local SmolVLM) can add a richer sentence afterwards.

    "Two people ahead, 2 metres. One bike on the left, 4 metres. Red light."

Only confirmed obstacles of the active profile are used, grouped by class and
side, nearest first, at most `max_items` groups.
"""
from __future__ import annotations

from typing import Iterable

from config.profiles import (NUMBER_WORDS, OBJECTS, OBJECTS_PLURAL, SIDE_PHRASE, UNITS, Profile,
                             format_distance, lines_for)
from occhio.spatial import SpatialInfo


def describe_detections(infos: Iterable[SpatialInfo], profile: Profile, lang: str,
                        light: str | None = None, max_items: int = 3) -> str:
    obstacles = set(profile["classes"])
    groups: dict[tuple[str, str], list[float]] = {}
    for info in infos:
        if info.confirmed and info.distance_m is not None and info.cls_name in obstacles:
            groups.setdefault((info.cls_name, info.side), []).append(info.distance_m)

    parts: list[str] = []
    for (cls_name, side), dists in sorted(groups.items(), key=lambda kv: min(kv[1]))[:max_items]:
        n = len(dists)
        count = NUMBER_WORDS[lang].get(n, str(n))
        name = OBJECTS[lang].get(cls_name, cls_name) if n == 1 else OBJECTS_PLURAL[lang].get(cls_name, cls_name)
        dist = format_distance(min(dists), lang)
        unit = UNITS[lang][0 if dist == "1" else 1]
        parts.append(f"{count} {name} {SIDE_PHRASE[lang][side]}, {dist} {unit}")
    lines = lines_for(profile, lang)
    if light in ("red", "green"):
        parts.append(lines[f"light_{light}"])
    if not parts:
        return lines["scene_none"]
    text = ". ".join(p[:1].upper() + p[1:] for p in parts)
    return text
