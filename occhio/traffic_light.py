"""Traffic light colour (F27): red or green, announced only when it changes.

Like OKO, only the most relevant signal counts: the biggest (= nearest)
traffic-light box in view. Its crop is converted to HSV and lit red vs green
pixels are counted. A colour must hold for `stable_frames` frames before it is
spoken, so a flicker or a passing red car does not produce a message.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from config.settings import TRAFFIC_LIGHT
from occhio.tracks import Track


def light_colour(image: np.ndarray, box: tuple[float, float, float, float],
                 settings: dict | None = None) -> str | None:
    """'red', 'green' or None for one traffic-light box."""
    cfg = {**TRAFFIC_LIGHT, **(settings or {})}
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(c)) for c in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (sat >= cfg["min_sat"]) & (val >= cfg["min_val"])
    red = np.zeros_like(lit)
    for lo, hi in cfg["red_hue"]:
        red |= (hue >= lo) & (hue <= hi)
    glo, ghi = cfg["green_hue"]
    green = (hue >= glo) & (hue <= ghi)
    n_red, n_green = int((lit & red).sum()), int((lit & green).sum())
    need = cfg["min_fraction"] * (x2 - x1) * (y2 - y1)
    if max(n_red, n_green) < need:
        return None
    return "red" if n_red > n_green else "green"


class TrafficLightMonitor:
    """Tracks the colour of the nearest traffic light and reports changes."""

    def __init__(self, settings: dict | None = None) -> None:
        self.cfg = {**TRAFFIC_LIGHT, **(settings or {})}
        self.stable_frames = int(self.cfg["stable_frames"])
        self.colour: str | None = None  # last announced colour
        self.box: tuple[float, float, float, float] | None = None  # current light, for the overlay
        self._candidate: str | None = None
        self._count = 0

    def reset(self) -> None:
        self.colour, self.box, self._candidate, self._count = None, None, None, 0

    def update(self, tracks: Iterable[Track], image: np.ndarray) -> str | None:
        """Returns the new colour ('red'/'green') when it changes and is stable, else None."""
        lights = [t for t in tracks if t.cls_name == "traffic light" and t.visible]
        if not lights:
            self.box = None
            self._candidate, self._count = None, 0
            return None
        nearest = max(lights, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
        self.box = nearest.box
        colour = light_colour(image, nearest.box, self.cfg)
        if colour is None:
            return None
        if colour == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = colour, 1
        if self._count >= self.stable_frames and colour != self.colour:
            self.colour = colour
            return colour
        return None
