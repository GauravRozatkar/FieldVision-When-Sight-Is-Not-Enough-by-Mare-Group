"""Step / drop-off warning (F28) from the depth map.

The floor right in front of a walking person is normally the closest thing in
the bottom of the image, so its disparity (inverse depth) is higher than the
strip just above it. When the bottom strip of the path (middle third of the
width) is suddenly *farther* than the strip above it, the floor drops away:
stairs going down, a kerb, a hole. After `consecutive` depth frames in a row
this says "Gradino in basso", then waits `cooldown_s`.

Relative depth makes this a heuristic: it is tested on synthetic maps but must
be checked on real stairs at the venue.
"""
from __future__ import annotations

import numpy as np

from config.settings import STEP
from occhio.depth import DepthFrame


class StepDetector:
    def __init__(self, settings: dict | None = None) -> None:
        cfg = {**STEP, **(settings or {})}
        self.strip = float(cfg["strip"])
        self.ratio_min = float(cfg["ratio_min"])
        self.consecutive = int(cfg["consecutive"])
        self.cooldown_s = float(cfg["cooldown_s"])
        self._last_ts: float | None = None
        self._hits = 0
        self._last_warn: float | None = None
        self.ratio: float | None = None  # last bottom/above ratio, for the overlay / tuning

    def update(self, depth: DepthFrame | None, now: float, path_blocked: bool = False) -> bool:
        """True when a step/drop-off should be announced now. Call every frame; works on new maps only."""
        if depth is None or depth.timestamp == self._last_ts:
            return False
        self._last_ts = depth.timestamp
        if path_blocked:  # someone standing in the bottom of the image hides the floor
            self._hits = 0
            return False
        disp = depth.disparity
        h, w = disp.shape
        n = max(1, int(h * self.strip))
        cols = slice(w // 3, 2 * w // 3)
        bottom = float(np.median(disp[h - n:, cols]))
        above = float(np.median(disp[h - 2 * n:h - n, cols]))
        if above <= 0:
            return False
        self.ratio = bottom / above
        self._hits = self._hits + 1 if self.ratio < self.ratio_min else 0
        if self._hits >= self.consecutive and (self._last_warn is None or now - self._last_warn >= self.cooldown_s):
            self._last_warn = now
            self._hits = 0
            return True
        return False
