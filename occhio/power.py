"""Power saving (F29): skip detection frames when nothing moves.

A fanless laptop throttles when YOLO runs flat out. When the image has been
still (tiny grey difference between frames) and nothing is tracked for
`still_after_s`, the detector runs only on every `detect_every`th frame. Any
motion switches back to every frame at once.
"""
from __future__ import annotations

import cv2
import numpy as np

from config.settings import ECO


class MotionGate:
    def __init__(self, settings: dict | None = None) -> None:
        cfg = {**ECO, **(settings or {})}
        self.still_after_s = float(cfg["still_after_s"])
        self.detect_every = max(1, int(cfg["detect_every"]))
        self.threshold = float(cfg["motion_threshold"])
        self._prev: np.ndarray | None = None
        self._last_activity: float | None = None
        self._skipped = 0
        self.eco = False
        self.motion = 0.0

    def should_detect(self, image: np.ndarray, now: float, tracking_something: bool) -> bool:
        small = cv2.cvtColor(cv2.resize(image, (160, 90), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        small = small.astype(np.int16)
        self.motion = float(np.abs(small - self._prev).mean()) if self._prev is not None else 255.0
        self._prev = small
        if self.motion >= self.threshold or tracking_something or self._last_activity is None:
            self._last_activity = now
        self.eco = now - self._last_activity >= self.still_after_s
        if not self.eco:
            self._skipped = 0
            return True
        self._skipped += 1
        if self._skipped >= self.detect_every:
            self._skipped = 0
            return True
        return False
