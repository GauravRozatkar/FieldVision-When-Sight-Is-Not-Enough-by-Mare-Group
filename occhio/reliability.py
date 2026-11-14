"""Reliability score (0-1) of the guidance.

Each frame is scored on brightness (too dark / over-exposed), sharpness
(variance of the Laplacian) and the mean confidence of the confirmed tracks; a
covered lens (almost no contrast) scores 0. The score is smoothed with an
exponential moving average. Below `unreliable_below` the guidance is
unreliable until it climbs back above `reliable_above` (hysteresis, so it does
not flicker around the threshold).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from config.settings import RELIABILITY
from occhio.tracks import Track, TrackStatus


def ramp(value: float, lo: float, hi: float) -> float:
    """0 at or below lo, 1 at or above hi, linear in between."""
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


@dataclass(frozen=True)
class ReliabilityState:
    score: float  # smoothed, 0-1
    raw: float  # this frame only
    brightness: float  # mean grey 0-255
    sharpness: float  # variance of Laplacian
    contrast: float  # grey std
    mean_conf: float | None  # confirmed visible tracks, None if there are none
    blocked: bool
    reliable: bool
    reason: str  # "ok" | "surface" | "blocked" | "dark" | "bright" | "blur" | "low confidence"


class ReliabilityMonitor:
    """Scores every frame and keeps the smoothed state."""

    def __init__(self, settings: dict | None = None) -> None:
        cfg = {**RELIABILITY, **(settings or {})}
        self.analysis_width = int(cfg["analysis_width"])
        self.dark = tuple(cfg["dark"])
        self.bright = tuple(cfg["bright"])
        self.blur = tuple(cfg["blur"])
        self.blocked_std = float(cfg["blocked_std"])
        self.alpha = float(cfg["ema_alpha"])
        self.unreliable_below = float(cfg["unreliable_below"])
        self.reliable_above = float(cfg["reliable_above"])
        self.surface_max_sharpness = float(cfg["surface_max_sharpness"])
        self.surface_max_contrast = float(cfg["surface_max_contrast"])
        self.surface_min_light = float(cfg["surface_min_light"])
        self.score = 1.0
        self.reliable = True

    def measure(self, image: np.ndarray) -> tuple[float, float, float]:
        """(brightness, sharpness, contrast) of a BGR frame, on a downscaled grey copy."""
        h, w = image.shape[:2]
        if w > self.analysis_width:
            image = cv2.resize(image, (self.analysis_width, max(1, round(h * self.analysis_width / w))),
                               interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mean, std = cv2.meanStdDev(grey)
        sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        return float(mean[0][0]), sharpness, float(std[0][0])

    def update(self, image: np.ndarray, tracks: Iterable[Track] = ()) -> ReliabilityState:
        brightness, sharpness, contrast = self.measure(image)
        confs = [t.mean_conf for t in tracks if t.status is TrackStatus.CONFIRMED and t.visible]
        mean_conf = sum(confs) / len(confs) if confs else None

        light = ramp(brightness, *self.dark) * (1.0 - ramp(brightness, *self.bright))
        sharp = ramp(sharpness, *self.blur)
        blocked = contrast < self.blocked_std
        # Plain surface right in front: the camera works fine, there is just nothing but wall.
        surface = (not blocked and light >= self.surface_min_light and sharpness < self.surface_max_sharpness
                   and contrast < self.surface_max_contrast)
        raw = 0.0 if blocked else 1.0 if surface else min(light, sharp)
        if mean_conf is not None:
            raw *= 0.6 + 0.4 * mean_conf  # low-confidence detections lower trust a little

        self.score += self.alpha * (raw - self.score)
        if self.reliable and self.score < self.unreliable_below:
            self.reliable = False
        elif not self.reliable and self.score > self.reliable_above:
            self.reliable = True

        if blocked:
            reason = "blocked"
        elif surface:
            reason = "surface"
        elif ramp(brightness, *self.dark) < 0.5:
            reason = "dark"
        elif ramp(brightness, *self.bright) > 0.5:
            reason = "bright"
        elif sharp < 0.5:
            reason = "blur"
        elif mean_conf is not None and mean_conf < 0.5:
            reason = "low confidence"
        else:
            reason = "ok"
        return ReliabilityState(self.score, raw, brightness, sharpness, contrast, mean_conf,
                                blocked, self.reliable, reason)
