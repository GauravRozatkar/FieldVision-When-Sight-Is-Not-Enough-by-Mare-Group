"""Restricted zone (F30, industrial): draw a polygon with the mouse; a person inside => warning.

Left click adds a corner, right click closes the polygon (or clears it when it is
already closed). Points are stored as fractions of the frame, so the zone survives
a resolution change, and saved to config/zone.json.
A person counts as inside when their feet (box bottom-centre) are inside.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config.settings import ZONE, ZONE_FILE
from occhio.tracks import Track, TrackStatus


class DangerZone:
    def __init__(self, path: Path | None = ZONE_FILE, settings: dict | None = None) -> None:
        cfg = {**ZONE, **(settings or {})}
        self.cooldown_s = float(cfg["cooldown_s"])
        self.path = path
        self.points: list[tuple[float, float]] = []
        self.closed = False
        self._last: dict[int, float] = {}
        self.inside: set[int] = set()  # track ids inside now (for the overlay)
        self.load()

    # ------------------------------------------------------------- editing
    def add_point(self, xn: float, yn: float) -> None:
        if self.closed:
            return
        self.points.append((min(1.0, max(0.0, xn)), min(1.0, max(0.0, yn))))

    def close_or_clear(self) -> None:
        """Right click: close an open polygon (3+ points), or clear a closed one."""
        if self.closed or len(self.points) < 3:
            self.points, self.closed = [], False
        else:
            self.closed = True
        self.save()

    def pixels(self, frame_shape: tuple[int, int]) -> list[tuple[int, int]]:
        fh, fw = frame_shape
        return [(round(x * fw), round(y * fh)) for x, y in self.points]

    def contains(self, x: float, y: float, frame_shape: tuple[int, int]) -> bool:
        if not self.closed:
            return False
        poly = np.array(self.pixels(frame_shape), dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    # --------------------------------------------------------------- check
    def check(self, tracks: Iterable[Track], frame_shape: tuple[int, int], now: float) -> list[int]:
        """Ids of confirmed people inside the zone that should be announced now."""
        self.inside = set()
        new: list[int] = []
        for t in tracks:
            if t.cls_name != "person" or not t.visible or t.status is not TrackStatus.CONFIRMED:
                continue
            x1, _, x2, y2 = t.box
            if self.contains((x1 + x2) / 2, y2, frame_shape):
                self.inside.add(t.track_id)
                last = self._last.get(t.track_id)
                if last is None or now - last >= self.cooldown_s:
                    self._last[t.track_id] = now
                    new.append(t.track_id)
        return new

    # ----------------------------------------------------------- persistence
    def load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(Path(self.path).read_text())
            self.points = [tuple(p) for p in data.get("points", [])]  # type: ignore[misc]
            self.closed = bool(data.get("closed")) and len(self.points) >= 3
        except (OSError, ValueError):
            pass

    def save(self) -> None:
        if self.path is None:
            return
        Path(self.path).write_text(json.dumps({"points": self.points, "closed": self.closed}, indent=2))
