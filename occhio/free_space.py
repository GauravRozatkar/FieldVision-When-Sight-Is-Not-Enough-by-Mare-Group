"""Walls and other big things in the way that the detector has no name for.

YOLO knows 80 object types, but not walls, doors, pillars or a parked van seen up
close. This module looks at the depth map in the walking path (middle third of
the image) from the bottom up:
- on the floor, depth increases steadily going up the image (the floor recedes);
- an upright surface (a wall, a door) keeps about the same depth over many rows.
The row where the floor stops and an upright surface starts is the wall's base.
Its distance follows from geometry alone (camera height + angle below the
horizon, using the phone's tilt when known), so the depth model's unknown scale
does not matter.

A level camera at chest height cannot see the floor closer than about 3 m, so a
nearer wall fills the bottom of the view and its base is out of sight. For that,
the depth model's scale is learned from the floor whenever floor is visible
(each floor row has a known distance, so metres = scale / disparity), and the
wall's distance comes from its disparity. With no scale learned yet nothing is
reported rather than a wrong number.

Needs a camera at a known height looking forward (a phone worn on the chest), so
it is on by default only with --source phone (--walls forces it on).
A result is reported only when it holds for `stable_frames` depth maps in a row.
"""
from __future__ import annotations

import statistics
from collections import deque

import numpy as np

from config.settings import FREE_SPACE
from occhio.depth import DepthFrame


def column_profile(disparity: np.ndarray) -> np.ndarray:
    """Median disparity of the middle third, row by row (top to bottom), lightly smoothed."""
    w = disparity.shape[1]
    profile = np.median(disparity[:, w // 3: 2 * w // 3], axis=1).astype(np.float64)
    return np.convolve(profile, np.ones(5) / 5, mode="same")


def wall_base_row(disparity: np.ndarray, flat_rel: float, flat_rows: float, floor_rel: float
                  ) -> tuple[float | None, bool, list[int]]:
    """(wall base as a fraction of the height or None, whether floor was seen below it, floor rows).

    Scans the middle third from the bottom up: floor = depth recedes going up; an upright
    surface = the same depth over a window of rows.
    """
    h = disparity.shape[0]
    profile = column_profile(disparity)
    k = max(3, int(flat_rows * h))
    floor_rows: list[int] = []
    for r in range(h - 3, h // 2 + k, -1):  # bottom (minus smoothing edge) up to the image middle
        here, above = profile[r], profile[r - k]
        if here <= 0:
            continue
        change = (here - above) / here  # > 0: farther as we go up (floor)
        if change >= floor_rel:
            floor_rows.append(r)
        elif abs(change) < flat_rel:
            return (r if floor_rows else h - 1) / h, bool(floor_rows), floor_rows
    return None, bool(floor_rows), floor_rows


class FreeSpace:
    """Distance to the first wall-like surface in the walking path, or None."""

    def __init__(self, settings: dict | None = None) -> None:
        cfg = {**FREE_SPACE, **(settings or {})}
        self.flat_rel = float(cfg["flat_rel"])
        self.flat_rows = float(cfg["flat_rows"])
        self.floor_rel = float(cfg["floor_rel"])
        self.stable_frames = int(cfg["stable_frames"])
        self.max_m = float(cfg["max_m"])
        self._last_ts: float | None = None
        self._recent: deque[float | None] = deque(maxlen=self.stable_frames)
        self.distance_m: float | None = None
        self.scale: float | None = None  # metres = scale / disparity, learned from the floor

    def update(self, depth: DepthFrame | None, frame_shape: tuple[int, int], floor_distance_at_row,
               fallback_scale: float | None = None) -> float | None:
        """floor_distance_at_row(row_px, frame_shape) -> metres (SpatialReasoner.floor_distance_at_row).

        fallback_scale: a depth scale learned elsewhere (e.g. from full-body people), used
        when no floor has been seen yet.
        """
        if depth is None or depth.timestamp == self._last_ts:
            return self.distance_m
        self._last_ts = depth.timestamp
        disp = depth.disparity
        h = disp.shape[0]
        fh = frame_shape[0]
        frac, floor_seen, floor_rows = wall_base_row(disp, self.flat_rel, self.flat_rows, self.floor_rel)
        profile = column_profile(disp)
        # learn the depth scale from the visible floor (known distance per row)
        ks = []
        for r in floor_rows[::3]:
            z = floor_distance_at_row((r + 0.5) * fh / h, frame_shape)
            if z is not None and profile[r] > 0 and z <= 8.0:
                ks.append(z * profile[r])
        if len(ks) >= 3:
            k = statistics.median(ks)
            self.scale = k if self.scale is None else self.scale + 0.3 * (k - self.scale)
        dist: float | None = None
        if frac is not None:
            if floor_seen:  # the wall's base is visible: pure geometry
                dist = floor_distance_at_row(frac * fh, frame_shape)
            else:  # the wall fills the bottom of the view: use the learned scale
                scale = self.scale or fallback_scale
                wall_disp = float(np.median(profile[int(h * 0.75):h - 2]))
                if scale and wall_disp > 0:
                    dist = scale / wall_disp
        self._recent.append(dist if dist is not None and dist <= self.max_m else None)
        values = [d for d in self._recent if d is not None]
        if len(self._recent) == self.stable_frames and len(values) == self.stable_frames:
            med = statistics.median(values)
            if max(values) - min(values) <= 0.35 * med + 0.2:
                self.distance_m = med
        elif not values:
            self.distance_m = None
        return self.distance_m
