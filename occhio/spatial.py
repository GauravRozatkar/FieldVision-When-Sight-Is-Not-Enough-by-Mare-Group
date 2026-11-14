"""Spatial reasoning: direction, distance, speed and time-to-collision.

For each visible track SpatialReasoner computes:
- side: left / ahead / right by thirds of the frame width;
- clock: clock position from the horizontal angle (12 = straight ahead, 10..2);
- distance_m: from the box size and a typical real size when the whole object is
  in the frame (accurate to ~5% for people). When the box is cut by the frame
  edge (someone right in front of the camera) the closest of: depth map
  (k / median disparity in the box), shoulder width, and the visible height
  (an upper bound). Reported as the median of the last 3 samples.
- approach_mps: minus the slope of a robust line (Theil-Sen: median of pairwise
  slopes) through the last distance samples, so one bad depth value cannot flip
  it (positive = coming closer); lateral_mps from pixel motion;
- speed_mps, fast flag, and time_to_collision_s when approaching.

Depth scale: `main.py --calibrate` stores k (and the focal length) in
config/calibration.json. Without that file, k is learned on the fly from
full-body people, whose box-size distance is reliable.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config.profiles import CATEGORY, Profile
from config.settings import CALIBRATION_FILE, CAMERA, SPATIAL
from occhio.depth import DepthFrame
from occhio.tracks import Track, TrackStatus, box_centre

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class SpatialInfo:
    """Where one tracked object is and how it moves."""

    track_id: int
    cls_name: str
    confirmed: bool
    mean_conf: float
    side: str  # "left" | "ahead" | "right"
    clock: int  # 10, 11, 12, 1, 2
    distance_m: float | None
    distance_source: str | None  # "depth" | "size" | None
    approach_mps: float  # > 0 = getting closer
    lateral_mps: float
    speed_mps: float
    fast: bool
    ttc_s: float | None  # time to collision, only when approaching
    in_corridor: bool = True  # feet inside the walking corridor (only these can be "danger")
    x_norm: float = 0.5  # box centre / frame width, 0 = left edge (for stereo panning)


@dataclass
class Calibration:
    """Depth scale and focal length; persisted by --calibrate."""

    depth_k: float | None = None  # metres = depth_k / disparity
    focal_px: float | None = None
    frame_width: int | None = None  # focal_px is valid for this frame width
    fixed: bool = False  # True when loaded from file: no auto-calibration

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "Calibration":
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return cls()
        return cls(data.get("depth_k"), data.get("focal_px"), data.get("frame_width"), fixed=True)

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        Path(path).write_text(json.dumps(
            {"depth_k": self.depth_k, "focal_px": self.focal_px, "frame_width": self.frame_width},
            indent=2))


def side_of(cx: float, frame_w: int, previous: str | None = None, margin: float = 0.0) -> str:
    """left / ahead / right by thirds of the frame width.

    With `previous` and a `margin` (fraction of the width) the side only changes once
    the object is clearly past the boundary, so leaning a little does not flip
    "ahead" <-> "on the left" every second.
    """
    lo, hi = frame_w / 3, 2 * frame_w / 3
    m = margin * frame_w
    if previous == "left":
        lo += m
    elif previous == "right":
        hi -= m
    elif previous == "ahead":
        lo, hi = lo - m, hi + m
    if cx < lo:
        return "left"
    if cx > hi:
        return "right"
    return "ahead"


def corridor_polygon(corridor: dict, frame_shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Corridor trapezoid in pixels: bottom-left, bottom-right, top-right, top-left."""
    fh, fw = frame_shape
    top_y = corridor["top_y"] * fh
    tw, bw = corridor["top_w"] * fw / 2, corridor["bottom_w"] * fw / 2
    cx = fw / 2
    return [(round(cx - bw), fh - 1), (round(cx + bw), fh - 1), (round(cx + tw), round(top_y)),
            (round(cx - tw), round(top_y))]


def in_corridor(x: float, y: float, corridor: dict, frame_shape: tuple[int, int]) -> bool:
    """Is the point (pixels) inside the walking corridor trapezoid?"""
    fh, fw = frame_shape
    yn = min(1.0, y / fh)
    top_y = corridor["top_y"]
    if yn < top_y:
        return False
    frac = (yn - top_y) / (1.0 - top_y) if top_y < 1.0 else 1.0
    half_w = (corridor["top_w"] + frac * (corridor["bottom_w"] - corridor["top_w"])) / 2
    return abs(x / fw - 0.5) <= half_w


def clock_of(cx: float, frame_w: int, focal_px: float) -> int:
    """Clock position from the horizontal angle: 12 straight ahead, 30 degrees per hour, 10..2."""
    angle = math.degrees(math.atan2(cx - frame_w / 2, focal_px))
    hour = round(12 + angle / 30)
    hour = max(10, min(14, hour))
    return hour if hour <= 12 else hour - 12


def robust_slope(samples: Iterable[tuple[float, float]]) -> float:
    """Theil-Sen slope: median of the slopes between every pair of samples (units per second)."""
    pts = list(samples)
    slopes = [(v2 - v1) / (t2 - t1) for i, (t1, v1) in enumerate(pts) for t2, v2 in pts[i + 1:] if t2 > t1]
    if not slopes:
        return 0.0
    slopes.sort()
    mid = len(slopes) // 2
    return slopes[mid] if len(slopes) % 2 else (slopes[mid - 1] + slopes[mid]) / 2


def fit_slope(samples: Iterable[tuple[float, float]]) -> float:
    """Least-squares slope of value over time (units per second)."""
    pts = list(samples)
    n = len(pts)
    if n < 2:
        return 0.0
    mt = sum(t for t, _ in pts) / n
    mv = sum(v for _, v in pts) / n
    var = sum((t - mt) ** 2 for t, _ in pts)
    if var <= 0:
        return 0.0
    return sum((t - mt) * (v - mv) for t, v in pts) / var


class SpatialReasoner:
    """Turns tracks (+ the latest depth map) into SpatialInfo, keeping distance history."""

    def __init__(self, profile: Profile, settings: dict | None = None,
                 calibration: Calibration | None = None, hfov_deg: float | None = None) -> None:
        cfg = {**SPATIAL, **(settings or {})}
        self.profile = profile
        self.history = int(cfg["history"])
        self.min_samples = int(cfg["min_samples"])
        self.max_sample_age_s = float(cfg["max_sample_age_s"])
        self.approach_ema = float(cfg["approach_ema"])
        self.heights_m: dict[str, float] = dict(cfg["heights_m"])
        self.ground_classes = set(cfg.get("ground_classes", []))
        self.camera_height_m = float(CAMERA.get("height_m", 1.2))
        self.person_width_m = float(cfg["person_width_m"])
        self.edge_margin = float(cfg["edge_margin"])
        self.auto_rate = float(cfg["auto_calibrate_rate"])
        self.hfov_deg = float(hfov_deg if hfov_deg is not None else CAMERA["hfov_deg"])
        self.calibration = calibration or Calibration()
        self._samples: dict[int, deque[tuple[float, float]]] = {}
        self._approach: dict[int, float] = {}  # smoothed approach speed per track
        self._sides: dict[int, str] = {}  # last side per track (hysteresis)
        self.pitch_deg: float | None = None  # phone tilt below the horizon (from its accelerometer)
        self.user_speed_mps = 0.0  # the user's walking speed (from the phone's step counter)
        self.side_margin = float(cfg.get("side_margin", 0.05))

    def set_profile(self, profile: Profile) -> None:
        self.profile = profile

    # ------------------------------------------------------------- geometry
    def focal_px(self, frame_w: int, frame_h: int | None = None) -> float:
        """Focal length in pixels. hfov_deg is the field of view across the *long* side of the
        image, so an upright (portrait) phone camera gets the same, correct focal length."""
        cal = self.calibration
        if cal.focal_px and cal.frame_width:
            return cal.focal_px * frame_w / cal.frame_width
        long_side = max(frame_w, frame_h or 0)
        return (long_side / 2) / math.tan(math.radians(self.hfov_deg / 2))

    def is_cut(self, box: BBox, frame_shape: tuple[int, int]) -> bool:
        """True when the box touches the top or bottom edge: its height is not the object's."""
        fh, _ = frame_shape
        margin = self.edge_margin * fh
        return box[1] <= margin or box[3] >= fh - margin

    def size_distance(self, cls_name: str, box: BBox, frame_shape: tuple[int, int]) -> float | None:
        """Distance from box size: f * real_height / pixel_height (width for cut-off people)."""
        fh, fw = frame_shape
        x1, y1, x2, y2 = box
        f = self.focal_px(fw, fh)
        if self.is_cut(box, frame_shape) and cls_name == "person":
            width = x2 - x1
            return f * self.person_width_m / width if width > 1 else None
        real_h = self.heights_m.get(cls_name)
        height = y2 - y1
        if real_h is None or height <= 1:
            return None
        return f * real_h / height

    def ground_distance(self, box: BBox, frame_shape: tuple[int, int]) -> float | None:
        """Distance of a point on the floor from its angle below the horizon.

        The camera is assumed level unless the phone reports its tilt (pitch_deg,
        degrees below the horizon, from gravity): then the angle is tilt + pixel angle.
        """
        return self.floor_distance_at_row(box[3], frame_shape)

    def floor_distance_at_row(self, row: float, frame_shape: tuple[int, int]) -> float | None:
        fh, fw = frame_shape
        f = self.focal_px(fw, fh)
        angle = math.atan2(row - fh / 2, f) + math.radians(self.pitch_deg or 0.0)
        if angle <= math.radians(1.0):
            return None  # at or above the horizon: not on the floor in front of us
        return self.camera_height_m / math.tan(angle)

    def is_full_body_person(self, cls_name: str, box: BBox, frame_shape: tuple[int, int]) -> bool:
        fh, fw = frame_shape
        x1, y1, x2, y2 = box
        if cls_name != "person":
            return False
        margin = self.edge_margin * fh
        edges_ok = y1 > margin and y2 < fh - margin and x1 > margin and x2 < fw - margin
        aspect = (y2 - y1) / max(1.0, x2 - x1)
        return edges_ok and 1.8 <= aspect <= 5.0

    def distance(self, track: Track, depth: DepthFrame | None,
                 frame_shape: tuple[int, int]) -> tuple[float | None, str | None]:
        """(metres, source). Box size when the object is whole, depth when it is cut."""
        box = track.box
        size_d = self.size_distance(track.cls_name, box, frame_shape)
        disparity = depth.median_in_box(box) if depth is not None else None
        cal = self.calibration
        if (disparity is not None and size_d is not None and not cal.fixed
                and track.status is TrackStatus.CONFIRMED
                and self.is_full_body_person(track.cls_name, box, frame_shape)):
            k_est = size_d * disparity  # learn the depth scale from a reliable size distance
            cal.depth_k = k_est if cal.depth_k is None else cal.depth_k + self.auto_rate * (k_est - cal.depth_k)
        if track.cls_name in self.ground_classes:
            ground_d = self.ground_distance(box, frame_shape)
            if ground_d is not None:
                return ground_d, "ground"
        cut = self.is_cut(box, frame_shape)
        if not cut and size_d is not None:
            return size_d, "size"
        # Cut by the frame edge: the visible height is shorter than the real one, so the
        # height-based distance is an upper bound. Take the closest estimate: for a safety
        # aid a distance that is too short is a false alarm, one that is too long is a miss.
        options: list[tuple[float, str]] = []
        real_h = self.heights_m.get(track.cls_name)
        if real_h is not None and box[3] - box[1] > 1:
            options.append((self.focal_px(frame_shape[1], frame_shape[0]) * real_h / (box[3] - box[1]), "size"))
        if size_d is not None:
            options.append((size_d, "size"))  # shoulder width for people
        if disparity is not None and cal.depth_k:
            options.append((cal.depth_k / disparity, "depth"))
        if not options:
            return None, None
        return min(options)

    def _side(self, track_id: int, cx: float, fw: int) -> str:
        side = side_of(cx, fw, self._sides.get(track_id), self.side_margin)
        self._sides[track_id] = side
        return side

    # ---------------------------------------------------------------- update
    def update(self, tracks: Iterable[Track], depth: DepthFrame | None,
               frame_shape: tuple[int, int], now: float) -> list[SpatialInfo]:
        fh, fw = frame_shape
        focal = self.focal_px(fw, fh)
        infos: list[SpatialInfo] = []
        alive: set[int] = set()
        for track in tracks:
            alive.add(track.track_id)
            if not track.visible:
                continue
            cx, _ = box_centre(track.box)
            dist, source = self.distance(track, depth, frame_shape)
            samples = self._samples.setdefault(track.track_id, deque(maxlen=self.history))
            if samples and now - samples[-1][0] > self.max_sample_age_s:
                samples.clear()
                self._approach.pop(track.track_id, None)
            if dist is not None:
                samples.append((now, dist))
                dist = statistics.median(d for _, d in list(samples)[-3:])  # smooth jitter
            raw = -robust_slope(samples) if len(samples) >= self.min_samples else 0.0
            prev = self._approach.get(track.track_id)
            approach = raw if prev is None or len(samples) < self.min_samples else prev + self.approach_ema * (raw - prev)
            self._approach[track.track_id] = approach
            vx, _ = track.centre_velocity(5)
            lateral = vx * dist / focal if dist is not None else 0.0
            speed = math.hypot(approach, lateral)
            # Walking toward something that does not move (a wall, a bench, a bag): the vision estimate
            # of its approach is noisy, the phone's step counter is not.
            if (self.user_speed_mps > 0.2 and CATEGORY.get(track.cls_name, "obstacle") == "obstacle"
                    and in_corridor(cx, track.box[3], self.profile["corridor"], frame_shape)):
                approach = max(approach, 0.8 * self.user_speed_mps)
            approaching = approach >= float(self.profile["approach_min_mps"])
            ttc = dist / approach if (approaching and dist is not None) else None
            infos.append(SpatialInfo(
                track_id=track.track_id, cls_name=track.cls_name,
                confirmed=track.status is TrackStatus.CONFIRMED, mean_conf=track.mean_conf,
                side=self._side(track.track_id, cx, fw), clock=clock_of(cx, fw, focal),
                distance_m=dist, distance_source=source,
                approach_mps=approach, lateral_mps=lateral, speed_mps=speed,
                fast=speed >= float(self.profile["fast_mps"]), ttc_s=ttc,
                in_corridor=in_corridor(cx, track.box[3], self.profile["corridor"], frame_shape),
                x_norm=cx / fw,
            ))
        for track_id in [tid for tid in self._samples if tid not in alive]:
            del self._samples[track_id]
            self._approach.pop(track_id, None)
            self._sides.pop(track_id, None)
        return infos
