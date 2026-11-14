"""Track memory: confirmed / uncertain / hidden state per track id.

ByteTrack gives IDs; TrackManager decides what we trust:

- uncertain: new or low-confidence; drawn yellow, never spoken.
- confirmed: seen in >= confirm_frames frames with mean confidence >= confirm_mean_conf.
- hidden:    a confirmed track that disappeared (occlusion). Its box is predicted
             with constant velocity for up to hidden_ttl_s, then it is deleted.
             If it comes back it is confirmed again at once.

Coming back: either ByteTrack reuses the same id, or (common after a partial
occlusion, when the box shrinks and ByteTrack loses the motion) it hands out a
new id. A new or still-unconfirmed id that appears along a hidden track's
predicted path, with the same class and a similar height, is re-identified as
that track: it keeps the old id from then on.

Lost memory: when a hidden track runs out of time it is not drawn any more, but
its appearance (a colour histogram of the torso area) is kept for lost_memory_s.
A new track whose appearance matches during its first reid_window_s takes the
old id back, so someone who walks out of view and returns keeps their id.

Lost uncertain tracks are kept invisibly for uncertain_ttl_s (so one missed
frame does not reset their count), then deleted. All times are passed in by the
caller, which keeps the class deterministic and easy to test.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import cv2
import numpy as np

from config.settings import TRACKING
from occhio.detector import BBox, Detection


class TrackStatus(str, Enum):
    UNCERTAIN = "uncertain"
    CONFIRMED = "confirmed"
    HIDDEN = "hidden"


def box_centre(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def appearance_signature(image: np.ndarray, box: BBox) -> np.ndarray | None:
    """Colour histogram (HSV 12x6x4, L1-normalised) of the box's torso area.

    The area is the middle 60% of the width and 15-65% of the height: for a
    person that is the clothes, which stay the same when they come back.
    Returns None when the area is too small or outside the image.
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ax1, ax2 = int(max(0, x1 + 0.2 * w)), int(min(w_img, x1 + 0.8 * w))
    ay1, ay2 = int(max(0, y1 + 0.15 * h)), int(min(h_img, y1 + 0.65 * h))
    if ax2 - ax1 < 8 or ay2 - ay1 < 8:
        return None
    hsv = cv2.cvtColor(image[ay1:ay2, ax1:ax2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, None, 1.0, 0.0, cv2.NORM_L1).flatten()


def appearance_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two signatures: 0 = identical, 1 = nothing in common."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


@dataclass
class Track:
    """Everything we remember about one track id."""

    track_id: int
    cls_name: str
    history_len: int = 30
    boxes: deque[tuple[float, BBox]] = field(init=False)  # (time, box)
    confs: deque[float] = field(init=False)
    frames_seen: int = 0
    last_seen: float = 0.0
    first_seen: float = 0.0
    status: TrackStatus = TrackStatus.UNCERTAIN
    visible: bool = True  # detected in the latest update
    was_confirmed: bool = False
    predicted_box: BBox | None = None  # set while hidden
    appearance: np.ndarray | None = None  # running-average appearance signature

    def __post_init__(self) -> None:
        self.boxes = deque(maxlen=self.history_len)
        self.confs = deque(maxlen=self.history_len)

    @property
    def mean_conf(self) -> float:
        """Mean confidence over the recent history (last history_len frames)."""
        return sum(self.confs) / len(self.confs) if self.confs else 0.0

    @property
    def conf(self) -> float:
        """Latest detection confidence."""
        return self.confs[-1] if self.confs else 0.0

    @property
    def last_box(self) -> BBox:
        """Last observed box."""
        return self.boxes[-1][1]

    @property
    def box(self) -> BBox:
        """Box to use now: predicted while hidden, observed otherwise."""
        if self.status is TrackStatus.HIDDEN and self.predicted_box is not None:
            return self.predicted_box
        return self.last_box

    @property
    def median_height(self) -> float:
        """Typical box height; robust to the box shrinking during a side occlusion."""
        return statistics.median(b[3] - b[1] for _, b in self.boxes) if self.boxes else 0.0

    def reference_boxes(self, full_ratio: float = 0.0) -> list[tuple[float, BBox]]:
        """History boxes not distorted by partial occlusion (height >= full_ratio x median).

        While someone walks behind a chair the box shrinks to the visible part,
        which would corrupt the velocity; those boxes are left out.
        """
        boxes = list(self.boxes)
        if full_ratio <= 0 or not boxes:
            return boxes
        min_h = full_ratio * self.median_height
        full = [(t, b) for t, b in boxes if b[3] - b[1] >= min_h]
        return full or boxes

    def centre_velocity(self, window: int, full_ratio: float = 0.0) -> tuple[float, float]:
        """Velocity of the box centre in pixels/s: median step over the last `window` reference boxes.

        The median keeps one jumpy box (a wrong or partial detection) from
        throwing the prediction off.
        """
        recent = self.reference_boxes(full_ratio)[-(window + 1):]
        vxs: list[float] = []
        vys: list[float] = []
        for (t0, b0), (t1, b1) in zip(recent, recent[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            (cx0, cy0), (cx1, cy1) = box_centre(b0), box_centre(b1)
            vxs.append((cx1 - cx0) / dt)
            vys.append((cy1 - cy0) / dt)
        if not vxs:
            return 0.0, 0.0
        return statistics.median(vxs), statistics.median(vys)

    def predict(self, now: float, window: int, full_ratio: float = 0.0) -> BBox:
        """Last reference box moved by the centre velocity until `now` (size unchanged)."""
        ref = self.reference_boxes(full_ratio)
        if not ref:
            raise ValueError("track has no boxes")
        t_ref, (x1, y1, x2, y2) = ref[-1]
        vx, vy = self.centre_velocity(window, full_ratio)
        dt = max(0.0, now - t_ref)
        dx, dy = vx * dt, vy * dt
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class TrackManager:
    """Keeps one Track per id and applies the trust, occlusion and re-identification rules."""

    def __init__(self, settings: dict | None = None) -> None:
        cfg = {**TRACKING, **(settings or {})}
        self.confirm_frames = int(cfg["confirm_frames"])
        self.confirm_mean_conf = float(cfg["confirm_mean_conf"])
        self.class_confirm: dict[str, tuple[float, int]] = dict(cfg.get("class_confirm", {}))
        self.hidden_ttl_s = float(cfg["hidden_ttl_s"])
        self.uncertain_ttl_s = float(cfg["uncertain_ttl_s"])
        self.history_len = int(cfg["history_len"])
        self.velocity_window = int(cfg["velocity_window"])
        self.full_box_ratio = float(cfg["full_box_ratio"])
        self.reid_gate = float(cfg["reid_gate"])
        self.reid_path_overshoot = float(cfg["reid_path_overshoot"])
        self.reid_max_height_ratio = float(cfg["reid_max_height_ratio"])
        self.reid_min_conf = float(cfg["reid_min_conf"])
        self.lost_memory_s = float(cfg["lost_memory_s"])
        self.appearance_max_dist = float(cfg["appearance_max_dist"])
        self.appearance_min_conf = float(cfg["appearance_min_conf"])
        self.appearance_ema = float(cfg["appearance_ema"])
        self.reid_window_s = float(cfg["reid_window_s"])
        self.tracks: dict[int, Track] = {}  # active: uncertain, confirmed, hidden
        self.lost: dict[int, Track] = {}  # not drawn; kept only to re-identify by appearance
        self.events: list[str] = []  # what happened in the latest update, for the console
        self._alias: dict[int, int] = {}  # ByteTrack id -> our track id (after re-identification)

    def update(self, detections: Iterable[Detection], now: float,
               image: np.ndarray | None = None) -> list[Track]:
        """Feed one frame's detections taken at time `now`; return the active tracks.

        Pass the frame `image` to enable appearance re-identification.
        """
        self.events = []
        seen: set[int] = set()
        # Strongest first, so a duplicate box of the same object is the one dropped.
        for det in sorted(detections, key=lambda d: d.conf, reverse=True):
            sig = appearance_signature(image, det.xyxy) if image is not None else None
            track_id = self._alias.get(det.track_id, det.track_id)
            track = self.tracks.get(track_id)
            if track is None and track_id in self.lost:  # ByteTrack itself kept the id
                track = self._revive(self.lost.pop(track_id), now, "same id")
            if track is None or not track.was_confirmed:
                # New, or still unconfirmed: it may be a hidden track coming back.
                match = self._match_hidden(det, now, seen)
                if match is not None:
                    if track is not None:
                        self._drop(track)  # the short-lived duplicate
                    self._alias[det.track_id] = match.track_id
                    self.events.append(f"#{match.track_id} {match.cls_name} back (path)")
                    track = match
            if track is None:
                track = Track(det.track_id, det.cls_name, self.history_len, first_seen=now)
                self.tracks[det.track_id] = track
            if track.track_id == det.track_id and now - track.first_seen <= self.reid_window_s:
                # Young track: maybe someone we lost a while ago.
                old = self._match_lost(track, det, sig)
                if old is not None:
                    track = self._adopt(track, old, det.track_id, now)
            if track.track_id in seen:
                continue  # two ByteTrack ids resolved to one track this frame
            seen.add(track.track_id)
            self._observe(track, det, now, sig)

        for track_id in [tid for tid in self.tracks if tid not in seen]:
            self._miss(self.tracks[track_id], now)
        for track_id in [tid for tid, t in self.lost.items() if now - t.last_seen > self.lost_memory_s]:
            del self.lost[track_id]
            self._unalias(track_id)

        return list(self.tracks.values())

    def reset(self) -> None:
        """Forget every track."""
        self.tracks.clear()
        self.lost.clear()
        self._alias.clear()
        self.events = []

    # ----------------------------------------------------------------- rules
    def _observe(self, track: Track, det: Detection, now: float, sig: np.ndarray | None) -> None:
        track.cls_name = det.cls_name
        track.boxes.append((now, det.xyxy))
        track.confs.append(det.conf)
        track.frames_seen += 1
        track.last_seen = now
        track.visible = True
        track.predicted_box = None
        if sig is not None and det.conf >= self.appearance_min_conf and self._is_full(track, det.xyxy):
            if track.appearance is None:
                track.appearance = sig
            else:
                track.appearance = (1 - self.appearance_ema) * track.appearance + self.appearance_ema * sig
        min_conf, min_frames = self.class_confirm.get(track.cls_name, (self.confirm_mean_conf, self.confirm_frames))
        if track.was_confirmed or (track.frames_seen >= min_frames and track.mean_conf >= min_conf):
            if not track.was_confirmed:
                self.events.append(f"#{track.track_id} {track.cls_name} confirmed")
            track.status = TrackStatus.CONFIRMED
            track.was_confirmed = True
        else:
            track.status = TrackStatus.UNCERTAIN

    def _is_full(self, track: Track, box: BBox) -> bool:
        """False for a box shrunk by partial occlusion (it would spoil the appearance)."""
        widths = [b[2] - b[0] for _, b in track.boxes]
        heights = [b[3] - b[1] for _, b in track.boxes]
        if len(heights) < 3:
            return True
        return (box[3] - box[1] >= self.full_box_ratio * statistics.median(heights)
                and box[2] - box[0] >= 0.8 * statistics.median(widths))

    def _miss(self, track: Track, now: float) -> None:
        track.visible = False
        lost_for = now - track.last_seen
        if not track.was_confirmed:
            if lost_for > self.uncertain_ttl_s:
                self._drop(track)
            return
        if lost_for > self.hidden_ttl_s:
            del self.tracks[track.track_id]
            if track.appearance is not None and self.lost_memory_s > 0:
                self.lost[track.track_id] = track  # keep the aliases: ByteTrack may still use them
                self.events.append(f"#{track.track_id} {track.cls_name} lost (remembered)")
            else:
                self._unalias(track.track_id)
                self.events.append(f"#{track.track_id} {track.cls_name} gone")
            return
        if track.status is not TrackStatus.HIDDEN:
            self.events.append(f"#{track.track_id} {track.cls_name} hidden")
        track.status = TrackStatus.HIDDEN
        track.predicted_box = track.predict(now, self.velocity_window, self.full_box_ratio)

    def _drop(self, track: Track) -> None:
        del self.tracks[track.track_id]
        self._unalias(track.track_id)

    def _unalias(self, track_id: int) -> None:
        self._alias = {k: v for k, v in self._alias.items() if v != track_id}

    def _revive(self, old: Track, now: float, how: str) -> Track:
        """Bring a lost track back as confirmed; its old boxes are too old to predict from."""
        old.boxes.clear()
        old.status = TrackStatus.CONFIRMED
        self.tracks[old.track_id] = old
        self.events.append(f"#{old.track_id} {old.cls_name} back after {now - old.last_seen:.1f}s ({how})")
        return old

    def _adopt(self, young: Track, old: Track, bytetrack_id: int, now: float) -> Track:
        """The young track is really the lost `old` one: give it the old id and memory."""
        del self.lost[old.track_id]
        del self.tracks[young.track_id]
        self.events.append(
            f"#{old.track_id} {old.cls_name} back after {now - old.last_seen:.1f}s (appearance)")
        young.track_id = old.track_id
        young.first_seen = old.first_seen
        young.frames_seen += old.frames_seen
        young.was_confirmed = True
        young.status = TrackStatus.CONFIRMED
        if old.appearance is not None:
            young.appearance = old.appearance if young.appearance is None else (
                0.5 * old.appearance + 0.5 * young.appearance)
        self.tracks[old.track_id] = young
        self._alias[bytetrack_id] = old.track_id
        return young

    def _match_lost(self, young: Track, det: Detection, sig: np.ndarray | None) -> Track | None:
        """Return the lost track that looks most like this young one, or None."""
        if det.conf < self.reid_min_conf or not self.lost:
            return None
        # Both the young track's average and this frame alone: while someone steps
        # out from behind something, the early (partial) frames skew the average.
        probes = [a for a in (young.appearance, sig) if a is not None]
        if not probes:
            return None
        best: Track | None = None
        best_dist = self.appearance_max_dist
        for old in self.lost.values():
            if old.cls_name != det.cls_name or old.appearance is None:
                continue
            dist = min(appearance_distance(a, old.appearance) for a in probes)
            if dist <= best_dist:
                best, best_dist = old, dist
        return best

    def _match_hidden(self, det: Detection, now: float, taken: set[int]) -> Track | None:
        """Return the hidden track this new id most likely is, by position, or None."""
        if det.conf < self.reid_min_conf:
            return None
        cx, cy = box_centre(det.xyxy)
        height = det.xyxy[3] - det.xyxy[1]
        best: Track | None = None
        best_dist = float("inf")
        for track in self.tracks.values():
            if (track.status is not TrackStatus.HIDDEN or track.track_id in taken
                    or track.cls_name != det.cls_name):
                continue
            ref_h = track.median_height
            if ref_h <= 0 or height <= 0:
                continue
            ratio = height / ref_h
            if not (1.0 / self.reid_max_height_ratio <= ratio <= self.reid_max_height_ratio):
                continue
            # Path: from where we last saw it to a bit past where it should be now,
            # so stopping, slowing down or speeding up behind the obstacle all match.
            t_ref, ref_box = track.reference_boxes(self.full_box_ratio)[-1]
            sx, sy = box_centre(ref_box)
            vx, vy = track.centre_velocity(self.velocity_window, self.full_box_ratio)
            dt = max(0.0, now - t_ref) * self.reid_path_overshoot
            dist = _point_segment_distance((cx, cy), (sx, sy), (sx + vx * dt, sy + vy * dt))
            if dist <= self.reid_gate * ref_h and dist < best_dist:
                best, best_dist = track, dist
        return best


def _point_segment_distance(p: tuple[float, float], a: tuple[float, float],
                            b: tuple[float, float]) -> float:
    """Euclidean distance from point p to segment a-b."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    abx, aby = bx - ax, by - ay
    length_sq = abx * abx + aby * aby
    u = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    qx, qy = ax + u * abx, ay + u * aby
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
