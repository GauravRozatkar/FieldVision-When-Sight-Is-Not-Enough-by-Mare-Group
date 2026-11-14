"""Drawing: boxes, corridor, FPS / latency readout and event log.

Phase 1: boxes coloured by track status (confirmed green, uncertain yellow,
hidden grey dashed), label '#id cls conf', FPS and per-stage latency.
Phase 2: red for danger, distance / approach in the label, reliability bar,
"unreliable" banner and the last spoken sentence.
Phase 3: walking corridor and event log panel.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable, Mapping

import cv2
import numpy as np

from occhio.reliability import ReliabilityState
from occhio.spatial import SpatialInfo
from occhio.tracks import Track, TrackStatus

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
GREEN = (0, 200, 0)
YELLOW = (0, 220, 255)
GREY = (160, 160, 160)
RED = (0, 0, 255)

STATUS_COLOURS: dict[TrackStatus, tuple[int, int, int]] = {
    TrackStatus.CONFIRMED: GREEN,
    TrackStatus.UNCERTAIN: YELLOW,
    TrackStatus.HIDDEN: GREY,
}


def ascii_text(text: str) -> str:
    """OpenCV fonts are ASCII only: 'Visibilità' -> 'Visibilita'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _text(image: np.ndarray, text: str, org: tuple[int, int], colour: tuple[int, int, int],
          scale: float = 0.6, thickness: int = 2) -> None:
    """Text with a dark outline, readable on any background."""
    cv2.putText(image, text, org, _FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, _FONT, scale, colour, thickness, cv2.LINE_AA)


def _dashed_rect(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
                 colour: tuple[int, int, int], thickness: int = 2, dash: int = 10) -> None:
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        xe = min(x + dash, x2)
        cv2.line(image, (x, y1), (xe, y1), colour, thickness)
        cv2.line(image, (x, y2), (xe, y2), colour, thickness)
    for y in range(y1, y2, dash * 2):
        ye = min(y + dash, y2)
        cv2.line(image, (x1, y), (x1, ye), colour, thickness)
        cv2.line(image, (x2, y), (x2, ye), colour, thickness)


def draw_tracks(image: np.ndarray, tracks: Iterable[Track],
                infos: Mapping[int, SpatialInfo] | None = None,
                danger_ids: Iterable[int] = (), scale: float = 1.0, skip_classes: Iterable[str] = ()) -> None:
    """Draw every visible or hidden track, in place.

    Lost uncertain tracks (kept only to bridge a missed frame) are not drawn.
    Tracks in `danger_ids` are red; `infos` adds distance and approach speed.
    """
    infos = infos or {}
    danger = set(danger_ids)
    skip = set(skip_classes)
    h, w = image.shape[:2]
    for track in tracks:
        if track.status is not TrackStatus.HIDDEN and not track.visible:
            continue
        if track.cls_name in skip:
            continue
        colour = RED if track.track_id in danger else STATUS_COLOURS[track.status]
        x1, y1, x2, y2 = (int(round(c * scale)) for c in track.box)
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue  # predicted box left the frame
        if track.status is TrackStatus.HIDDEN:
            _dashed_rect(image, (x1, y1), (x2, y2), colour)
            label = f"#{track.track_id} {track.cls_name} hidden"
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3 if track.track_id in danger else 2)
            label = f"#{track.track_id} {track.cls_name} {track.conf:.2f}"
            info = infos.get(track.track_id)
            if info is not None and info.distance_m is not None:
                src = "" if info.distance_source == "depth" else "~"  # ~ = from box size
                label += f" {src}{info.distance_m:.1f}m"
                if info.approach_mps >= 0.3:
                    label += f" >{info.approach_mps:.1f}m/s"
                if info.fast:
                    label += " FAST"
        _text(image, label, (x1 + 2, max(16, y1 - 6)), colour, scale=0.55)


def draw_stats(image: np.ndarray, fps: float, stage_ms: Mapping[str, float]) -> None:
    """FPS and per-stage latency in the top-left corner, in place."""
    _text(image, f"FPS {fps:4.1f}", (12, 34), GREEN, scale=1.0)
    line = "  ".join(f"{name} {ms:4.1f}ms" for name, ms in stage_ms.items())
    if line:
        _text(image, line, (12, 62), GREEN, scale=0.55)


def draw_status(image: np.ndarray, reliability: ReliabilityState | None, last_message: str | None,
                header: str = "", banner: str = "") -> None:
    """Reliability bar (top right), `banner` while unreliable, and the last spoken sentence (bottom)."""
    h, w = image.shape[:2]
    if header:
        _text(image, header, (12, 90), GREEN, scale=0.55)
    if reliability is not None:
        x0, y0, bw, bh = w - 230, 16, 200, 16
        colour = GREEN if reliability.reliable else RED
        cv2.rectangle(image, (x0, y0), (x0 + bw, y0 + bh), (40, 40, 40), -1)
        cv2.rectangle(image, (x0, y0), (x0 + int(bw * reliability.score), y0 + bh), colour, -1)
        _text(image, f"rel {reliability.score:.2f} {reliability.reason}", (x0, y0 + bh + 20),
              colour, scale=0.5)
        if not reliability.reliable and banner:
            banner = ascii_text(banner).upper()
            (tw, th), _ = cv2.getTextSize(banner, _FONT, 1.2, 3)
            cv2.rectangle(image, (0, h // 2 - th - 20), (w, h // 2 + 20), (0, 0, 120), -1)
            _text(image, banner, ((w - tw) // 2, h // 2), (255, 255, 255), scale=1.2, thickness=3)
    if last_message:
        _text(image, ascii_text(last_message), (12, h - 18), (255, 255, 255), scale=0.8)


def draw_corridor(image: np.ndarray, polygon: list[tuple[int, int]], colour: tuple[int, int, int] = (255, 200, 0),
                  alpha: float = 0.12, thickness: int = 2) -> None:
    """Walking corridor: translucent fill plus outline, in place."""
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    layer = image.copy()
    cv2.fillPoly(layer, [pts], colour)
    cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, dst=image)
    cv2.polylines(image, [pts], True, colour, thickness, cv2.LINE_AA)


def draw_event_log(image: np.ndarray, entries: Iterable[tuple[str, str, bool]], width: int = 380,
                   top: int = 70) -> None:
    """Panel on the right with the latest spoken lines: (HH:MM:SS, text, urgent), newest last."""
    entries = list(entries)
    if not entries:
        return
    h, w = image.shape[:2]
    x0 = max(0, w - width - 10)
    line_h = 24
    y1 = min(h - 50, top + 12 + line_h * len(entries))
    panel = image[top:y1, x0:w - 10]
    if panel.size:
        panel[:] = (panel * 0.35).astype(np.uint8)  # darken behind the text
    y = top + line_h
    for stamp, text, urgent in entries:
        colour = RED if urgent else (230, 230, 230)
        line = ascii_text(f"{stamp} {text}")
        while line and cv2.getTextSize(line, _FONT, 0.45, 1)[0][0] > width - 12:
            line = line[:-2]
        cv2.putText(image, line, (x0 + 6, y), _FONT, 0.45, colour, 1, cv2.LINE_AA)
        y += line_h
