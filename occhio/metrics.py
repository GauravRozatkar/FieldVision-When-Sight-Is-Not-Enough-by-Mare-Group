"""Metrics: FPS and latency now; warnings, false alarms and metrics.csv later.

Phase 0: FpsCounter (smoothed frames per second over a sliding time window).
Phase 1: StageTimes, per-stage latency in ms (capture, detect, logic, ...).
Phase 4: MetricsLogger writes one row per second to metrics.csv (FPS, latency per
stage, warnings) and measures warning lead time: seconds between the first
warning about an object and that object reaching the danger distance.
summary() gives the numbers for the slides.
"""
from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Mapping, TextIO


class FpsCounter:
    """Frames per second over the last `window_s` seconds."""

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = window_s
        self._ticks: deque[float] = deque()

    def tick(self, now: float | None = None) -> float:
        """Register one frame and return the current FPS."""
        now = time.perf_counter() if now is None else now
        self._ticks.append(now)
        while self._ticks and now - self._ticks[0] > self.window_s:
            self._ticks.popleft()
        return self.fps

    @property
    def fps(self) -> float:
        """Current FPS; 0.0 until two frames have been seen."""
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        return (len(self._ticks) - 1) / span if span > 0 else 0.0


class StageTimes:
    """Smoothed latency per pipeline stage, in milliseconds.

    `smoothed` (EMA) is for the on-screen readout; `pop_window_means()` returns
    the plain averages since the last call, for the once-a-second console line.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.smoothed: dict[str, float] = {}
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def add(self, stage: str, ms: float) -> None:
        prev = self.smoothed.get(stage)
        self.smoothed[stage] = ms if prev is None else prev + self.alpha * (ms - prev)
        self._sums[stage] = self._sums.get(stage, 0.0) + ms
        self._counts[stage] = self._counts.get(stage, 0) + 1

    def pop_window_means(self) -> dict[str, float]:
        means = {s: self._sums[s] / self._counts[s] for s in self._sums if self._counts[s]}
        self._sums.clear()
        self._counts.clear()
        return means


class MetricsLogger:
    """metrics.csv, one row per second, plus warning lead times."""

    FIELDS = ["time_s", "fps", "capture_ms", "detect_ms", "logic_ms", "total_ms", "depth_ms",
              "reliability", "confirmed", "warnings", "eco"]

    def __init__(self, path: str | Path | None) -> None:
        self._file: TextIO | None = None
        self._writer = None
        if path:
            self._file = open(path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
            self._writer.writeheader()
        self.start: float | None = None
        self.rows: list[dict] = []
        self.warnings_total = 0
        self._warnings_window = 0
        self._first_warning: dict[int, float] = {}
        self._reached: set[int] = set()
        self._approached: set[int] = set()
        self.lead_times: list[float] = []

    def warning(self, track_id: int | None, now: float) -> None:
        """A warning about `track_id` was spoken at `now`."""
        self.warnings_total += 1
        self._warnings_window += 1
        if track_id is not None:
            self._first_warning.setdefault(track_id, now)

    def distances(self, items: Iterable[tuple[int, float | None]], danger_m: float, now: float) -> None:
        """Feed (track_id, distance); records lead time when a warned object reaches danger_m."""
        for track_id, dist in items:
            if dist is None or track_id in self._reached:
                continue
            if dist >= danger_m:
                self._approached.add(track_id)  # seen outside the danger distance first
                continue
            self._reached.add(track_id)
            first = self._first_warning.get(track_id)
            # Only objects that came from outside count: already-close ones were never "in advance".
            if track_id in self._approached and first is not None and first <= now:
                self.lead_times.append(now - first)

    def row(self, now: float, fps: float, stages: Mapping[str, float], depth_ms: float | None,
            reliability: float, confirmed: int, eco: bool) -> None:
        if self.start is None:
            self.start = now
        row = {
            "time_s": round(now - self.start, 2), "fps": round(fps, 1),
            "capture_ms": round(stages.get("capture", 0.0), 1), "detect_ms": round(stages.get("detect", 0.0), 1),
            "logic_ms": round(stages.get("logic", 0.0), 1), "total_ms": round(stages.get("total", 0.0), 1),
            "depth_ms": round(depth_ms, 1) if depth_ms is not None else "",
            "reliability": round(reliability, 2), "confirmed": confirmed,
            "warnings": self._warnings_window, "eco": int(eco),
        }
        self._warnings_window = 0
        self.rows.append(row)
        if self._writer is not None:
            self._writer.writerow(row)
            self._file.flush()  # type: ignore[union-attr]

    def summary(self) -> dict[str, float]:
        if not self.rows:
            return {}
        n = len(self.rows)
        minutes = max(n / 60.0, 1 / 60.0)
        out = {
            "seconds": float(n),
            "mean_fps": sum(r["fps"] for r in self.rows) / n,
            "min_fps": min(r["fps"] for r in self.rows),
            "mean_total_ms": sum(r["total_ms"] for r in self.rows) / n,
            "warnings_per_min": self.warnings_total / minutes,
        }
        if self.lead_times:
            out["mean_lead_time_s"] = sum(self.lead_times) / len(self.lead_times)
        return out

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
