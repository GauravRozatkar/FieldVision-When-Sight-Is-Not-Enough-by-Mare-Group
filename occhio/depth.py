"""Depth stage: Depth Anything V2 Small, run every Nth frame in a background thread.

DepthEstimator.submit() is called on every frame from the main loop and never
blocks: it downsizes every Nth frame and hands it to the worker, unless the
worker is still busy (then the frame is skipped). `latest` is the most recent
result: a relative inverse-depth map (bigger = closer). Turning it into metres
is spatial.py's job (calibration constant / box-size fallback).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from config.settings import DEPTH

InferFn = Callable[[np.ndarray], np.ndarray]  # BGR image (h, w, 3) -> disparity (h, w)


@dataclass(frozen=True)
class DepthFrame:
    """One depth result."""

    disparity: np.ndarray  # float32, model resolution, relative inverse depth (bigger = closer)
    frame_shape: tuple[int, int]  # (h, w) of the camera frame it came from
    timestamp: float  # camera timestamp of that frame
    infer_ms: float

    def median_in_box(self, box: tuple[float, float, float, float], inner: float = 0.5) -> float | None:
        """Median disparity in the central `inner` fraction of a box given in frame pixels."""
        fh, fw = self.frame_shape
        mh, mw = self.disparity.shape
        x1, y1, x2, y2 = box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner, (y2 - y1) * inner
        sx, sy = mw / fw, mh / fh
        ax1, ax2 = int(max(0, (cx - w / 2) * sx)), int(min(mw, (cx + w / 2) * sx))
        ay1, ay2 = int(max(0, (cy - h / 2) * sy)), int(min(mh, (cy + h / 2) * sy))
        if ax2 - ax1 < 1 or ay2 - ay1 < 1:
            return None
        value = float(np.median(self.disparity[ay1:ay2, ax1:ax2]))
        return value if value > 0 else None


def load_depth_anything(settings: dict | None = None) -> InferFn:
    """Build the real Depth Anything V2 Small inference function (slow import, do it once)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    from occhio.detector import pick_device

    cfg = {**DEPTH, **(settings or {})}
    device = pick_device(cfg["device"])
    processor = AutoImageProcessor.from_pretrained(cfg["model"])
    model = AutoModelForDepthEstimation.from_pretrained(cfg["model"]).to(device).eval()
    w, h = cfg["input_size"]

    def infer(image_bgr: np.ndarray) -> np.ndarray:
        inputs = processor(images=Image.fromarray(image_bgr[..., ::-1]), return_tensors="pt",
                           size={"height": h, "width": w}, keep_aspect_ratio=False).to(device)
        with torch.no_grad():
            out = model(**inputs).predicted_depth
        return out[0].float().cpu().numpy()

    return infer


class DepthEstimator:
    """Runs `infer` on every Nth submitted frame in a background thread."""

    def __init__(self, infer: InferFn, settings: dict | None = None) -> None:
        cfg = {**DEPTH, **(settings or {})}
        self.every_n = max(1, int(cfg["every_n"]))
        self.input_size: tuple[int, int] = tuple(cfg["input_size"])  # type: ignore[assignment]
        self._infer = infer
        self._cond = threading.Condition()
        self._pending: tuple[np.ndarray, tuple[int, int], float] | None = None
        self._busy = False
        self._latest: DepthFrame | None = None
        self._count = 0
        self.runs = 0
        self.skipped_busy = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="DepthEstimator", daemon=True)

    def start(self) -> "DepthEstimator":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def latest(self) -> DepthFrame | None:
        return self._latest

    def submit(self, image: np.ndarray, timestamp: float) -> bool:
        """Offer a frame; returns True if it was handed to the worker. Never blocks."""
        self._count += 1
        if (self._count - 1) % self.every_n:
            return False
        with self._cond:
            if self._busy or self._pending is not None:
                self.skipped_busy += 1
                return False
            small = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)  # also a copy; AREA ~5 ms
            self._pending = (small, image.shape[:2], timestamp)
            self._cond.notify()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._pending is not None or self._stop.is_set())
                if self._stop.is_set():
                    return
                job, self._pending = self._pending, None
                self._busy = True
            small, frame_shape, timestamp = job  # type: ignore[misc]
            try:
                t0 = time.perf_counter()
                disparity = np.asarray(self._infer(small), dtype=np.float32)
                self._latest = DepthFrame(disparity, frame_shape, timestamp,
                                          (time.perf_counter() - t0) * 1000.0)
                self.runs += 1
            except Exception as exc:  # never kill the thread: distance falls back to box size
                self.errors += 1
                if self.errors <= 3:
                    print(f"[occhio] depth error: {exc!r}", flush=True)
            finally:
                with self._cond:
                    self._busy = False
