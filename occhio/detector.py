"""Detection stage: YOLO11n + ByteTrack wrapper (device=mps).

A Detector runs model.track() on a frame (persist=True, so ByteTrack keeps IDs
across calls) and returns Detection dataclasses filtered by the active
profile's classes. Detections without a track ID are dropped: the rest of the
pipeline reasons about tracks only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from ultralytics import YOLO

from config.settings import DETECTOR

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Detection:
    """One tracked detection in one frame."""

    track_id: int
    cls_name: str
    conf: float
    xyxy: BBox


def pick_device(wanted: str) -> str:
    """Return `wanted` if usable, else 'cpu'."""
    if wanted == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return wanted


class Detector:
    """YOLO + ByteTrack, filtered to a list of class names."""

    def __init__(self, classes: Sequence[str], settings: dict | None = None) -> None:
        cfg = {**DETECTOR, **(settings or {})}
        self.device = pick_device(cfg["device"])
        self.imgsz = int(cfg["imgsz"])
        self.tracker = cfg["tracker"]
        self.model = YOLO(cfg["model"])
        self._names: dict[int, str] = dict(self.model.names)
        self.class_ids: list[int] = []
        self.class_names: frozenset[str] = frozenset()
        self.set_classes(classes)
        self._warm_up()

    def set_classes(self, classes: Sequence[str]) -> None:
        """Change the classes we report (used when switching profile)."""
        name_to_id = {name: i for i, name in self._names.items()}
        unknown = [c for c in classes if c not in name_to_id]
        if unknown:
            raise ValueError(f"Classes not in the model: {unknown}")
        self.class_ids = sorted(name_to_id[c] for c in classes)
        self.class_names = frozenset(classes)

    def reset(self) -> None:
        """Forget all ByteTrack state (new video, new scene)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect and track objects in one BGR frame."""
        results = self.model.track(
            image,
            persist=True,
            tracker=self.tracker,
            device=self.device,
            imgsz=self.imgsz,
            classes=self.class_ids,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().tolist()
        cls_ids = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
        xyxys = boxes.xyxy.tolist()
        detections: list[Detection] = []
        for track_id, cls_id, conf, xyxy in zip(ids, cls_ids, confs, xyxys):
            name = self._names[cls_id]
            if name in self.class_names:
                detections.append(Detection(track_id, name, float(conf), tuple(xyxy)))  # type: ignore[arg-type]
        return detections

    def _warm_up(self) -> None:
        """First MPS inference is slow; do it before the camera starts."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detect(dummy)  # goes through track() so the tracker is set up too
        self.reset()
