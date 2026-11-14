"""Capture stage: camera thread that always hands out only the latest frame.

The camera is read in a background thread. Each new frame replaces the previous
one, so a slow consumer never works on stale images: old frames are dropped to
keep latency low. Sources: 0 (webcam), 1 (iPhone Continuity Camera) or a video
file path (played at its native frame rate).
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    """One captured frame."""

    image: np.ndarray  # BGR, owned by the consumer once returned
    index: int  # 0-based count of frames grabbed from the source
    timestamp: float  # time.perf_counter() when the frame was grabbed


class CameraStream:
    """Reads a camera or video file in a background thread; keeps the latest frame only.

    Usage:
        stream = CameraStream(0).start()
        frame = stream.read()      # newest frame not yet returned, or None
        stream.stop()

    Meant for a single consumer (the main loop).
    """

    def __init__(self, source: int | str, width: int = 1280, height: int = 720) -> None:
        self.source = source
        self.is_file = isinstance(source, str)
        if self.is_file:
            self._cap = cv2.VideoCapture(source)
        else:
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(source, backend)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source {source!r}")

        file_fps = self._cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0.0
        self._file_period = 1.0 / file_fps if file_fps and file_fps > 1.0 else 1.0 / 30.0

        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._last_returned = -1
        self._grabbed = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="CameraStream", daemon=True)

    # ------------------------------------------------------------------ control
    def start(self) -> "CameraStream":
        """Start the reader thread. Returns self so it can be chained."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the reader thread and release the device."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cap.release()

    @property
    def running(self) -> bool:
        """True while the reader thread is still producing frames."""
        return self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def frames_grabbed(self) -> int:
        """Total frames read from the source (dropped ones included)."""
        return self._grabbed

    # --------------------------------------------------------------------- read
    def read(self, timeout: float = 1.0) -> Frame | None:
        """Return the newest frame not returned before.

        Waits up to `timeout` seconds for one to arrive; returns None on timeout
        or when the source has ended. Never returns the same frame twice.
        """
        with self._cond:
            self._cond.wait_for(self._has_new_or_ended, timeout=timeout)
            frame = self._latest
            if frame is None or frame.index == self._last_returned:
                return None
            self._last_returned = frame.index
            return frame

    def _has_new_or_ended(self) -> bool:
        has_new = self._latest is not None and self._latest.index != self._last_returned
        return has_new or self._stop_event.is_set()

    # ------------------------------------------------------------ reader thread
    def _run(self) -> None:
        next_due = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                ok, image = self._cap.read()
                if not ok:
                    break  # end of file or camera unplugged
                frame = Frame(image=image, index=self._grabbed, timestamp=time.perf_counter())
                self._grabbed += 1
                with self._cond:
                    self._latest = frame  # replaces (drops) any frame not yet consumed
                    self._cond.notify_all()
                if self.is_file:  # a file would otherwise be read as fast as the disk allows
                    next_due += self._file_period
                    delay = next_due - time.perf_counter()
                    if delay > 0:
                        self._stop_event.wait(delay)
                    else:
                        next_due = time.perf_counter()
        finally:
            self._stop_event.set()
            with self._cond:
                self._cond.notify_all()


class VideoRecorder:
    """Writes raw camera frames to a video file in a background thread (for re-playable tests).

    write() never blocks: when the writer falls behind, frames are dropped and counted.
    """

    def __init__(self, path: str, fps: float = 30.0, max_queue: int = 30) -> None:
        import queue

        self.path = path
        self.fps = fps
        self.dropped = 0
        self.written = 0
        self._queue: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=max_queue)
        self._writer: cv2.VideoWriter | None = None
        self._thread = threading.Thread(target=self._run, name="VideoRecorder", daemon=True)
        self._thread.start()

    def write(self, image: np.ndarray) -> None:
        import queue

        try:
            self._queue.put_nowait(image.copy())
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while True:
            image = self._queue.get()
            if image is None:
                break
            if self._writer is None:
                h, w = image.shape[:2]
                self._writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self._writer.write(image)
            self.written += 1
        if self._writer is not None:
            self._writer.release()
