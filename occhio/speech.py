"""Audio output: spoken lines, never blocking the main loop.

Speaker keeps a tiny queue and a worker thread that runs macOS `say` (English
voice Samantha by default, Italian voice Alice with --lang it). say() returns at once:
- normal messages are queued; when the queue is full the oldest is dropped,
  because only the latest situation matters;
- urgent messages clear the queue and cut a normal sentence being spoken; they
  never cut another urgent sentence (it finishes, then the new one plays), so
  a busy scene cannot turn speech into fragments.
With enabled=False nothing is spoken; messages are still reported to `on_spoken`.

Beeper plays parking-sensor beeps through sounddevice: a short 880 Hz tone,
panned left/right by where the target is (equal-power stereo), repeated from
1 Hz at the warn distance up to 6 Hz at the danger distance. The audio callback
runs on sounddevice's own thread; the main loop only calls set_target().
"""
from __future__ import annotations

import math
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Protocol

import numpy as np

from config.profiles import VOICES
from config.settings import BEEPS, SPEECH


class Process(Protocol):
    def wait(self) -> int: ...
    def terminate(self) -> None: ...
    def poll(self) -> int | None: ...


Runner = Callable[[str], Process]


def say_runner(voice: str, rate_wpm: int) -> Runner:
    """Runner that starts `say -v voice -r rate text`."""
    def run(text: str) -> Process:
        return subprocess.Popen(["say", "-v", voice, "-r", str(rate_wpm), text],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return run


class Speaker:
    """Non-blocking speech queue with urgent interruption."""

    def __init__(self, lang: str = "en", enabled: bool = True, settings: dict | None = None,
                 runner: Runner | None = None,
                 on_spoken: Callable[[str, bool], None] | None = None) -> None:
        cfg = {**SPEECH, **(settings or {})}
        self.enabled = enabled
        self.max_queue = int(cfg["max_queue"])
        self._runner = runner or say_runner(VOICES[lang], int(cfg["rate_wpm"]))
        self._on_spoken = on_spoken
        self._queue: deque[str] = deque()
        self._cond = threading.Condition()
        self._current: Process | None = None
        self._current_urgent = False
        self._urgent_pending = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Speaker", daemon=True)
        self.spoken: list[str] = []  # everything started, in order (for tests and the event log)

    def start(self) -> "Speaker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._queue.clear()
            self._kill_current()
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_language(self, lang: str, rate_wpm: int | None = None) -> None:
        self._runner = say_runner(VOICES[lang], rate_wpm or int(SPEECH["rate_wpm"]))

    @property
    def busy(self) -> bool:
        with self._cond:
            return bool(self._queue) or (self._current is not None and self._current.poll() is None)

    def say(self, text: str, urgent: bool = False) -> None:
        """Queue a sentence; urgent ones interrupt. Returns immediately."""
        if self._on_spoken is not None:
            self._on_spoken(text, urgent)
        if not self.enabled:
            return
        with self._cond:
            if urgent:
                self._queue.clear()
                playing = self._current is not None and self._current.poll() is None
                if not (playing and self._current_urgent):
                    self._kill_current()
                self._queue.append(text)
                self._urgent_pending = True
            else:
                if self._urgent_pending:
                    return  # an urgent sentence is waiting: drop the less important one
                self._queue.append(text)
                while len(self._queue) > self.max_queue:
                    self._queue.popleft()
            self._cond.notify()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """For tests and shutdown: wait until nothing is queued or playing."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if not self.busy:
                return True
            time.sleep(0.01)
        return False

    def _kill_current(self) -> None:  # caller holds the lock
        if self._current is not None and self._current.poll() is None:
            self._current.terminate()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: bool(self._queue) or self._stop.is_set())
                if self._stop.is_set():
                    return
                text = self._queue.popleft()
                self._current_urgent, self._urgent_pending = self._urgent_pending, False
                try:
                    self._current = self._runner(text)
                except OSError as exc:  # `say` missing: keep running silently
                    print(f"[occhio] speech error: {exc!r}", flush=True)
                    self._current = None
                    continue
                self.spoken.append(text)
                process = self._current
            process.wait()  # outside the lock, so urgent messages can kill it


# --------------------------------------------------------------------- beeps
def beep_rate(distance_m: float | None, warn_m: float, danger_m: float,
              min_hz: float = BEEPS["min_hz"], max_hz: float = BEEPS["max_hz"]) -> float | None:
    """Beeps per second: None beyond warn distance, min_hz at warn, max_hz at danger and closer."""
    if distance_m is None or distance_m > warn_m:
        return None
    if distance_m <= danger_m or warn_m <= danger_m:
        return max_hz
    frac = (warn_m - distance_m) / (warn_m - danger_m)
    return min_hz + frac * (max_hz - min_hz)


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power stereo gains for pan in [-1 (left), 1 (right)]."""
    angle = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4
    return math.cos(angle), math.sin(angle)


class Beeper:
    """Parking-sensor beeps; silent while the target is None."""

    def __init__(self, enabled: bool = True, settings: dict | None = None,
                 stream_factory: Callable[..., Any] | None = None) -> None:
        cfg = {**BEEPS, **(settings or {})}
        self.enabled = enabled and bool(cfg["enabled"])
        self.sr = int(cfg["samplerate"])
        self.freq = float(cfg["freq_hz"])
        self.volume = float(cfg["volume"])
        self.beep_len = max(1, int(self.sr * float(cfg["beep_ms"]) / 1000))
        fade = max(1, int(self.sr * float(cfg["fade_ms"]) / 1000))
        env = np.ones(self.beep_len, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, min(fade, self.beep_len // 2), dtype=np.float32)
        env[:len(ramp)] = ramp
        env[self.beep_len - len(ramp):] = ramp[::-1]
        t = np.arange(self.beep_len, dtype=np.float32) / self.sr
        self._tone = (np.sin(2 * np.pi * self.freq * t) * env * self.volume).astype(np.float32)
        self._factory = stream_factory
        self._stream: Any = None
        self._target: tuple[float, float, float] | None = None  # (rate_hz, pan, freq); replaced atomically
        self._tones: dict[float, np.ndarray] = {self.freq: self._tone}
        self._env, self._t = env, t
        self._beep_pos = -1  # samples into the current beep, -1 = not beeping
        self._gap = 0  # samples of silence left before the next beep
        self._gains = (0.7071, 0.7071)
        self._playing = self._tone

    def start(self) -> "Beeper":
        if not self.enabled:
            return self
        try:
            if self._factory is None:
                import sounddevice as sd
                self._factory = sd.OutputStream
            self._stream = self._factory(samplerate=self.sr, channels=2, dtype="float32",
                                         callback=self._callback)
            self._stream.start()
        except Exception as exc:  # no audio device: speech still works
            print(f"[occhio] beeps disabled: {exc!r}", flush=True)
            self.enabled = False
            self._stream = None
        return self

    def stop(self) -> None:
        self._target = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def set_target(self, rate_hz: float | None, pan: float = 0.0, freq_hz: float | None = None) -> None:
        """Beep `rate_hz` times per second panned to `pan` (-1 left .. 1 right) at `freq_hz`; None = silence."""
        self._target = None if rate_hz is None or rate_hz <= 0 else (float(rate_hz), float(pan),
                                                                     float(freq_hz or self.freq))

    def _tone_for(self, freq: float) -> np.ndarray:
        tone = self._tones.get(freq)
        if tone is None:
            tone = (np.sin(2 * np.pi * freq * self._t) * self._env * self.volume).astype(np.float32)
            self._tones[freq] = tone
        return tone

    @property
    def target(self) -> tuple[float, float, float] | None:
        return self._target

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        outdata[:] = self.render(frames)

    def render(self, frames: int) -> np.ndarray:
        """Next `frames` stereo samples (frames x 2, float32)."""
        out = np.zeros((frames, 2), dtype=np.float32)
        target = self._target
        if target is not None:  # a faster rate takes effect at once, not after the old gap
            self._gap = min(self._gap, max(0, int(self.sr / target[0]) - self.beep_len))
        i = 0
        while i < frames:
            if self._beep_pos >= 0:
                n = min(frames - i, self.beep_len - self._beep_pos)
                chunk = self._playing[self._beep_pos:self._beep_pos + n]
                out[i:i + n, 0] = chunk * self._gains[0]
                out[i:i + n, 1] = chunk * self._gains[1]
                self._beep_pos += n
                i += n
                if self._beep_pos >= self.beep_len:
                    self._beep_pos = -1
                continue
            if target is None:
                self._gap = 0  # the next target beeps straight away
                break
            if self._gap <= 0:
                rate, pan, freq = target
                self._gains = pan_gains(pan)
                self._playing = self._tone_for(freq)
                self._beep_pos = 0
                self._gap = max(0, int(self.sr / rate) - self.beep_len)
                continue
            n = min(frames - i, self._gap)
            self._gap -= n
            i += n
        return out
