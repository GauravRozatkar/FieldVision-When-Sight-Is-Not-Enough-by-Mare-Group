"""Sound awareness: warn about what the camera cannot see (behind, around corners).

A blind pedestrian also relies on hearing; Occhio listens with them. The
microphone is analysed in 100 ms blocks (numpy FFT, no extra dependencies):

- horn / bike bell: a sudden, loud, *steady* tone in 250-1200 Hz (same pitch for
  0.3 s: a voice is tonal too, but its pitch wobbles);
- siren: a loud tonal sound whose pitch sweeps (500-1800 Hz) over ~1.5 s;
- approaching: the level rising *steadily* (~8 dB in 2 s, straight-line fit
  R^2 >= 0.7), like a vehicle coming from behind; a sudden step is not "approaching".

Direction: the MacBook microphone is mono, so it can say *that* something is
there, not where. With a stereo microphone (2 input channels) the level
difference between channels gives left / right. Front / back needs more mics.

While Occhio itself beeps or speaks (`is_self_audio()`), the microphone is
ignored, so our own 880 Hz beeps are never mistaken for a horn.

These are heuristics, tuned on synthetic sounds; real street audio must be
checked at the venue. A trained audio classifier (e.g. an AudioSet model) would
be the next step.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable

import numpy as np

from config.settings import SOUND


def level_db(block: np.ndarray) -> float:
    """RMS level in dBFS (0 = full scale)."""
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))) if block.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-9))


def dominant_tone(block: np.ndarray, sr: int, lo: float = 250.0, hi: float = 3500.0) -> tuple[float, float]:
    """(frequency of the strongest 50 Hz band in lo..hi, its share of the energy in lo..hi)."""
    spec = np.abs(np.fft.rfft(block * np.hanning(len(block)))) ** 2
    freqs = np.fft.rfftfreq(len(block), 1.0 / sr)
    band = (freqs >= lo) & (freqs <= hi)
    total = float(spec[band].sum())
    if total <= 0:
        return 0.0, 0.0
    f, e = freqs[band], spec[band]
    peak = int(np.argmax(e))
    near = np.abs(f - f[peak]) <= 25.0
    return float(f[peak]), float(e[near].sum() / total)


class SoundAnalyzer:
    """Feed it audio blocks; it returns at most one event per block: 'horn' | 'siren' | 'approach'."""

    def __init__(self, samplerate: int = SOUND["samplerate"], settings: dict | None = None) -> None:
        self.cfg = {**SOUND, **(settings or {})}
        self.sr = samplerate
        block_s = float(self.cfg["block_s"])
        self._levels: deque[float] = deque(maxlen=max(2, int(self.cfg["approach_window_s"] / block_s)))
        self._background: deque[float] = deque(maxlen=max(5, int(3.0 / block_s)))
        self._tones: deque[tuple[float, float]] = deque(maxlen=max(3, int(self.cfg["siren_window_s"] / block_s)))
        self._last: dict[str, float] = {}
        self.time = 0.0
        self.db = -120.0  # latest level, for the overlay
        self.side = ""  # "left" | "right" | "" (unknown / mono)
        self._horn_freq: float | None = None
        self._horn_run = 0

    def feed(self, block: np.ndarray) -> str | None:
        """block: (n,) mono or (n, 2) stereo float32 in -1..1."""
        self.time += len(block) / self.sr
        if block.ndim == 2 and block.shape[1] >= 2:
            l_db, r_db = level_db(block[:, 0]), level_db(block[:, 1])
            diff = l_db - r_db
            self.side = "left" if diff >= self.cfg["side_db"] else "right" if -diff >= self.cfg["side_db"] else ""
            mono = block[:, :2].mean(axis=1)
        else:
            self.side = ""
            mono = block.reshape(-1)
        db = level_db(mono)
        self.db = db
        background = float(np.median(self._background)) if self._background else db
        freq, share = dominant_tone(mono, self.sr)
        loud = db >= self.cfg["loud_db"]
        tonal = share >= self.cfg["tonal_ratio"]
        self._tones.append((freq if (loud and tonal) else 0.0, db))
        self._levels.append(db)

        hlo, hhi = self.cfg["horn_band"]
        horn_like = (db >= self.cfg["horn_db"] and share >= self.cfg["horn_tonal_ratio"] and hlo <= freq <= hhi
                     and db - background >= self.cfg["horn_jump_db"])
        steady = self._horn_freq is not None and abs(freq - self._horn_freq) <= 0.06 * self._horn_freq
        self._horn_run = (self._horn_run + 1 if steady else 1) if horn_like else 0
        self._horn_freq = freq if horn_like else None

        event: str | None = None
        slo, shi = self.cfg["siren_band"]
        siren_tones = [f for f, _ in self._tones if slo <= f <= shi]
        if len(siren_tones) >= int(0.8 * self._tones.maxlen) and \
                max(siren_tones) - min(siren_tones) >= self.cfg["siren_sweep_hz"]:
            event = "siren"
        elif self._horn_run >= int(self.cfg["horn_steady_blocks"]):
            event = "horn"
        elif len(self._levels) == self._levels.maxlen and loud and self._horn_run == 0:
            # Approaching = a *steady* rise (vehicle getting closer), not a sudden step (horn, door).
            levels = np.array(self._levels)
            x = np.arange(len(levels), dtype=np.float64)
            slope, intercept = np.polyfit(x, levels, 1)
            fit = slope * x + intercept
            ss_tot = float(np.sum((levels - levels.mean()) ** 2))
            r2 = 1.0 - float(np.sum((levels - fit) ** 2)) / ss_tot if ss_tot > 0 else 0.0
            if slope * (len(levels) - 1) >= self.cfg["approach_rise_db"] and r2 >= 0.7:
                event = "approach"
        if not (loud and tonal) and self._horn_run == 0:
            self._background.append(db)  # learn the background from ordinary sound only
        if self.time < float(self.cfg["warmup_s"]):
            return None  # still learning what this room sounds like
        if event is not None:
            last = self._last.get(event)
            if last is not None and self.time - last < self.cfg["cooldown_s"]:
                return None
            self._last[event] = self.time
        return event


class SoundMonitor:
    """Microphone -> SoundAnalyzer on sounddevice's audio thread; events go to on_event(kind, side)."""

    def __init__(self, on_event: Callable[[str, str], None], enabled: bool = True,
                 settings: dict | None = None, stream_factory: Callable[..., Any] | None = None,
                 is_self_audio: Callable[[], bool] | None = None) -> None:
        self.cfg = {**SOUND, **(settings or {})}
        self.enabled = enabled and bool(self.cfg["enabled"])
        self.sr = int(self.cfg["samplerate"])
        self.analyzer = SoundAnalyzer(self.sr, self.cfg)
        self._on_event = on_event
        self._is_self_audio = is_self_audio or (lambda: False)
        self._factory = stream_factory
        self._stream: Any = None
        self.paused = threading.Event()  # set while a voice command records
        self.channels = 0

    def start(self) -> "SoundMonitor":
        if not self.enabled:
            return self
        try:
            import sounddevice as sd

            factory = self._factory or sd.InputStream
            if self._factory is None:
                self.channels = min(2, int(sd.query_devices(kind="input")["max_input_channels"]))
            else:
                self.channels = 1
            self._stream = factory(samplerate=self.sr, channels=max(1, self.channels), dtype="float32",
                                   blocksize=int(self.sr * float(self.cfg["block_s"])), callback=self._callback)
            self._stream.start()
        except Exception as exc:
            print(f"[occhio] sound awareness disabled: {exc!r}", flush=True)
            self.enabled = False
            self._stream = None
        return self

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    @property
    def stereo(self) -> bool:
        return self.channels >= 2

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if self.paused.is_set() or self._is_self_audio():
            return
        block = indata.copy() if indata.shape[1] >= 2 else indata[:, 0].copy()
        event = self.analyzer.feed(block)
        if event is not None:
            self._on_event(event, self.analyzer.side)
