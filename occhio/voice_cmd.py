"""Voice commands (F23): push-to-talk 'v', then e.g. "Occhio, cosa c'è intorno?".

Pressing v says "Ti ascolto", records `record_s` seconds from the microphone,
transcribes with faster-whisper (offline, CPU int8) and maps the words to an
intent: describe, read, street / industrial mode, repeat. Everything runs in a
background thread; the result comes back through `on_command(intent, text)`.

The whisper model is loaded from the local cache only (no surprise downloads at
the venue). If it is missing, on_command("unavailable", reason) is called.
"""
from __future__ import annotations

import threading
import time
import unicodedata
from typing import Callable

import numpy as np

from config.settings import VOICE

Recorder = Callable[[float, int], np.ndarray]  # (seconds, samplerate) -> mono float32
Transcriber = Callable[[np.ndarray, str], str]  # (audio, lang) -> text

INTENTS: list[tuple[str, tuple[str, ...]]] = [  # first match wins; accent-free, lower case
    ("read", ("leggi", "scritto", "read", "text", "testo", "autobus", "cartello")),
    ("mode_industrial", ("industrial", "fabbrica", "hangar")),
    ("mode_street", ("strada", "street")),
    ("repeat", ("ripeti", "repeat", "again", "ancora")),
    ("describe", ("intorno", "cosa c", "descrivi", "davanti", "around", "describe", "what", "vedi", "scena")),
]


def normalise(text: str) -> str:
    """Lower case, no accents, single spaces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(text.replace("'", " ' ").split())


def parse_intent(text: str) -> str | None:
    """'Occhio, cosa c'è intorno?' -> 'describe'. None when nothing matches."""
    t = normalise(text).replace(" ' ", "'").replace(" '", "'")
    for intent, words in INTENTS:
        if any(w in t for w in words):
            return intent
    return None


def sounddevice_recorder(seconds: float, samplerate: int) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def whisper_transcriber(settings: dict | None = None) -> Transcriber:
    cfg = {**VOICE, **(settings or {})}
    state: dict = {}

    def transcribe(audio: np.ndarray, lang: str) -> str:
        if "model" not in state:
            from faster_whisper import WhisperModel

            state["model"] = WhisperModel(cfg["model"], device="cpu", compute_type=cfg["compute_type"],
                                          local_files_only=True)
        segments, _ = state["model"].transcribe(audio, language=lang, vad_filter=True, beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    return transcribe


class VoiceCommander:
    def __init__(self, lang: str, on_command: Callable[[str, str], None],
                 say: Callable[[str], None] | None = None, listening_line: str = "",
                 settings: dict | None = None, recorder: Recorder | None = None,
                 transcriber: Transcriber | None = None) -> None:
        cfg = {**VOICE, **(settings or {})}
        self.lang = lang
        self.record_s = float(cfg["record_s"])
        self.lead_in_s = float(cfg["lead_in_s"])
        self.samplerate = int(cfg["samplerate"])
        self._on_command = on_command
        self._say = say
        self.listening_line = listening_line
        self._recorder = recorder or sounddevice_recorder
        self._transcriber = transcriber or whisper_transcriber(cfg)
        self._busy = threading.Event()
        self.state = "idle"  # idle | listening | thinking (for the overlay)

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def press(self) -> bool:
        """Start one push-to-talk command; False if one is already running."""
        if self._busy.is_set():
            return False
        self._busy.set()
        threading.Thread(target=self._run, name="VoiceCommander", daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            if self._say and self.listening_line:
                self._say(self.listening_line)
                time.sleep(self.lead_in_s)  # do not record our own voice
            self.state = "listening"
            audio = self._recorder(self.record_s, self.samplerate)
            self.state = "thinking"
            text = self._transcriber(audio, self.lang)
        except Exception as exc:
            self._finish("unavailable", f"{type(exc).__name__}: {exc}"[:160])
            return
        self._finish(parse_intent(text) or "unknown", text)

    def _finish(self, intent: str, text: str) -> None:
        self.state = "idle"
        self._busy.clear()
        self._on_command(intent, text)
