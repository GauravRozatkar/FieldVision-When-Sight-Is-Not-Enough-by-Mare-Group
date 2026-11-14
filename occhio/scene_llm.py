"""Describe-scene (key d) and read-text (key t) with a vision LLM, in a background thread.

Like Seeing AI's "channels", the same frame can be asked two ways: `describe`
(obstacles, people, crossings) or `read` (signs, bus numbers). On a key the current frame is JPEG-encoded (longest side <= 768 px) and sent
to the Anthropic API with the profile's describe prompt; the one-sentence answer
comes back through `on_result` (spoken and shown on screen by main.py). While a
request is running new ones are ignored. Any failure (no API key, no network,
refusal, timeout) returns the profile's "describe_failed" / "read_failed" line.

Model: env OCCHIO_LLM_MODEL, default claude-opus-5-5 at low effort. Credentials:
ANTHROPIC_API_KEY (e.g. in .env, loaded by main.py). For models that support it,
the request opts into the server-side refusal fallback ("fallbacks": "default").
"""
from __future__ import annotations

import base64
import os
import threading
import time
from typing import Callable

import cv2
import numpy as np

from config.profiles import Profile, lines_for
from config.settings import SCENE_LLM

LlmCall = Callable[[bytes, str], str]  # (jpeg bytes, prompt) -> one sentence; raises on failure

FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class DescribeError(RuntimeError):
    """The model gave no usable answer (refusal, empty)."""


def encode_jpeg(image: np.ndarray, max_px: int = SCENE_LLM["max_image_px"],
                quality: int = SCENE_LLM["jpeg_quality"]) -> bytes:
    """BGR frame -> JPEG bytes, longest side at most max_px."""
    h, w = image.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise DescribeError("JPEG encoding failed")
    return buf.tobytes()


def anthropic_call(settings: dict | None = None) -> LlmCall:
    """The real call: Claude vision via the official SDK (client created lazily, once)."""
    cfg = {**SCENE_LLM, **(settings or {})}
    model = os.environ.get(cfg["model_env"]) or cfg["default_model"]
    state: dict = {}

    def call(jpeg: bytes, prompt: str) -> str:
        import anthropic

        if "client" not in state:
            state["client"] = anthropic.Anthropic(timeout=float(cfg["timeout_s"]), max_retries=1)
        client = state["client"]
        extra: dict = {}
        if model in FALLBACK_MODELS:
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = client.beta.messages.create(
            model=model,
            max_tokens=int(cfg["max_tokens"]),
            output_config={"effort": cfg["effort"]},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.standard_b64encode(jpeg).decode("ascii")}},
                    {"type": "text", "text": prompt},
                ],
            }],
            **extra,
        )
        if response.stop_reason == "refusal":
            raise DescribeError("refusal")
        text = " ".join(b.text.strip() for b in response.content if b.type == "text").strip()
        if not text:
            raise DescribeError(f"no text (stop_reason={response.stop_reason})")
        return text

    call.model = model  # type: ignore[attr-defined]
    return call


def describe_error(exc: BaseException) -> str:
    """Short reason for the console; never includes credentials."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return type(exc).__name__
    if isinstance(exc, anthropic.AuthenticationError):
        return "authentication failed (check ANTHROPIC_API_KEY)"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate limited"
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "no connection"
    if isinstance(exc, DescribeError):
        return str(exc)
    if "api_key" in str(exc).lower() or "auth" in str(exc).lower():
        return "no API key: put ANTHROPIC_API_KEY=... in .env"
    return type(exc).__name__


class SceneDescriber:
    """Runs one describe request at a time in a background thread."""

    def __init__(self, profile: Profile, lang: str, on_result: Callable[[str, bool], None],
                 call: LlmCall | None = None,
                 prompt_for: Callable[[str, str], str] | None = None) -> None:
        self.profile = profile
        self._prompt_for = prompt_for  # (kind, lang) -> prompt; default: the profile's prompts
        self.lang = lang
        self._on_result = on_result
        self._call = call or anthropic_call()
        self._busy = threading.Event()
        self._kind = "describe"
        self.last_error: str | None = None
        self.last_latency_s: float | None = None

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def set_profile(self, profile: Profile, lang: str | None = None) -> None:
        self.profile = profile
        self.lang = lang or self.lang

    def request(self, image: np.ndarray, kind: str = "describe") -> bool:
        """Start a `kind` ('describe' or 'read') request; False (ignored) if one is still running."""
        if self._busy.is_set():
            return False
        self._busy.set()
        try:
            jpeg = encode_jpeg(image)  # ~5 ms, done here so the frame can be drawn on afterwards
        except Exception as exc:
            self._busy.clear()
            self._kind = kind
            self._finish_failed(exc)
            return True
        self._kind = kind
        prompt = (self._prompt_for(kind, self.lang) if self._prompt_for
                  else self.profile[f"{kind}_prompt"][self.lang])
        threading.Thread(target=self._run, args=(jpeg, prompt), name="SceneDescriber", daemon=True).start()
        return True

    def _run(self, jpeg: bytes, prompt: str) -> None:
        t0 = time.perf_counter()
        try:
            text = self._call(jpeg, prompt)
        except Exception as exc:
            self.last_latency_s = time.perf_counter() - t0
            self._busy.clear()
            self._finish_failed(exc)
            return
        self.last_latency_s = time.perf_counter() - t0
        self.last_error = None
        self._busy.clear()
        self._on_result(text, True)

    def _finish_failed(self, exc: BaseException) -> None:
        self.last_error = describe_error(exc)
        self._on_result(lines_for(self.profile, self.lang)[f"{self._kind}_failed"], False)
