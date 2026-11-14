"""Offline vision-language model (SmolVLM via transformers) for describe / read without an API key.

Same interface as the Claude call: (jpeg bytes, prompt) -> one sentence. The model
is loaded once in a background thread at start-up (from the local Hugging Face
cache only), and generation runs on the describe thread, never the main loop.
Small model = modest quality: fine for "a hallway with a door on the right" and
big clear signs, weak on small text.
"""
from __future__ import annotations

import io
import threading
from typing import Any

from config.settings import LOCAL_VLM

LOCAL_PROMPTS: dict[str, dict[str, str]] = {  # small models need short, plain prompts
    "describe": {
        "en": "Describe this scene in one short sentence for a blind person walking. "
              "Mention people, vehicles, doors, stairs and obstacles.",
        "it": "Describe this scene in one short sentence for a blind person walking. "
              "Mention people, vehicles, doors, stairs and obstacles. Answer in Italian.",
    },
    "read": {
        "en": "What text is written in this image, such as signs and bus numbers? Answer with the text only.",
        "it": "What text is written in this image, such as signs and bus numbers? Answer with the text only.",
    },
}


# Small models invent text when asked to read an image without any, so reading is two
# questions: first "is there text?" (yes/no), and only on "yes" "what does it say?".
READ_CHECK_PROMPT = "Is there any written text in this image? Answer yes or no."
NO_TEXT = "No text."


def is_cached(model: str = LOCAL_VLM["model"]) -> bool:
    """Is the model already downloaded? (No network.)"""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model, "config.json"), str)
    except Exception:
        return False


class LocalVLM:
    """Callable like the Claude backend; loads itself in the background."""

    def __init__(self, settings: dict | None = None) -> None:
        self.cfg = {**LOCAL_VLM, **(settings or {})}
        self.model_name: str = self.cfg["model"]
        self.model = self.model_name  # shown in the start-up line, like the Claude backend
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self.error: str | None = None
        self._processor: Any = None
        self._model: Any = None
        self.device = "cpu"

    def start_loading(self) -> "LocalVLM":
        threading.Thread(target=self._load, name="LocalVLM-load", daemon=True).start()
        return self

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self.error is None

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            from occhio.detector import pick_device

            self.device = pick_device(self.cfg["device"])
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self._processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, dtype=dtype, local_files_only=True).to(self.device).eval()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"[:200]
        finally:
            self._ready.set()

    def __call__(self, jpeg: bytes, prompt: str) -> str:
        if not self._ready.wait(float(self.cfg["load_timeout_s"])):
            raise RuntimeError("local model still loading")
        if self.error:
            raise RuntimeError(f"local model unavailable: {self.error}")
        from PIL import Image

        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if prompt in LOCAL_PROMPTS["read"].values():
            if self._generate(image, READ_CHECK_PROMPT).lower().startswith("no"):
                return NO_TEXT
        return self._generate(image, prompt)

    def _generate(self, image: Any, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        with self._lock:  # one generation at a time
            text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(text=text, images=[image], return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=int(self.cfg["max_new_tokens"]),
                                           do_sample=False)
            answer = self._processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                                  skip_special_tokens=True)[0]
        answer = " ".join(answer.split()).strip()
        if not answer:
            raise RuntimeError("empty answer")
        return answer


def choose_backend(preference: str, has_api_key: bool, local_cached: bool) -> str:
    """'claude' | 'local' | 'none'. auto = Claude if there is a key, else the local model if downloaded."""
    if preference == "claude":
        return "claude"
    if preference == "local":
        return "local" if local_cached else "none"
    if preference == "none":
        return "none"
    if has_api_key:
        return "claude"
    return "local" if local_cached else "none"
