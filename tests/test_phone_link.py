"""The phone link: the page, frames in, speech / beeps / status out, over HTTP and HTTPS."""
from __future__ import annotations

import json
import ssl
import urllib.request

import cv2
import numpy as np
import pytest

import occhio.phone_link as pl
from occhio.phone_link import PhoneCamera, ensure_certificate


def jpeg(value: int = 100) -> bytes:
    return cv2.imencode(".jpg", np.full((540, 960, 3), value, np.uint8))[1].tobytes()


def post(url: str, body: bytes, ctx: ssl.SSLContext | None = None) -> dict:
    req = urllib.request.Request(url, data=body, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=5, context=ctx).read())


def test_frames_in_messages_out() -> None:
    phone = PhoneCamera(port=8797, use_https=False, strings={"start": "Start"}).start()
    try:
        page = urllib.request.urlopen("http://127.0.0.1:8797/", timeout=5).read().decode()
        assert "getUserMedia" in page and "speechSynthesis" in page and "createStereoPanner" in page
        reply = post("http://127.0.0.1:8797/frame?seq=0", jpeg())
        frame = phone.read(timeout=2)
        assert frame is not None and frame.image.shape == (540, 960, 3) and phone.frames_grabbed == 1
        assert reply["say"] == [] and reply["beep"] is None

        phone.say("Person ahead, 2 metres")
        phone.say("Collision in one second", urgent=True)
        phone.set_beep((2.0, -0.5, 660.0))
        phone.set_status("danger", "DANGER")
        reply = post("http://127.0.0.1:8797/frame?seq=0", jpeg(120))
        assert [m["text"] for m in reply["say"]] == ["Person ahead, 2 metres", "Collision in one second"]
        assert reply["say"][1]["urgent"] and reply["vibrate"] == 300
        assert reply["beep"] == {"rate": 2.0, "pan": -0.5, "freq": 660} and reply["label"] == "DANGER"
        again = post(f"http://127.0.0.1:8797/frame?seq={reply['seq']}", jpeg(130))
        assert again["say"] == [] and again["vibrate"] is None  # nothing is said twice
        assert phone.read(timeout=2).index == 2  # only the newest frame
    finally:
        phone.stop()


def test_bad_frame_is_rejected() -> None:
    phone = PhoneCamera(port=8796, use_https=False).start()
    try:
        req = urllib.request.Request("http://127.0.0.1:8796/frame", data=b"not a jpeg", method="POST")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req, timeout=5)
        assert phone.frames_grabbed == 0
    finally:
        phone.stop()


def test_https_with_self_signed_certificate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pl, "ensure_certificate", lambda ip: ensure_certificate(ip, tmp_path))
    phone = PhoneCamera(port=8795).start()
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE  # what "visit this website" does
        page = urllib.request.urlopen("https://127.0.0.1:8795/", timeout=5, context=ctx).read()
        assert b"Occhio" in page and phone.url.startswith("https://")
        post("https://127.0.0.1:8795/frame", jpeg(), ctx)
        assert phone.read(timeout=2) is not None
        assert (tmp_path / "cert.pem").exists()
        first = (tmp_path / "cert.pem").read_bytes()
        ensure_certificate(phone.ip, tmp_path)
        assert (tmp_path / "cert.pem").read_bytes() == first  # made once, then reused
    finally:
        phone.stop()


def test_stop_releases_a_waiting_read() -> None:
    import threading
    phone = PhoneCamera(port=8794, use_https=False).start()
    out = []
    t = threading.Thread(target=lambda: out.append(phone.read(timeout=5)))
    t.start()
    phone.stop()
    t.join(2)
    assert out == [None] and not phone.running
