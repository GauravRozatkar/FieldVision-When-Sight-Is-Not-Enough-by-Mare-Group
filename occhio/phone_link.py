"""iPhone (or any phone) as camera, voice and beeper, through Safari: no app to install.

    python main.py --source phone        # prints https://<mac-ip>:8443 to open on the phone

The Mac serves one web page over HTTPS (Safari only gives web pages the camera
on https). The phone opens it, taps Start, and then:
- sends its back-camera picture to the Mac as JPEG frames, as fast as the
  network allows (one frame in flight at a time, so latency never builds up);
- gets back, with every frame, what to say (speechSynthesis, urgent sentences
  interrupt), the beep target (WebAudio, panned left/right) and the status
  (CLEAR / CAUTION / DANGER) to show in big colours.
The Mac runs the whole Occhio pipeline unchanged: PhoneCamera behaves like CameraStream.

The certificate is self-signed (made once with the `openssl` command in
config/phone_cert/, never committed): Safari shows "This Connection Is Not
Private" the first time -> Show Details -> visit this website.
Only Python's standard library is used (http.server + ssl).
"""
from __future__ import annotations

import json
import ssl
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from config.settings import CONFIG_DIR
from occhio.capture import Frame
from occhio.vibration import lan_ip

CERT_DIR = CONFIG_DIR / "phone_cert"

PAGE = """<!doctype html><html lang="__LANG__"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Occhio</title>
<style>
 :root{--bg:#141618;--fg:#f4f4f4;--muted:#a0a4a8}
 *{box-sizing:border-box} html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
 font-family:-apple-system,system-ui,sans-serif}
 #wrap{display:flex;flex-direction:column;height:100%;padding:env(safe-area-inset-top) 16px 16px}
 #top{display:flex;justify-content:space-between;align-items:center;padding:12px 0}
 #name{font-weight:700;font-size:22px} #net{color:var(--muted);font-size:14px}
 #status{border-radius:22px;padding:22px 16px;text-align:center;font-weight:800;font-size:44px;
  letter-spacing:1px;background:#3a3d40;transition:background .2s}
 #say{flex:1;display:flex;align-items:center;justify-content:center;text-align:center;
  font-size:30px;font-weight:600;padding:12px;line-height:1.25}
 #preview{width:100%;max-height:28vh;object-fit:cover;border-radius:14px;opacity:.85;background:#000}
 button{width:100%;font-size:24px;font-weight:700;padding:20px;border:0;border-radius:18px;
  background:#2fa36b;color:#fff;margin-top:12px}
 #err{color:#ff8a80;font-size:16px;text-align:center;min-height:20px}
</style></head><body><div id="wrap">
<div id="top"><div id="name">Occhio</div><div id="net">__READY__</div></div>
<div id="status">__READY__</div>
<div id="say">__TAP__</div>
<video id="preview" playsinline muted autoplay></video>
<div id="err"></div>
<button id="go">__START__</button>
</div><canvas id="c" style="display:none"></canvas>
<script>
const LANG = "__VOICE__";
const COLOURS = {clear:"#2f9e57", caution:"#e39b16", danger:"#d93838", unreliable:"#6b6f73"};
let running = false, audio = null, beepTimer = null, beep = null, lastSeq = 0, sent = 0, t0 = 0;
const v = document.getElementById("preview"), c = document.getElementById("c"), g = c.getContext("2d");
const err = m => document.getElementById("err").textContent = m || "";

function speak(text, urgent) {
  if (!("speechSynthesis" in window)) return;
  if (urgent) speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text); u.lang = LANG; u.rate = 1.05;
  speechSynthesis.speak(u);
}
function playBeep() {
  if (!audio || !beep) return;
  const o = audio.createOscillator(), gain = audio.createGain(), pan = audio.createStereoPanner();
  o.frequency.value = beep.freq; pan.pan.value = beep.pan;
  const now = audio.currentTime;
  gain.gain.setValueAtTime(0, now); gain.gain.linearRampToValueAtTime(0.12, now + 0.012);
  gain.gain.linearRampToValueAtTime(0, now + 0.045);
  o.connect(gain).connect(pan).connect(audio.destination); o.start(now); o.stop(now + 0.05);
}
function setBeep(b) {
  const changed = JSON.stringify(b) !== JSON.stringify(beep); beep = b;
  if (!changed) return;
  clearInterval(beepTimer); beepTimer = null;
  if (b) { playBeep(); beepTimer = setInterval(playBeep, 1000 / b.rate); }
}
async function loop() {
  while (running) {
    if (v.videoWidth) {
      const scale = Math.min(1, 960 / Math.max(v.videoWidth, v.videoHeight));
      c.width = Math.round(v.videoWidth * scale); c.height = Math.round(v.videoHeight * scale);
      g.drawImage(v, 0, 0, c.width, c.height);
      const blob = await new Promise(r => c.toBlob(r, "image/jpeg", 0.7));
      try {
        const r = await fetch("/frame?seq=" + lastSeq, {method: "POST", body: blob, cache: "no-store"});
        const d = await r.json(); sent++;
        for (const m of d.say) { speak(m.text, m.urgent); document.getElementById("say").textContent = m.text; }
        if (d.say.length) lastSeq = d.seq; else lastSeq = Math.max(lastSeq, d.seq);
        const st = document.getElementById("status"); st.textContent = d.label; st.style.background = COLOURS[d.status] || "#3a3d40";
        setBeep(d.beep);
        if (d.vibrate && navigator.vibrate) navigator.vibrate(d.vibrate);
        document.getElementById("net").textContent = Math.round(sent / ((performance.now() - t0) / 1000)) + " fps";
        err("");
      } catch (e) { err("Mac not reachable: same Wi-Fi? " + e); await new Promise(r => setTimeout(r, 500)); }
    } else { await new Promise(r => setTimeout(r, 50)); }
  }
}
document.getElementById("go").onclick = async () => {
  try {
    audio = new (window.AudioContext || window.webkitAudioContext)(); await audio.resume();
    speak(" ", false);  // unlocks speech on iOS (must happen inside a tap)
    const s = await navigator.mediaDevices.getUserMedia({audio: false,
      video: {facingMode: {ideal: "environment"}, width: {ideal: 1280}, height: {ideal: 720}}});
    v.srcObject = s; await v.play();
    if ("wakeLock" in navigator) { try { await navigator.wakeLock.request("screen"); } catch (e) {} }
    document.getElementById("go").style.display = "none";
    running = true; t0 = performance.now(); loop();
  } catch (e) { err("Camera not allowed: " + e + (window.isSecureContext ? "" : " (page is not https)")); }
};
</script></body></html>"""


def ensure_certificate(ip: str, cert_dir: Path = CERT_DIR) -> tuple[Path, Path]:
    """Self-signed certificate for this Mac's address (made once with the `openssl` command)."""
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    stamp = cert_dir / "ip.txt"
    if cert.exists() and key.exists() and stamp.exists() and stamp.read_text().strip() == ip:
        return cert, key
    cert_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
                    "-out", str(cert), "-days", "30", "-subj", "/CN=Occhio",
                    "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
                   check=True, capture_output=True)
    stamp.write_text(ip)
    return cert, key


class PhoneCamera:
    """Receives frames from the phone page; same read()/running/stop() interface as CameraStream.

    Things to send back go through say() / set_beep() / set_status(); every frame
    request returns the sentences the phone has not spoken yet.
    """

    def __init__(self, port: int = 8443, lang: str = "en", use_https: bool = True,
                 strings: dict[str, str] | None = None) -> None:
        self.port = port
        self.use_https = use_https
        self.ip = lan_ip()
        self.lang = lang
        self.strings = strings or {}
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._last_returned = -1
        self._grabbed = 0
        self._stopped = threading.Event()
        self._messages: deque[tuple[int, str, bool]] = deque(maxlen=20)
        self._seq = 0
        self._beep: dict | None = None
        self._status = ("clear", "")
        self._vibrate: int | None = None
        self.last_frame_time: float | None = None
        self._server: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------ to the phone
    def say(self, text: str, urgent: bool = False) -> None:
        with self._cond:
            self._seq += 1
            self._messages.append((self._seq, text, urgent))
            if urgent:
                self._vibrate = 300

    def set_beep(self, target: tuple[float, float, float] | None) -> None:
        self._beep = None if target is None else {"rate": round(target[0], 2), "pan": round(target[1], 2),
                                                   "freq": round(target[2])}

    def set_status(self, status: str, label: str) -> None:
        self._status = (status, label)

    def _reply(self, since: int) -> dict:
        with self._cond:
            msgs = [{"text": t, "urgent": u} for s, t, u in self._messages if s > since]
            vib, self._vibrate = self._vibrate, None
            return {"say": msgs, "seq": self._seq, "status": self._status[0], "label": self._status[1],
                    "beep": self._beep, "vibrate": vib}

    # ------------------------------------------------------------- from the phone
    def _receive(self, jpeg: bytes) -> bool:
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return False
        now = time.perf_counter()
        with self._cond:
            self._latest = Frame(image=image, index=self._grabbed, timestamp=now)
            self._grabbed += 1
            self.last_frame_time = now
            self._cond.notify_all()
        return True

    @property
    def url(self) -> str:
        return f"{'https' if self.use_https else 'http'}://{self.ip}:{self.port}"

    @property
    def frames_grabbed(self) -> int:
        return self._grabbed

    @property
    def running(self) -> bool:
        return not self._stopped.is_set()

    def read(self, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            self._cond.wait_for(lambda: (self._latest is not None and self._latest.index != self._last_returned)
                                or self._stopped.is_set(), timeout=timeout)
            frame = self._latest
            if frame is None or frame.index == self._last_returned:
                return None
            self._last_returned = frame.index
            return frame

    # ---------------------------------------------------------------- server
    def page(self) -> bytes:
        s = self.strings
        voice = {"en": "en-US", "it": "it-IT"}.get(self.lang, "en-US")
        html = (PAGE.replace("__LANG__", self.lang).replace("__VOICE__", voice)
                .replace("__READY__", s.get("ready", "Ready")).replace("__TAP__", s.get("tap", "Tap Start"))
                .replace("__START__", s.get("start", "Start")))
        return html.encode()

    def start(self) -> "PhoneCamera":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                self._send(200, owner.page(), "text/html; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.startswith("/frame"):
                    self._send(404, b"{}", "application/json")
                    return
                length = int(self.headers.get("Content-Length", 0))
                ok = owner._receive(self.rfile.read(length)) if 0 < length < 8_000_000 else False
                try:
                    since = int(self.path.split("seq=")[1]) if "seq=" in self.path else 0
                except ValueError:
                    since = 0
                body = json.dumps(owner._reply(since) if ok else {"error": "bad frame"}).encode()
                self._send(200 if ok else 400, body, "application/json")

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        if self.use_https:
            cert, key = ensure_certificate(self.ip)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(cert), str(key))
            self._server.socket = ctx.wrap_socket(self._server.socket, server_side=True)
        threading.Thread(target=self._server.serve_forever, name="PhoneLink", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stopped.set()
        with self._cond:
            self._cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
