"""iPhone (or any phone) as camera, voice and beeper, through Safari: no app to install.

    python main.py --source phone        # prints https://<mac-ip>:8443 to open on the phone

The Mac serves one web page over HTTPS (Safari only gives web pages the camera
on https). The phone opens it, taps Start, and then:
- sends its back-camera picture to the Mac as 640 px JPEG frames (two uploads
  in flight), plus its motion sensors: the user's walking speed (step counter),
  the phone's tilt (gravity) and turning rate;
- gets back, with every frame, what to say (speechSynthesis, unlocked with a
  real sentence in the Start tap; urgent sentences interrupt after a short pause
  because iOS goes silent otherwise; until the phone reports its voice works,
  the Mac speaks too), the beep target (WebAudio, panned left/right) and the status
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
 :root{--bg:#000;--fg:#f4f4f4;--muted:#c0c4c8}
 *{box-sizing:border-box} html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
 font-family:-apple-system,system-ui,sans-serif;overflow:hidden}
 #stage{position:fixed;inset:0}
 #preview,#boxes{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
 #top{position:absolute;left:0;right:0;top:0;padding:calc(env(safe-area-inset-top) + 10px) 14px 10px;
  display:flex;flex-direction:column;gap:8px;background:linear-gradient(#000c,#0000)}
 #row{display:flex;justify-content:space-between;align-items:center}
 #name{font-weight:700;font-size:20px} #net{color:var(--muted);font-size:13px;text-align:right}
 #status{border-radius:18px;padding:10px 14px;text-align:center;font-weight:800;font-size:30px;
  letter-spacing:1px;background:#3a3d40cc;transition:background .2s}
 #bottom{position:absolute;left:0;right:0;bottom:0;padding:14px 14px calc(env(safe-area-inset-bottom) + 12px);
  background:linear-gradient(#0000,#000d 35%)}
 #say{text-align:center;font-size:26px;font-weight:700;line-height:1.25;min-height:34px;
  text-shadow:0 2px 6px #000}
 button{width:100%;font-size:22px;font-weight:700;padding:18px;border:0;border-radius:16px;
  background:#2fa36b;color:#fff;margin-top:10px}
 #voice{display:none;background:#3a3d40cc;font-size:16px;padding:10px}
 #err{color:#ff8a80;font-size:15px;text-align:center;min-height:18px}
</style></head><body><div id="stage">
<video id="preview" playsinline muted autoplay></video>
<canvas id="boxes"></canvas>
<div id="top"><div id="row"><div id="name">Occhio</div><div id="net">__READY__</div></div>
<div id="status">__READY__</div></div>
<div id="bottom"><div id="say">__TAP__</div><div id="err"></div>
<button id="voice">__TESTVOICE__</button><button id="go">__START__</button></div>
</div>
<script>
const LANG = "__VOICE__", VOICE_ON_TEXT = "__VOICEON__";
const COLOURS = {clear:"#2f9e57", caution:"#e39b16", danger:"#d93838", unreliable:"#6b6f73"};
let running = false, audio = null, beepTimer = null, beep = null, sent = 0, t0 = 0;
const v = document.getElementById("preview");
const err = m => document.getElementById("err").textContent = m || "";

// ---- speech: iOS needs a real sentence spoken inside a tap, and a pause after cancel()
let voiceOk = false, voice = null;
const queue = [];
function pickVoice() {
  const vs = speechSynthesis.getVoices();
  voice = vs.find(x => x.lang === LANG) || vs.find(x => x.lang && x.lang.startsWith(LANG.slice(0, 2))) || null;
}
if ("speechSynthesis" in window) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }
function utter(text) {
  const u = new SpeechSynthesisUtterance(text);
  u.lang = LANG; u.rate = 1.05; u.volume = 1; if (voice) u.voice = voice;
  u.onstart = () => { voiceOk = true; };
  u.onend = () => { if (queue.length) setTimeout(() => utter(queue.shift()), 60); };
  u.onerror = e => { if (e.error !== "interrupted" && e.error !== "canceled") err("voice: " + e.error);
                     if (queue.length) setTimeout(() => utter(queue.shift()), 60); };
  speechSynthesis.speak(u);
}
function speak(text, urgent) {
  if (!("speechSynthesis" in window)) return;
  if (urgent) {
    queue.length = 0;
    if (speechSynthesis.speaking || speechSynthesis.pending) { speechSynthesis.cancel(); setTimeout(() => utter(text), 250); }
    else utter(text);
    return;
  }
  if (speechSynthesis.speaking || speechSynthesis.pending) { queue.push(text); while (queue.length > 2) queue.shift(); }
  else utter(text);
}
function unlockVoice() { if ("speechSynthesis" in window) { speechSynthesis.cancel(); utter(VOICE_ON_TEXT); } }

// ---- beeps (WebAudio)
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

// ---- motion sensors: phone tilt (for floor distances) and the user's walking speed (steps)
const motion = {pitch: null, speed: 0, turn: 0};
let grav = null, lastStep = 0; const steps = [];
function onMotion(e) {
  const g = e.accelerationIncludingGravity, a = e.acceleration, r = e.rotationRate;
  const now = performance.now();
  if (g && g.z !== null) {
    grav = grav ? {x: grav.x * 0.9 + g.x * 0.1, y: grav.y * 0.9 + g.y * 0.1, z: grav.z * 0.9 + g.z * 0.1} : {x: g.x, y: g.y, z: g.z};
    const n = Math.hypot(grav.x, grav.y, grav.z) || 1;
    const zSign = /iPhone|iPad|iPod/.test(navigator.userAgent) ? -1 : 1;  // iOS reports z with the other sign
    motion.pitch = Math.asin(Math.max(-1, Math.min(1, zSign * grav.z / n))) * 180 / Math.PI;  // camera below horizon, deg
  }
  if (a && a.x !== null) {
    const mag = Math.hypot(a.x, a.y, a.z);
    if (mag > 1.3 && now - lastStep > 300) { lastStep = now; steps.push(now); }
    while (steps.length && now - steps[0] > 4000) steps.shift();
    const cadence = steps.length >= 3 ? (steps.length - 1) / ((steps[steps.length - 1] - steps[0]) / 1000) : 0;
    motion.speed = now - lastStep > 1500 ? 0 : Math.min(2.0, cadence * 0.65);  // ~0.65 m per step
  }
  if (r && r.alpha !== null) motion.turn = Math.hypot(r.alpha, r.beta, r.gamma);
}

// ---- detection boxes from the Mac, drawn over the picture
const bc = document.getElementById("boxes"), bg = bc.getContext("2d");
function drawBoxes(boxes) {
  const dpr = window.devicePixelRatio || 1, W = bc.clientWidth, H = bc.clientHeight;
  if (bc.width !== W * dpr || bc.height !== H * dpr) { bc.width = W * dpr; bc.height = H * dpr; }
  bg.setTransform(dpr, 0, 0, dpr, 0, 0); bg.clearRect(0, 0, W, H);
  if (!v.videoWidth || !boxes) return;
  const s = Math.min(W / v.videoWidth, H / v.videoHeight), vw = v.videoWidth * s, vh = v.videoHeight * s;
  const ox = (W - vw) / 2, oy = (H - vh) / 2;
  bg.font = "600 16px -apple-system, system-ui, sans-serif"; bg.lineWidth = 3;
  for (const b of boxes) {
    const x = ox + b.x1 * vw, y = oy + b.y1 * vh, w = (b.x2 - b.x1) * vw, h = (b.y2 - b.y1) * vh;
    bg.strokeStyle = b.c; bg.strokeRect(x, y, w, h);
    const tw = bg.measureText(b.label).width + 14, ty = Math.max(oy, y - 26);
    bg.fillStyle = b.c; bg.beginPath(); bg.roundRect ? bg.roundRect(x, ty, tw, 24, 7) : bg.rect(x, ty, tw, 24); bg.fill();
    bg.fillStyle = "#fff"; bg.fillText(b.label, x + 7, ty + 17);
  }
}

// ---- frames out, guidance in (two uploads in flight hide the Wi-Fi round trip)
async function loop() {
  const cv = document.createElement("canvas"), cg = cv.getContext("2d");
  while (running) {
    if (v.videoWidth) {
      const scale = Math.min(1, 640 / Math.max(v.videoWidth, v.videoHeight));
      cv.width = Math.round(v.videoWidth * scale); cv.height = Math.round(v.videoHeight * scale);
      cg.drawImage(v, 0, 0, cv.width, cv.height);
      const blob = await new Promise(r => cv.toBlob(r, "image/jpeg", 0.65));
      const q = "voice=" + (voiceOk ? 1 : 0) + "&speed=" + motion.speed.toFixed(2) + "&turn=" + Math.round(motion.turn)
              + (motion.pitch === null ? "" : "&pitch=" + motion.pitch.toFixed(1));
      try {
        const r = await fetch("/frame?" + q, {method: "POST", body: blob, cache: "no-store"});
        const d = await r.json(); sent++;
        for (const m of d.say) { speak(m.text, m.urgent); document.getElementById("say").textContent = m.text; }
        const st = document.getElementById("status"); st.textContent = d.label; st.style.background = COLOURS[d.status] || "#3a3d40";
        setBeep(d.beep);
        drawBoxes(d.boxes);
        if (d.vibrate && navigator.vibrate) navigator.vibrate(d.vibrate);
        const fps = Math.round(sent / ((performance.now() - t0) / 1000));
        document.getElementById("net").textContent = fps + " fps" + (motion.speed > 0.2 ? " · " + motion.speed.toFixed(1) + " m/s" : "")
          + (voiceOk ? "" : " · no voice yet");
        err("");
      } catch (e) { err("Mac not reachable: same Wi-Fi? " + e); await new Promise(r => setTimeout(r, 500)); }
    } else { await new Promise(r => setTimeout(r, 50)); }
  }
}
document.getElementById("voice").onclick = unlockVoice;
document.getElementById("go").onclick = async () => {
  try {
    unlockVoice();  // must happen inside the tap, with a real sentence
    audio = new (window.AudioContext || window.webkitAudioContext)(); await audio.resume();
    if (typeof DeviceMotionEvent !== "undefined" && DeviceMotionEvent.requestPermission) {
      try { if (await DeviceMotionEvent.requestPermission() === "granted") addEventListener("devicemotion", onMotion); }
      catch (e) {}
    } else { addEventListener("devicemotion", onMotion); }
    const s = await navigator.mediaDevices.getUserMedia({audio: false,
      video: {facingMode: {ideal: "environment"}, width: {ideal: 1280}, height: {ideal: 720}}});
    v.srcObject = s; await v.play();
    if ("wakeLock" in navigator) { try { await navigator.wakeLock.request("screen"); } catch (e) {} }
    document.getElementById("go").style.display = "none";
    document.getElementById("voice").style.display = "block";
    running = true; t0 = performance.now(); loop(); setTimeout(loop, 40);
  } catch (e) { err("Camera not allowed: " + e + (window.isSecureContext ? "" : " (page is not https)")); }
};
</script></body></html>"""


def _jsonable(value):
    """numpy scalars (np.bool_, np.float64...) -> plain Python values for json."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


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
        self._delivered = 0  # messages up to this seq were already sent to the phone
        self._beep: dict | None = None
        self._boxes: list[dict] = []
        self._status = ("clear", "")
        self._vibrate: int | None = None
        self.last_frame_time: float | None = None
        self.phone_voice_ok = False  # the phone confirmed its speech works (onstart fired)
        self.walk_speed = 0.0  # m/s from the phone's step counter
        self.pitch_deg: float | None = None  # camera angle below the horizon, from gravity
        self.turn_dps = 0.0  # how fast the phone is turning (deg/s)
        self._server: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------ to the phone
    def say(self, text: str, urgent: bool = False) -> None:
        urgent = bool(urgent)  # numpy booleans are not JSON
        with self._cond:
            self._seq += 1
            self._messages.append((self._seq, str(text), urgent))
            if urgent:
                self._vibrate = 300

    def set_beep(self, target: tuple[float, float, float] | None) -> None:
        self._beep = None if target is None else {"rate": round(float(target[0]), 2),
                                                   "pan": round(float(target[1]), 2), "freq": int(round(target[2]))}

    def set_boxes(self, boxes: list[dict]) -> None:
        """[{x1, y1, x2, y2 (fractions of the frame), label, c (CSS colour)}] drawn on the phone."""
        self._boxes = boxes

    def set_status(self, status: str, label: str) -> None:
        self._status = (str(status), str(label))

    def _reply(self, since: int = 0) -> dict:
        """Everything for the phone; each sentence is delivered exactly once."""
        with self._cond:
            msgs = [{"text": t, "urgent": u} for s, t, u in self._messages if s > max(since, self._delivered)]
            self._delivered = self._seq
            vib, self._vibrate = self._vibrate, None
            return {"say": msgs, "seq": self._seq, "status": self._status[0], "label": self._status[1],
                    "beep": self._beep, "vibrate": vib, "boxes": self._boxes}

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

    def _read_query(self, path: str) -> None:
        """voice=1 (the phone's speech works), speed (m/s, from steps), pitch (deg below horizon), turn (deg/s)."""
        from urllib.parse import parse_qs, urlparse
        q = {k: v[0] for k, v in parse_qs(urlparse(path).query).items()}
        try:
            if "voice" in q:
                self.phone_voice_ok = q["voice"] == "1"
            self.walk_speed = max(0.0, min(3.0, float(q.get("speed", self.walk_speed))))
            self.turn_dps = float(q.get("turn", self.turn_dps))
            if "pitch" in q:
                self.pitch_deg = max(-60.0, min(80.0, float(q["pitch"])))
        except ValueError:
            pass

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
                .replace("__START__", s.get("start", "Start"))
                .replace("__TESTVOICE__", s.get("test_voice", "Test voice"))
                .replace("__VOICEON__", s.get("voice_on", "Voice on")))
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
                owner._read_query(self.path)
                try:
                    body = json.dumps(owner._reply() if ok else {"error": "bad frame"}, default=_jsonable).encode()
                except Exception as exc:  # never leave the phone without an answer
                    body = json.dumps({"error": repr(exc)[:200]}).encode()
                    ok = False
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
