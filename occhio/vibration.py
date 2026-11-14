"""Deaf-blind vibration (F24, stretch): the danger level goes to a phone that vibrates.

A tiny HTTP server (Python standard library, own thread) serves one page and a
/level endpoint. The phone opens http://<mac-ip>:PORT on the same Wi-Fi or
hotspot, taps "Start" once (browsers require a tap before vibrating), and the
page polls the level 5 times a second and calls navigator.vibrate with pulses
that get faster as the danger level rises, like the beeps.

navigator.vibrate works in Chrome on Android; iPhone Safari does not support it
(the page then shows the level in big colours instead).
"""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config.settings import VIBRATION

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Occhio vibra</title>
<style>
 body{margin:0;font-family:-apple-system,system-ui,sans-serif;background:#111;color:#fff;
      display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh}
 #lvl{font-size:22vw;font-weight:700} #txt{font-size:6vw;margin:8px 16px;text-align:center}
 button{font-size:7vw;padding:16px 32px;border-radius:16px;border:0;background:#2a7;color:#fff}
</style></head><body>
<div id="lvl">-</div><div id="txt">Tocca Start / Tap Start</div><button id="go">Start</button>
<script>
let on=false, last=0;
const colours=["#111","#264","#a70","#c22"];
document.getElementById("go").onclick=()=>{on=true;document.getElementById("go").style.display="none";
  if(navigator.vibrate)navigator.vibrate(50);};
async function poll(){
  try{const r=await fetch("/level",{cache:"no-store"});const d=await r.json();
    document.getElementById("lvl").textContent=d.level;
    document.getElementById("txt").textContent=d.text||"";
    document.body.style.background=colours[Math.min(d.level,3)];
    const now=Date.now(), gap=d.level>=3?250:d.level==2?600:d.level==1?1200:0;
    if(on&&gap&&now-last>=gap&&navigator.vibrate){navigator.vibrate(d.level>=3?180:100);last=now;}
  }catch(e){document.getElementById("txt").textContent="...";}
  setTimeout(poll,200);}
poll();
</script></body></html>"""


def lan_ip() -> str:
    """Best guess of this Mac's address on the local network (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class VibrationServer:
    """level: 0 nothing, 1 warn, 2 approaching / close, 3 danger."""

    def __init__(self, port: int | None = None) -> None:
        self.port = int(port or VIBRATION["port"])
        self.level = 0
        self.text = ""
        self._server: ThreadingHTTPServer | None = None

    def set(self, level: int, text: str = "") -> None:
        self.level, self.text = max(0, min(3, int(level))), text

    @property
    def url(self) -> str:
        return f"http://{lan_ip()}:{self.port}"

    def start(self) -> "VibrationServer":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                if self.path.startswith("/level"):
                    body = json.dumps({"level": owner.level, "text": owner.text}).encode()
                    ctype = "application/json"
                else:
                    body, ctype = PAGE.encode(), "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:  # keep the console clean
                pass

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, name="VibrationServer", daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
