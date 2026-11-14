"""Make a 60 s "busy room" test video: two people close to the camera, moving around.

    python tools/make_busy_clip.py busy.mp4
    python main.py --source busy.mp4 --no-voice      # count the [voice] lines

Person A sits about 1 m in front and leans across the left/ahead boundary; person B walks
back and forth on the right between 1.2 m and 3 m. Used to check that speech stays calm
and clear in crowded scenes.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from occhio.detector import Detector  # noqa: E402

W, H, FPS, SECONDS = 1280, 720, 30, 60
FOCAL = (W / 2) / math.tan(math.radians(35))


def background() -> np.ndarray:
    rng = np.random.default_rng(3)
    img = np.zeros((360, 640), dtype=np.float32)
    for _ in range(25):
        x, y = rng.integers(0, 640), rng.integers(0, 360)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(40, 250)), int(y + rng.integers(40, 200))),
                      float(rng.normal(0, 1)), -1)
    img += rng.normal(0, 0.15, img.shape).astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6) * 45 + 140
    return cv2.resize(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (W, H))


def paste(frame: np.ndarray, crop: np.ndarray, dist: float, cx: float) -> None:
    h = int(FOCAL * 1.70 / dist)
    c = cv2.resize(crop, None, fx=h / crop.shape[0], fy=h / crop.shape[0])
    ch, cw = c.shape[:2]
    bottom = H - 20 if dist > 2.3 else H - 20 + int((2.3 - dist) * 120)
    top, left = bottom - ch, int(cx - cw / 2)
    ys, ye, xs, xe = max(top, 0), min(bottom, H), max(left, 0), min(left + cw, W)
    if ye > ys and xe > xs:
        frame[ys:ye, xs:xe] = c[ys - top:ye - top, xs - left:xe - left]


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "busy.mp4"
    bus = cv2.imread(str(ROOT / "bus.jpg"))
    det = Detector(["person"])
    people = sorted((d for d in det.detect(bus) if d.conf > 0.8), key=lambda d: -d.conf)[:2]
    crops = [bus[int(p.xyxy[1]):int(p.xyxy[3]), int(p.xyxy[0]):int(p.xyxy[2])] for p in people]
    bg = background()
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(SECONDS * FPS):
        t = i / FPS
        f = bg.copy()
        dist_b = 2.1 + 0.9 * math.sin(2 * math.pi * t / 8)  # B: 1.2 m .. 3 m and back, every 8 s
        paste(f, crops[1], dist_b, W * 0.80 + 40 * math.sin(2 * math.pi * t / 5))
        dist_a = 1.15 + 0.12 * math.sin(2 * math.pi * t / 3)  # A: about 1 m, small movements
        paste(f, crops[0], dist_a, W / 3 + 70 * math.sin(2 * math.pi * t / 4))  # leaning across the boundary
        out.write(f)
    out.release()
    print(f"wrote {out_path}: {SECONDS} s, 2 people close to the camera")


if __name__ == "__main__":
    main()
