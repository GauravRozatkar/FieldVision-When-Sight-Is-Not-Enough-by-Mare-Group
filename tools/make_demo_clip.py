"""Make a test video that follows the live demo script, so Occhio can be tested without space.

    python tools/make_demo_clip.py demo.mp4
    python main.py --source demo.mp4 --simulate-keys "<printed by this script>"

A person (cut out of bus.jpg) walks toward the camera, crosses fast, walks behind a
"chair", a traffic light turns from red to green, the lens is covered, then the keys
d and m are pressed automatically. The last line printed is the exact command to run.
Prints the timeline so you know what Occhio should say when.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def webcam_like(mean: float = 150, contrast: float = 60, seed: int = 0) -> np.ndarray:
    """A room-like background: big shapes with fine texture (reliability sees a normal image)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((360, 640), dtype=np.float32)
    for _ in range(25):
        x, y = rng.integers(0, 640), rng.integers(0, 360)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(40, 250)), int(y + rng.integers(40, 200))),
                      float(rng.normal(0, 1)), -1)
    img += rng.normal(0, 0.15, img.shape).astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6) * contrast + mean
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


from occhio.detector import Detector
from config.profiles import PROFILES
bus = cv2.imread(str(Path(__file__).resolve().parent.parent / "bus.jpg")); d = Detector(PROFILES["street"]["classes"])
W, H, FPS = 1280, 720, 30; focal = (W / 2) / math.tan(math.radians(35))
bg = cv2.resize(webcam_like(mean=140, contrast=45, seed=3), (W, H))


def conf_at(crop, dist):
    """YOLO confidence for this cut-out pasted at `dist` metres on the clip background."""
    h = int(focal * 1.70 / dist)
    c = cv2.resize(crop, None, fx=h / crop.shape[0], fy=h / crop.shape[0])
    f = bg.copy(); f[H - 20 - c.shape[0]:H - 20, 600:600 + c.shape[1]] = c
    d.reset()
    return max((x.conf for x in [d.detect(f) for _ in range(3)][-1] if x.cls_name == "person"), default=0.0)


d.reset()
crops = [bus[int(p.xyxy[1]):int(p.xyxy[3]), int(p.xyxy[0]):int(p.xyxy[2])]
         for p in d.detect(bus) if p.cls_name == "person" and p.conf > 0.5]
crop = max(crops, key=lambda c: conf_at(c, 4.0) + conf_at(c, 2.5))  # the one YOLO trusts at all distances
d.reset()
out = cv2.VideoWriter(sys.argv[1] if len(sys.argv) > 1 else "demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
n = 0; marks = []

def person(f, dist, cx):
    h = int(focal * 1.70 / dist); sc = h / crop.shape[0]
    c = cv2.resize(crop, None, fx=sc, fy=sc); ch, cw = c.shape[:2]
    bottom = H - 20 if dist > 2.3 else H - 20 + int((2.3 - dist) * 120)  # feet leave the frame up close
    top, left = bottom - ch, int(cx - cw / 2)
    ys, ye, xs, xe = max(top, 0), min(bottom, H), max(left, 0), min(left + cw, W)
    if ye > ys and xe > xs:
        f[ys:ye, xs:xe] = c[ys - top:ye - top, xs - left:xe - left]

def write(f, k=1):
    global n
    for _ in range(k): out.write(f); n += 1

def mark(label): marks.append((n / FPS, label))

mark("empty room"); write(bg, 30)
mark("1. walk toward the camera 4.5 m -> 1.3 m at 0.6 m/s")
dist = 4.5
while dist > 1.3:
    f = bg.copy(); person(f, dist, W / 2); write(f); dist -= 0.6 / FPS
mark("walks away (empty)"); write(bg, 45)
mark("2. fast crossing left -> right at 2.6 m")
for i in range(40):
    f = bg.copy(); person(f, 2.6, -150 + i * 40); write(f)
write(bg, 30)
mark("3. walks behind a chair at 3.2 m")
for i in range(110):
    f = bg.copy(); person(f, 3.2, 150 + i * 9)
    cv2.rectangle(f, (560, 150), (760, H), (35, 30, 30), -1); write(f)
write(bg, 30)
mark("4. traffic light: red for 3 s, then green for 3 s")
for lit in ("red", "green"):
    for _ in range(90):
        f = bg.copy()
        cv2.rectangle(f, (628, 200), (640, 560), (60, 60, 60), -1)  # pole
        cv2.rectangle(f, (595, 60), (675, 260), (25, 25, 25), -1)  # housing
        for i, c in enumerate([(30, 30, 230), (0, 160, 230), (60, 220, 40)]):
            on = (lit == "red" and i == 0) or (lit == "green" and i == 2)
            cv2.circle(f, (635, 95 + i * 65), 25, c if on else (50, 50, 50), -1)
        write(f)
write(bg, 20)
last = bg
mark("5. lens covered for 2 s"); rng = np.random.default_rng(1)
for _ in range(60): write(np.clip(last * 0.03 + rng.normal(0, 2, last.shape), 0, 255).astype(np.uint8))
mark("lens free"); write(bg, 45)
mark("6. press d here (describe)"); write(bg, 60)
mark("7. press m here (industrial), then a person stands at 4 m")
write(bg, 20)
for _ in range(150):
    f = bg.copy(); person(f, 4.0, W / 2 + 50); write(f)
write(bg, 30)
out.release()
for t, label in marks: print(f"{t:6.2f}s  {label}")
print(f"{n / FPS:6.2f}s  end")
key_d = next(t for t, label in marks if label.startswith("6.")) + 0.3
key_m = next(t for t, label in marks if label.startswith("7.")) + 0.1
name = sys.argv[1] if len(sys.argv) > 1 else "demo.mp4"
print(f'\nrun it:  python main.py --source {name} --simulate-keys "{key_d:.1f}:d,{key_m:.1f}:m"')
