"""The app window: camera view plus a clean heads-up display.

    +----------------------------------------------------------------------------+
    | Occhio  [STREET]           (  DANGER  )                   MIC  ECO  25 FPS  |
    +---------------------------------------------------+------------------------+
    |                                                   | NOW                    |
    |   camera: walking corridor, simple labels         |  Person ahead, 1 metre |
    |   ("Person · 1.2 m"), red = danger                | RADAR (top view)       |
    |                                                   | AROUND YOU             |
    |                                                   |  Person  ahead  1.2 m  |
    |                                                   | RECENT                 |
    +---------------------------------------------------+------------------------+
    |  [D] Describe  [T] Read  [V] Voice  [M] Mode  [I] Details  [Q] Quit  Sound ▮▮ |
    +----------------------------------------------------------------------------+

Shapes are drawn with OpenCV (fast). Text uses the system font (SF on macOS),
which looks far better than OpenCV's font and supports accents: each distinct
label is rendered once by Pillow, cached, and alpha-blended onto the frame, so
compose() costs only a few milliseconds per frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from occhio.spatial import SpatialInfo

# BGR colours
BG = (24, 22, 20)
PANEL = (36, 33, 30)
CARD = (48, 44, 40)
LINE = (70, 66, 62)
WHITE = (245, 245, 245)
MUTED = (165, 160, 155)
GREEN = (90, 200, 90)
AMBER = (40, 170, 245)
RED = (60, 60, 230)
BLUE = (235, 160, 60)
GREY = (140, 140, 140)
YELLOW = (60, 210, 240)
ORANGE = AMBER

TOP_H, BOTTOM_H, PANEL_W = 60, 52, 360
CAM_W, CAM_H = 1024, 576

STATUS_COLOURS = {"clear": GREEN, "caution": AMBER, "danger": RED, "unreliable": GREY}

FONT_FILES = ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf", "DejaVuSans.ttf"]


@lru_cache(maxsize=32)
def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_FILES:
        try:
            f = ImageFont.truetype(path, size)
            try:
                f.set_variation_by_name(weight)
            except Exception:
                pass
            return f
        except OSError:
            continue
    return ImageFont.load_default(size)


def rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return bgr[2], bgr[1], bgr[0]


@dataclass
class Label:
    """Text drawn in the Pillow pass. x, y = top-left in window pixels."""

    x: int
    y: int
    text: str
    colour: tuple[int, int, int] = WHITE
    size: int = 15
    weight: str = "Regular"
    pill: tuple[int, int, int] | None = None  # background colour (rounded) or None
    anchor: str = "la"  # Pillow anchor: la = left/top, ma = centre/top, ra = right/top


@dataclass
class Nearby:
    name: str
    where: str
    distance_m: float
    colour: tuple[int, int, int]


@dataclass
class HudState:
    mode_label: str
    industrial: bool
    status: str  # clear | caution | danger | unreliable
    status_label: str
    now_text: str
    now_urgent: bool
    fps: float
    chips: list[tuple[str, tuple[int, int, int]]] = field(default_factory=list)
    radar: list[SpatialInfo] = field(default_factory=list)
    danger_ids: set[int] = field(default_factory=set)
    nearby: list[Nearby] = field(default_factory=list)
    recent: list[tuple[str, str, bool]] = field(default_factory=list)
    warn_m: float = 3.0
    danger_m: float = 1.5
    hfov_deg: float = 70.0
    sound_db: float = -120.0
    sound_last: str = ""
    keys: list[tuple[str, str]] = field(default_factory=list)  # (key, label)
    hint: str = ""  # replaces the keys row when set (e.g. drawing the danger zone)
    titles: dict[str, str] = field(default_factory=dict)  # now / radar / around / recent / nothing / sound
    labels: list[Label] = field(default_factory=list)  # object labels from the camera view (window coords)
    details: str = ""  # debug line (FPS per stage), shown with the 'i' key


def _rounded(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, r: int, colour, thickness: int = -1) -> None:
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), colour, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), colour, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, colour, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)


@lru_cache(maxsize=4096)
def text_width(text: str, size: int, weight: str = "Regular") -> int:
    return int(font(size, weight).getlength(text))


@lru_cache(maxsize=2048)
def _patch(text: str, size: int, weight: str, colour: tuple[int, int, int],
           pill: tuple[int, int, int] | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Rendered label: (BGR pixels, alpha 0..1, text width). Colours are BGR."""
    f = font(size, weight)
    tw = text_width(text, size, weight)
    pad = 7 if pill is not None else 2
    w, h = tw + 2 * pad, int(size * 1.35) + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if pill is not None:
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=min(9, h // 2), fill=(*pill, 255))
    d.text((pad, pad), text, font=f, fill=(*colour, 255))  # BGR values in "RGB" slots: stays BGR
    arr = np.asarray(im)
    return arr[..., :3].copy(), (arr[..., 3:4].astype(np.float32) / 255.0), tw


def _blit(img: np.ndarray, lab: "Label") -> None:
    rgb_, alpha, tw = _patch(lab.text, lab.size, lab.weight, tuple(lab.colour),
                             tuple(lab.pill) if lab.pill is not None else None)
    pad = 7 if lab.pill is not None else 2
    x = lab.x - pad - (tw // 2 if lab.anchor[0] == "m" else tw if lab.anchor[0] == "r" else 0)
    y = lab.y - pad
    h, w = rgb_.shape[:2]
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    src, a = rgb_[y1 - y:y2 - y, x1 - x:x2 - x], alpha[y1 - y:y2 - y, x1 - x:x2 - x]
    dst = img[y1:y2, x1:x2]
    dst[:] = (src * a + dst * (1.0 - a)).astype(np.uint8)


@lru_cache(maxsize=4)
def _background(h: int, w: int, ch: int, cw: int) -> np.ndarray:
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    cv2.rectangle(img, (cw, TOP_H), (w - 1, TOP_H + ch - 1), PANEL, -1)
    cv2.line(img, (0, TOP_H + ch), (w, TOP_H + ch), LINE, 1)
    return img


@lru_cache(maxsize=256)
def wrap(text: str, size: int, width: int, weight: str = "Regular", max_lines: int = 3) -> list[str]:
    return _wrap(text, size, width, weight, max_lines)


def _wrap(text: str, size: int, width: int, weight: str = "Regular", max_lines: int = 3) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if text_width(trial, size, weight) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,") + "..."
    return tuple(lines)


def _radar(img: np.ndarray, x0: int, y0: int, w: int, h: int, st: HudState, labels: list[Label]) -> None:
    cx, cy = x0 + w // 2, y0 + h - 10
    max_m = max(st.warn_m * 1.6, 4.0)
    px_per_m = min((h - 24) / max_m, (w / 2 - 10) / max_m)
    half = math.radians(st.hfov_deg / 2)
    fan = [(cx, cy)] + [(int(cx + math.sin(a) * max_m * px_per_m), int(cy - math.cos(a) * max_m * px_per_m))
                        for a in np.linspace(-half, half, 24)]
    cv2.fillPoly(img, [np.array(fan, np.int32)], (58, 54, 50), cv2.LINE_AA)
    for m in range(1, int(max_m) + 1):
        r = int(m * px_per_m)
        cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, LINE, 1, cv2.LINE_AA)
        if m in (1, 2, 3, 5):
            labels.append(Label(cx + 4, cy - r + 1, f"{m} m", MUTED, 11))
    for dist, colour, th in ((st.danger_m, RED, 2), (st.warn_m, AMBER, 1)):
        r = int(dist * px_per_m)
        cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, colour, th, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 6, WHITE, -1, cv2.LINE_AA)
    for info in st.radar:
        if info.distance_m is None or not info.confirmed:
            continue
        a = (info.x_norm - 0.5) * math.radians(st.hfov_deg)
        d = min(info.distance_m, max_m)
        px, py = int(cx + math.sin(a) * d * px_per_m), int(cy - math.cos(a) * d * px_per_m)
        danger = info.track_id in st.danger_ids
        colour = RED if danger else AMBER if info.distance_m <= st.warn_m else GREEN
        cv2.circle(img, (px, py), 8 if danger else 6, colour, -1, cv2.LINE_AA)
        if info.approach_mps >= 0.3:  # small arrow toward the user
            k = min(1.0, info.approach_mps / 2.0) * 20
            cv2.arrowedLine(img, (px, py), (int(px - math.sin(a) * k), int(py + math.cos(a) * k)), colour, 2,
                            cv2.LINE_AA, tipLength=0.4)


def compose(camera: np.ndarray, st: HudState) -> np.ndarray:
    """Full window image: the camera view (already drawn on, CAM_W x CAM_H) plus the HUD."""
    ch, cw = camera.shape[:2]
    W, H = cw + PANEL_W, TOP_H + ch + BOTTOM_H
    img = _background(H, W, ch, cw).copy()
    img[TOP_H:TOP_H + ch, :cw] = camera
    labels: list[Label] = list(st.labels)
    status_colour = STATUS_COLOURS.get(st.status, GREY)

    # ---- top bar: name, mode, big status, chips
    labels.append(Label(18, 16, "Occhio", WHITE, 24, "Bold"))
    mode_w = text_width(st.mode_label, 13, "Semibold") + 22
    _rounded(img, 118, 18, 118 + mode_w, 42, 12, AMBER if st.industrial else GREEN)
    labels.append(Label(118 + mode_w // 2, 22, st.mode_label, (25, 25, 25), 13, "Semibold", anchor="ma"))
    sw = max(200, text_width(st.status_label, 20, "Bold") + 60)
    sx = cw // 2 - sw // 2
    _rounded(img, sx, 9, sx + sw, 51, 21, status_colour)
    labels.append(Label(cw // 2, 17, st.status_label, (255, 255, 255), 20, "Bold", anchor="ma"))
    x = W - 16
    fps_text = f"{st.fps:.0f} FPS"
    labels.append(Label(x, 21, fps_text, GREEN if st.fps >= 20 else AMBER, 15, "Semibold", anchor="ra"))
    x -= text_width(fps_text, 15, "Semibold") + 14
    for label, colour in st.chips:
        cw_ = text_width(label, 12, "Semibold") + 18
        _rounded(img, x - cw_, 19, x, 41, 11, colour)
        labels.append(Label(x - cw_ // 2, 23, label, (25, 25, 25), 12, "Semibold", anchor="ma"))
        x -= cw_ + 8

    # ---- right panel
    px = cw
    pad, inner = 14, PANEL_W - 28
    y = TOP_H + 12
    # NOW card
    card_h = 108
    _rounded(img, px + pad, y, px + pad + inner, y + card_h, 12, CARD)
    cv2.rectangle(img, (px + pad, y + 8), (px + pad + 5, y + card_h - 8), RED if st.now_urgent else status_colour, -1)
    labels.append(Label(px + pad + 18, y + 10, st.titles.get("now", "NOW"), MUTED, 12, "Semibold"))
    for i, line in enumerate(wrap(st.now_text, 21, inner - 30, "Semibold", 3)):
        labels.append(Label(px + pad + 18, y + 32 + i * 25, line, WHITE, 21, "Semibold"))
    y += card_h + 14
    # radar
    labels.append(Label(px + pad, y, st.titles.get("radar", "RADAR"), MUTED, 12, "Semibold"))
    _radar(img, px + pad, y + 16, inner, 150, st, labels)
    y += 176
    # around you
    labels.append(Label(px + pad, y, st.titles.get("around", "AROUND YOU"), MUTED, 12, "Semibold"))
    y += 20
    if not st.nearby:
        labels.append(Label(px + pad, y, st.titles.get("nothing", "Nothing nearby"), MUTED, 15))
        y += 26
    for item in st.nearby[:4]:
        cv2.circle(img, (px + pad + 6, y + 10), 6, item.colour, -1, cv2.LINE_AA)
        labels.append(Label(px + pad + 20, y + 1, item.name, WHITE, 15, "Semibold"))
        labels.append(Label(px + pad + 130, y + 2, item.where, MUTED, 14))
        labels.append(Label(px + pad + inner, y + 1, f"{item.distance_m:.1f} m", WHITE, 15, "Semibold", anchor="ra"))
        bar = int(max(0.0, 1.0 - item.distance_m / max(st.warn_m * 1.6, 1.0)) * (inner - 20))
        cv2.rectangle(img, (px + pad + 20, y + 22), (px + pad + inner, y + 24), LINE, -1)
        cv2.rectangle(img, (px + pad + 20, y + 22), (px + pad + 20 + bar, y + 24), item.colour, -1)
        y += 32
    y += 6
    # recent
    labels.append(Label(px + pad, y, st.titles.get("recent", "RECENT"), MUTED, 12, "Semibold"))
    y += 20
    for stamp, text, urgent in reversed(st.recent[-6:]):
        if y > TOP_H + ch - 22:
            break
        line = wrap(text, 13, inner - 62, max_lines=1)[0] if text else ""
        labels.append(Label(px + pad, y, stamp[-8:-3] if len(stamp) >= 8 else stamp, MUTED, 12))
        labels.append(Label(px + pad + 48, y, line, (120, 120, 245) if urgent else (215, 215, 215), 13))
        y += 20

    # ---- bottom bar: keys + sound meter
    by = TOP_H + ch
    x = 16
    if st.hint:
        labels.append(Label(x, by + 16, st.hint, AMBER, 14, "Semibold"))
    else:
        for key, label in st.keys:
            _rounded(img, x, by + 13, x + 26, by + 39, 6, CARD)
            cv2.rectangle(img, (x, by + 13), (x + 26, by + 39), LINE, 1)
            labels.append(Label(x + 13, by + 16, key.upper(), WHITE, 14, "Bold", anchor="ma"))
            labels.append(Label(x + 33, by + 17, label, MUTED, 14))
            x += 33 + text_width(label, 14) + 18
    meter_w = 120
    mx = W - 16 - meter_w
    level = min(1.0, max(0.0, (st.sound_db + 70) / 60))
    cv2.rectangle(img, (mx, by + 22), (mx + meter_w, by + 30), CARD, -1)
    cv2.rectangle(img, (mx, by + 22), (mx + int(meter_w * level), by + 30), AMBER if level > 0.66 else GREEN, -1)
    labels.append(Label(mx - 8, by + 17, st.sound_last or st.titles.get("sound", "Sound"),
                        AMBER if st.sound_last else MUTED, 13, anchor="ra"))
    if st.details:
        labels.append(Label(12, TOP_H + ch - 24, st.details, (200, 200, 200), 12, pill=(0, 0, 0)))

    # ---- text: cached patches blended on top
    for lab in labels:
        _blit(img, lab)
    return img


def track_label(x1: int, y1: int, text: str, colour: tuple[int, int, int], frame_h: int) -> Label:
    """A label pill just above a box (inside the frame), in window coordinates."""
    y = max(TOP_H + 4, TOP_H + y1 - 26)
    if y1 < 30:
        y = TOP_H + y1 + 6
    return Label(x1 + 6, y, text, (255, 255, 255), 14, "Semibold", pill=colour)


def to_camera_norm(x: int, y: int, cam_shape: tuple[int, int]) -> tuple[float, float] | None:
    """Window pixel -> (x, y) as fractions of the camera image, or None if outside it."""
    ch, cw = cam_shape
    if not (0 <= x < cw and TOP_H <= y < TOP_H + ch):
        return None
    return x / cw, (y - TOP_H) / ch


def draw_light(img: np.ndarray, box: tuple[float, float, float, float], colour_name: str | None,
               scale: float) -> tuple[int, int, tuple[int, int, int]]:
    """Box around the traffic light; returns where to put its label and the colour."""
    x1, y1, x2, y2 = (int(round(c * scale)) for c in box)
    colour = RED if colour_name == "red" else GREEN if colour_name == "green" else GREY
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3, cv2.LINE_AA)
    return x1, y1, colour


def draw_zone(img: np.ndarray, points, closed: bool, active: bool) -> None:
    pts = np.array(list(points), np.int32)
    if len(pts) == 0:
        return
    colour = RED if active else AMBER
    if closed and len(pts) >= 3:
        layer = img.copy()
        cv2.fillPoly(layer, [pts.reshape(-1, 1, 2)], colour)
        cv2.addWeighted(layer, 0.25 if active else 0.12, img, 0.75 if active else 0.88, 0, dst=img)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], closed, colour, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, (int(p[0]), int(p[1])), 4, colour, -1)
