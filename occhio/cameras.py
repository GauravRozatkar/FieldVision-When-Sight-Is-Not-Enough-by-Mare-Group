"""Find cameras by name (macOS), e.g. the iPhone used through Continuity Camera.

OpenCV opens cameras by number, and the numbers are not fixed: on this Mac
1 is the "Desk View" camera, and an iPhone can appear as 0, 1, 2 or 3 depending
on what is connected. ffmpeg's AVFoundation listing gives the names in the same
order OpenCV uses, so `--source iphone` can pick the right number.
Needs the `ffmpeg` command (Homebrew); without it, use --list-cameras and a number.
"""
from __future__ import annotations

import re
import shutil
import subprocess

_LINE = re.compile(r"\[(\d+)\]\s+(.+)$")


def parse_avfoundation(output: str) -> list[tuple[int, str]]:
    """Video devices from `ffmpeg -f avfoundation -list_devices true -i ""` output (screens excluded)."""
    cameras: list[tuple[int, str]] = []
    in_video = False
    for raw in output.splitlines():
        if "AVFoundation video devices" in raw:
            in_video = True
            continue
        if "AVFoundation audio devices" in raw:
            break
        if in_video:
            m = _LINE.search(raw.strip())
            if m and not m.group(2).startswith("Capture screen"):
                cameras.append((int(m.group(1)), m.group(2).strip()))
    return cameras


def list_cameras() -> list[tuple[int, str]]:
    """(number, name) of every camera, or [] when ffmpeg is not installed."""
    if shutil.which("ffmpeg") is None:
        return []
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                          capture_output=True, text=True, timeout=10)
    return parse_avfoundation(proc.stderr + proc.stdout)


def find_camera(name_part: str, cameras: list[tuple[int, str]] | None = None) -> int | None:
    """Number of the first camera whose name contains `name_part` (case-insensitive)."""
    for index, name in (list_cameras() if cameras is None else cameras):
        if name_part.lower() in name.lower():
            return index
    return None
