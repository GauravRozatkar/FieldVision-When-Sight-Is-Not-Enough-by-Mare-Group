"""Occhio entry point.

    python main.py --mode street --source 0
    python main.py --source demo.mp4 --simulate-keys "18.6:d,20.4:m"    # rehearse on a video
    python main.py --source 0 --record walk.mp4                         # record a test video
    python main.py --mode street --source 0 --calibrate                 # stand at 2 m, press c

Keys in the window: d describe the scene, t read text, v voice command, m street/industrial,
q quit (c saves the calibration with --calibrate). Industrial mode: left click draws the danger
zone, right click closes / clears it. The app itself lives in occhio/app.py.
"""
from __future__ import annotations

import argparse
import sys

from config.profiles import DEFAULT_MODE, LANGUAGES, PROFILES
from config.settings import METRICS
from occhio.app import App, parse_key_script  # noqa: F401  (parse_key_script re-exported for tests)


def parse_source(value: str) -> int | str:
    """'0' / '1' -> camera number; 'iphone' -> found by name later; anything else -> video file path."""
    return int(value) if value.isdigit() else value


def resolve_source(source: int | str) -> int | str | None:
    """'iphone' (or any camera name part after 'cam:') -> its camera number; None if not found."""
    from occhio.cameras import find_camera, list_cameras

    if isinstance(source, str) and (source.lower() == "iphone" or source.lower().startswith("cam:")):
        name = "iphone" if source.lower() == "iphone" else source[4:]
        cameras = list_cameras()
        index = find_camera(name, cameras)
        if index is None:
            print(f"[occhio] no camera named like {name!r}. Cameras found: "
                  + (", ".join(f"{i} = {n}" for i, n in cameras) or "none (is ffmpeg installed?)"), file=sys.stderr)
            if name == "iphone":
                print("[occhio] iPhone checklist: iOS 16+ and macOS 13+, same Apple ID as the Mac, Wi-Fi and Bluetooth on, "
                      "iPhone locked, near the Mac and not in use by another app. Then try again.", file=sys.stderr)
            return None
        print(f"[occhio] using camera {index}: {dict(cameras)[index]}", flush=True)
        return index
    return source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occhio - real-time assistive vision")
    parser.add_argument("--mode", choices=sorted(PROFILES), default=DEFAULT_MODE,
                        help="profile to use (default: %(default)s)")
    parser.add_argument("--source", type=parse_source, default=0,
                        help="camera number (0 = built-in), 'phone' (iPhone via Safari, no app needed), "
                             "'iphone' (Continuity Camera by name), "
                             "'cam:NAME', or a video file path")
    parser.add_argument("--list-cameras", action="store_true", help="show the cameras with their numbers and exit")
    parser.add_argument("--phone-port", type=int, default=8443, help="port of the phone page (--source phone)")
    parser.add_argument("--mac-voice", action="store_true", help="with --source phone, also speak on the Mac")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="rotate the camera image, e.g. 90 for an iPhone worn upright on the chest")
    parser.add_argument("--no-voice", action="store_true", help="disable speech and beeps")
    parser.add_argument("--no-beeps", action="store_true", help="keep speech, no beeps (key b toggles them)")
    parser.add_argument("--lang", choices=LANGUAGES, default=None,
                        help="language of voice, screen and AI answers (default: English)")
    parser.add_argument("--no-depth", action="store_true",
                        help="skip the depth model; distance from box size only")
    parser.add_argument("--no-sound", action="store_true",
                        help="do not listen to the microphone for horns / sirens / approaching noise")
    parser.add_argument("--calibrate", type=float, nargs="?", const=2.0, default=None, metavar="METRES",
                        help="calibration mode: stand at METRES (default 2) from the camera and press c")
    parser.add_argument("--simulate-keys", default="", metavar="T:K,...",
                        help="press keys automatically for tests/rehearsal, e.g. '2:d,5:m' (seconds:key)")
    parser.add_argument("--record", default=None, metavar="FILE.mp4",
                        help="also save the raw camera video, to replay it later with --source FILE.mp4")
    parser.add_argument("--metrics", default=METRICS["file"], metavar="FILE.csv",
                        help="write one row per second of FPS/latency/warnings ('' to disable)")
    parser.add_argument("--llm", choices=["auto", "claude", "local", "none"], default="auto",
                        help="describe/read backend: auto = Claude if ANTHROPIC_API_KEY is set, "
                             "else the offline SmolVLM if downloaded, else detections only")
    parser.add_argument("--vibrate", type=int, nargs="?", const=8765, default=None, metavar="PORT",
                        help="serve a page that makes a phone on the same Wi-Fi vibrate (Android Chrome)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cameras:
        from occhio.cameras import list_cameras
        cameras = list_cameras()
        for index, name in cameras:
            print(f"{index}: {name}")
        if not cameras:
            print("no cameras found (the names come from ffmpeg: brew install ffmpeg)")
        return 0
    source = resolve_source(args.source)
    if source is None:
        return 1
    args.source = source
    return App(args).run()


if __name__ == "__main__":
    sys.exit(main())
