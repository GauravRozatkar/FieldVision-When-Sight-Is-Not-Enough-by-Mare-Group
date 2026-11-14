"""The Occhio app: builds every part of the pipeline and runs the main loop.

Per frame: capture -> (eco gate) -> detect + track -> depth (background) ->
spatial -> reliability -> priority + signals (traffic light, step, zone) ->
speech / beeps / phone vibration -> HUD window.

Things that finish on other threads (describe / read with the LLM, voice
commands, sounds from the microphone) are queued in `self._pending` and handled
on the main thread, so all decisions and all speech come from one place.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import time
from collections import deque
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

from config.profiles import (OBJECTS, SIDE_PHRASE, SOUND_WHERE, UI_TEXT, detect_classes, get_profile,
                             lines_for, next_mode, render)
from config.settings import CAMERA, CALIBRATION_FILE, DETECTOR, RELIABILITY_PHONE, SPATIAL, TALK
from occhio.capture import CameraStream, VideoRecorder
from occhio.depth import DepthEstimator, load_depth_anything
from occhio.detector import Detector
from occhio.free_space import FreeSpace
from occhio.ground import StepDetector
from occhio.hud import (AMBER, BLUE, CAM_H, CAM_W, GREEN, RED, YELLOW, HudState, Label, Nearby, compose,
                        draw_light, draw_zone, fit_geometry, to_camera_norm, track_label)
from occhio.metrics import FpsCounter, MetricsLogger, StageTimes
from occhio.overlay import GREY, _dashed_rect, draw_corridor
from occhio.phone_link import PhoneCamera
from occhio.power import MotionGate
from occhio.priority import BeepPolicy, PriorityPlanner
from occhio.reliability import ReliabilityMonitor, ReliabilityState
from occhio.scene_llm import SceneDescriber, anthropic_call
from occhio.scene_local import describe_detections
from occhio.vlm_local import LOCAL_PROMPTS, LocalVLM, choose_backend, is_cached
from occhio.sound import SoundMonitor
from occhio.spatial import Calibration, SpatialInfo, SpatialReasoner, corridor_polygon
from occhio.speech import Beeper, Speaker
from occhio.tracks import TrackManager, TrackStatus
from occhio.traffic_light import TrafficLightMonitor
from occhio.vibration import VibrationServer
from occhio.voice_cmd import VoiceCommander
from occhio.zone import DangerZone

WINDOW_NAME = "Occhio"
WARNING_KINDS = {"danger", "ttc", "approach", "ahead", "side", "zone", "step", "sound"}


def is_no_text(answer: str) -> bool:
    """Did the reader say there is nothing to read? ('No text.', 'Nessun testo', 'None', ...)"""
    a = "".join(c for c in answer.lower() if c.isalnum() or c == " ").strip()
    return a in {"", "no", "none", "no text", "nessun testo", "no text found", "there is no text"} \
        or a.startswith(("no text", "there is no text", "nessun testo"))


def parse_key_script(spec: str) -> list[tuple[float, str]]:
    """'2:d,5.5:m' -> [(2.0, 'd'), (5.5, 'm')], sorted by time."""
    events = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        when, key = part.split(":", 1)
        events.append((float(when), key[:1]))
    return sorted(events)


def calibrate(spatial: SpatialReasoner, depth: DepthEstimator | None, tracks, frame_shape: tuple[int, int],
              metres: float) -> str:
    """Save depth scale and focal length from the biggest confirmed person at `metres`."""
    people = [t for t in tracks if t.cls_name == "person" and t.visible and t.status is TrackStatus.CONFIRMED]
    if not people:
        return "calibration: no confirmed person in view"
    person = max(people, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
    x1, y1, x2, y2 = person.box
    _, fw = frame_shape
    if spatial.is_full_body_person("person", person.box, frame_shape):
        focal = metres * (y2 - y1) / float(SPATIAL["heights_m"]["person"])
    else:
        focal = metres * (x2 - x1) / float(SPATIAL["person_width_m"])
    disparity = depth.latest.median_in_box(person.box) if depth and depth.latest else None
    cal = Calibration(depth_k=metres * disparity if disparity else None, focal_px=focal, frame_width=fw, fixed=True)
    cal.save()
    spatial.calibration = cal
    return (f"calibration saved to {CALIBRATION_FILE.name}: focal {focal:.0f}px"
            + (f", depth_k {cal.depth_k:.3f}" if cal.depth_k else ", no depth"))


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        load_dotenv()  # ANTHROPIC_API_KEY, OCCHIO_LLM_MODEL from .env (never committed)
        self.args = args
        self.mode = args.mode
        self.profile = get_profile(self.mode)
        self.lang = args.lang or self.profile["lang"]
        self.ui = UI_TEXT[self.lang]

        print("[occhio] loading detector...", flush=True)
        size = getattr(args, "model", "auto")
        size = ("s" if args.source == "phone" else "n") if size == "auto" else size
        model_path = DETECTOR["models"][size]
        if size == "s" and not Path(model_path).exists():
            print("[occhio] yolo11s.pt not found: using yolo11n", flush=True)
            size, model_path = "n", DETECTOR["models"]["n"]
        self.detector_size = size
        self.detector = Detector(detect_classes(self.profile), {"model": model_path})
        self.tracks_mgr = TrackManager()
        self.depth: DepthEstimator | None = None
        if not args.no_depth:
            print("[occhio] loading depth model...", flush=True)
            try:
                self.depth = DepthEstimator(load_depth_anything()).start()
            except Exception as exc:  # no model / no network: box-size distance still works
                print(f"[occhio] depth disabled: {exc!r}", file=sys.stderr, flush=True)
        self.calibration = Calibration() if args.calibrate is not None else Calibration.load()
        self.spatial = SpatialReasoner(self.profile, calibration=self.calibration)
        self.reliability = ReliabilityMonitor(RELIABILITY_PHONE if args.source == "phone" else None)
        self.planner = PriorityPlanner(self.profile, self.lang)
        self.traffic = TrafficLightMonitor()
        self.step = StepDetector()
        self.free_space = FreeSpace() if (args.source == "phone" or getattr(args, "walls", False)) else None
        self.zone = DangerZone()
        self.gate = MotionGate()
        self.metrics = MetricsLogger(args.metrics or None)

        self.event_log: deque[tuple[str, str, bool]] = deque(maxlen=int(TALK["event_log_size"]))
        self.caption = ("", False)
        self.sound_last = ""
        self._pending: deque[tuple[str, Any, Any]] = deque()  # filled by other threads

        # With --source phone the phone speaks and beeps (the Mac may be in a bag); --mac-voice adds the Mac.
        self.phone_mode = args.source == "phone"
        self.phone: PhoneCamera | None = None
        voice_wanted = not args.no_voice
        mac_voice = voice_wanted and (not self.phone_mode or getattr(args, "mac_voice", False))
        self.mac_voice = mac_voice
        # Phone mode: the Mac also speaks until the phone reports that its own voice works.
        self.speaker = Speaker(self.lang, enabled=voice_wanted, on_spoken=self._on_spoken).start()
        self.beeper = Beeper(enabled=mac_voice).start()
        self.beeps_on = voice_wanted and not getattr(args, "no_beeps", False)  # 'b' toggles
        self.beep_policy = BeepPolicy(self.profile)
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        self.llm_backend = choose_backend(getattr(args, "llm", "auto"), has_key, is_cached())
        on_llm = lambda text, ok: self._pending.append(("llm", text, ok))  # noqa: E731
        if self.llm_backend == "local":
            call = LocalVLM().start_loading()  # ~15 s in the background, ready before the first 'd'
            self.describer = SceneDescriber(self.profile, self.lang, on_llm, call=call,
                                            prompt_for=lambda kind, lang: LOCAL_PROMPTS[kind][lang])
        else:
            self.describer = SceneDescriber(self.profile, self.lang, on_llm,
                                            call=anthropic_call() if self.llm_backend == "claude" else None)
        self._llm_kind = "describe"
        self.voice = VoiceCommander(self.lang, lambda intent, text: self._pending.append(("voice", intent, text)),
                                    say=lambda text: self.speaker.say(text, urgent=True),
                                    listening_line=lines_for(self.profile, self.lang)["listening"])
        self.sound = SoundMonitor(lambda kind, side: self._pending.append(("sound", kind, side)),
                                  enabled=not args.no_sound and not self.phone_mode,  # Mac mic is in the bag
                                  is_self_audio=lambda: self.beeper.target is not None or self.speaker.busy).start()
        self.vibration = VibrationServer(args.vibrate).start() if args.vibrate else None
        self.recorder = VideoRecorder(args.record) if args.record else None

        self.tracks: list = []
        self.infos: list = []
        self._frame_shape = (720, 1280)
        self.rel = ReliabilityState(1.0, 1.0, 0.0, 0.0, 0.0, None, False, True, "ok")
        self.show_details = False  # 'i': ids, confidence and timings on screen
        self._announced_signs: set[int] = set()

    # ------------------------------------------------------------ speaking
    def _on_spoken(self, text: str, urgent: bool) -> None:  # Speaker callback (any thread)
        self.event_log.append((datetime.now().strftime("%H:%M:%S"), text, urgent))
        self.caption = (text, urgent)
        if self.phone is not None and not self.args.no_voice:
            self.phone.say(text, urgent)
        print(f"[voice]{' (urgent)' if urgent else ''} {text}", flush=True)

    def say(self, text: str, urgent: bool = False, kind: str = "", track_id: int | None = None,
            now: float = 0.0) -> None:
        self.speaker.say(text, urgent=urgent)
        if kind in WARNING_KINDS:
            self.metrics.warning(track_id, now)

    def line(self, key: str, **kw: Any) -> str:
        return render(self.profile, self.lang, key, **kw)

    # -------------------------------------------------------- mode / input
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.profile = get_profile(mode)
        self.detector.set_classes(detect_classes(self.profile))
        self.spatial.set_profile(self.profile)
        self.planner.set_profile(self.profile)
        self.beep_policy.set_profile(self.profile)
        self.describer.set_profile(self.profile)
        self.traffic.reset()
        self.voice.listening_line = lines_for(self.profile, self.lang)["listening"]
        self.beeper.set_target(None)
        self.say(self.line("mode"), urgent=True)
        print(f"[occhio] mode -> {mode}", flush=True)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if not self.profile.get("danger_zone"):
            return
        pos = to_camera_norm(x, y, (CAM_H, CAM_W))
        if event == cv2.EVENT_LBUTTONDOWN and pos is not None:
            s, ox, oy, w, h = fit_geometry(*self._frame_shape)  # window -> frame fractions
            fx, fy = (pos[0] * CAM_W - ox) / w, (pos[1] * CAM_H - oy) / h
            if 0 <= fx <= 1 and 0 <= fy <= 1:
                self.zone.add_point(fx, fy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.zone.close_or_clear()

    def handle_key(self, key: int, image: np.ndarray) -> bool:
        """Returns False to quit."""
        if key == ord("q"):
            return False
        if key == ord("d"):
            self.describe(image)
        elif key == ord("t"):
            self.read_text(image)
        elif key == ord("v"):
            self.sound.paused.set()  # the microphone is busy with the command
            if self.voice.press():
                print("[occhio] voice command: listening...", flush=True)
        elif key == ord("m"):
            self.set_mode(next_mode(self.mode))
        elif key == ord("i"):
            self.show_details = not self.show_details
        elif key == ord("b"):
            self.beeps_on = not self.beeps_on
            if not self.beeps_on:
                self.beeper.set_target(None)
            print(f"[occhio] beeps {'on' if self.beeps_on else 'off'}", flush=True)
        elif key == ord("c") and self.args.calibrate is not None:
            print(f"[occhio] {calibrate(self.spatial, self.depth, self.tracks, self._frame_shape, self.args.calibrate)}",
                  flush=True)
        return True

    def handle_pending(self, image: np.ndarray, now: float) -> None:
        while self._pending:
            source, a, b = self._pending.popleft()
            if source == "llm":
                if b:
                    print(f"[occhio] {self.llm_backend} answered in {self.describer.last_latency_s:.1f}s", flush=True)
                    if self._llm_kind == "read":
                        lines = lines_for(self.profile, self.lang)
                        self.say(lines["read_none"] if is_no_text(a) else lines["read_result"].format(text=a),
                                 urgent=True)
                    else:
                        self.say(a)  # after the instant summary, not over it
                else:
                    print(f"[occhio] {self.llm_backend} failed: {self.describer.last_error}", flush=True)
                    if self._llm_kind == "read":  # 'describe' already spoke the instant summary
                        self.say(a, urgent=True)
            elif source == "sound":
                # Hearing works even when the camera is unreliable, so no reliability check here.
                where = SOUND_WHERE[self.lang][b or ""]
                text = lines_for(self.profile, self.lang)[f"sound_{a}"].format(where=where)
                self.sound_last = f"{self.ui[a]}{where}"
                self.say(text, urgent=a in ("horn", "siren"), kind="sound", now=now)
            elif source == "voice":
                self.sound.paused.clear()
                print(f"[occhio] voice command: {a!r} ({b!r})", flush=True)
                if a == "describe":
                    self.describe(image)
                elif a == "read":
                    self.read_text(image)
                elif a == "mode_industrial" and self.mode != "industrial":
                    self.set_mode("industrial")
                elif a == "mode_street" and self.mode != "street":
                    self.set_mode("street")
                elif a == "repeat" and self.caption[0]:
                    self.say(self.caption[0], urgent=True)
                elif a == "unavailable":
                    self.say(self.line("voice_unavailable"), urgent=True)
                elif a not in ("mode_industrial", "mode_street", "repeat"):
                    self.say(self.line("command_unknown"), urgent=True)

    def describe(self, image: np.ndarray) -> None:
        """Instant summary from the detections, then (if available) a vision model's sentence."""
        light = self.traffic.colour if self.traffic.box is not None else None
        self.say(describe_detections(self.infos, self.profile, self.lang, light), urgent=True)
        if self.llm_backend != "none":
            self._llm_kind = "describe"
            if self.describer.request(image, "describe"):
                print(f"[occhio] describing scene with {self.llm_backend}...", flush=True)

    def read_text(self, image: np.ndarray) -> None:
        if self.llm_backend == "none":
            self.say(self.line("read_failed"), urgent=True)
            print("[occhio] reading needs ANTHROPIC_API_KEY or the local model: see README", flush=True)
            return
        self._llm_kind = "read"
        if self.describer.request(image, "read"):
            print(f"[occhio] reading text with {self.llm_backend}...", flush=True)

    # ------------------------------------------------------------- main loop
    def run(self) -> int:
        args = self.args
        try:
            if self.phone_mode:
                self.phone = PhoneCamera(port=getattr(args, "phone_port", 8443), lang=self.lang, strings={
                    "ready": self.ui["phone_ready"], "tap": self.ui["phone_tap"], "start": self.ui["phone_start"],
                    "test_voice": self.ui["test_voice"], "voice_on": self.ui["voice_on"],
                }).start()
                stream = self.phone
                print(f"[occhio] PHONE: open {self.phone.url} in Safari on the iPhone (same Wi-Fi or its hotspot), "
                      "accept the certificate warning once, tap Start", flush=True)
            else:
                stream = CameraStream(args.source, rotate=getattr(args, "rotate", 0)).start()
        except RuntimeError as exc:
            print(f"[occhio] {exc}", file=sys.stderr)
            self.close()
            return 1
        print(f"[occhio] mode={self.mode} lang={self.lang} source={args.source!r} device={self.detector.device} "
              f"model=yolo11{self.detector_size} "
              f"depth={'on' if self.depth else 'off'} voice={'off' if args.no_voice else 'on'} "
              f"beeps={'on' if self.beeper.enabled else 'off'} sound={'on' if self.sound.enabled else 'off'}"
              f"{' (stereo)' if self.sound.stereo else ''} llm={self.llm_backend}"
              f"{' (' + str(getattr(self.describer._call, 'model', '')) + ')' if self.llm_backend != 'none' else ''}",
              flush=True)
        print("[occhio] keys: d describe, t read, v voice command, m mode, b beeps, i details, q quit"
              + (" | danger zone: left click corners, right click close/clear" if self.profile.get("danger_zone")
                 else ""), flush=True)
        if self.vibration:
            print(f"[occhio] phone vibration: open {self.vibration.url} on a phone on the same Wi-Fi", flush=True)
        if args.calibrate is not None:
            print(f"[occhio] CALIBRATION: stand {args.calibrate:g} m from the camera, press c", flush=True)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        fps_counter, stages = FpsCounter(), StageTimes()
        last_report = time.perf_counter()
        shown = 0
        key_script = parse_key_script(args.simulate_keys)
        started = time.perf_counter()
        try:
            while True:
                frame = stream.read(timeout=0.2)
                if frame is None:
                    if not stream.running:
                        break
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                    continue
                if not self.step_frame(frame, stages, fps_counter):
                    break
                key = cv2.waitKey(1) & 0xFF
                if key_script and time.perf_counter() - started >= key_script[0][0]:
                    key = ord(key_script.pop(0)[1])
                if key != 255 and not self.handle_key(key, frame.image):
                    break

                now = time.perf_counter()
                stages.add("total", (now - frame.timestamp) * 1000.0)
                shown += 1
                if now - last_report >= 1.0:
                    self.report(stages, fps_counter.fps, stream.frames_grabbed - shown, frame.timestamp)
                    last_report = now
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            self.close()
        return 0

    def step_frame(self, frame, stages: StageTimes, fps_counter: FpsCounter) -> bool:
        image, now = frame.image, frame.timestamp
        self._frame_shape = image.shape[:2]
        if self.recorder:
            self.recorder.write(image)
        t_start = time.perf_counter()
        stages.add("capture", (t_start - now) * 1000.0)

        tracking = any(t.visible for t in self.tracks)
        if self.gate.should_detect(image, now, tracking):
            detections = self.detector.detect(image)
            t_detect = time.perf_counter()
            self.tracks = self.tracks_mgr.update(detections, now, image)
            for event in self.tracks_mgr.events:
                print(f"[occhio] {event}", flush=True)
        else:
            t_detect = time.perf_counter()
        stages.add("detect", (t_detect - t_start) * 1000.0)

        if self.depth:
            self.depth.submit(image, now)
        depth_frame = self.depth.latest if self.depth else None
        if self.phone is not None:
            self.speaker.enabled = not self.args.no_voice and (self.mac_voice or not self.phone.phone_voice_ok)
            self.spatial.pitch_deg = self.phone.pitch_deg
            self.spatial.user_speed_mps = self.phone.walk_speed
            self.planner.extra_warn_m = min(1.5, self.phone.walk_speed * 1.0)
        self.infos = self.spatial.update(self.tracks, depth_frame, self._frame_shape, now)
        if self.free_space is not None and depth_frame is not None:
            wall = self.free_space.update(depth_frame, self._frame_shape, self.spatial.floor_distance_at_row,
                                          fallback_scale=self.spatial.calibration.depth_k)
            nearer = any(i.confirmed and i.in_corridor and i.distance_m is not None and i.distance_m <= (wall or 0) + 0.5
                         for i in self.infos)
            if wall is not None and not nearer:  # something big in the path the detector has no name for
                approach = 0.8 * self.spatial.user_speed_mps
                self.infos.append(SpatialInfo(
                    track_id=-1, cls_name="obstacle", confirmed=True, mean_conf=0.8, side="ahead", clock=12,
                    distance_m=wall, distance_source="depth", approach_mps=approach, lateral_mps=0.0,
                    speed_mps=approach, fast=False, ttc_s=wall / approach if approach >= 0.3 else None,
                    in_corridor=True, x_norm=0.5))
        rel = self.reliability.update(image, self.tracks)
        self.rel = rel
        message = self.planner.plan(self.infos, rel, now)
        if message is not None:
            self.say(message.text, message.urgent, message.kind, message.track_id, now)

        if rel.reliable:  # signals only when the camera can be trusted
            if self.profile.get("signal_classes"):
                colour = self.traffic.update(self.tracks, image)
                if colour:
                    self.say(self.line(f"light_{colour}"), kind="light", now=now)
            for t in self.tracks:
                if (t.cls_name == "stop sign" and t.visible and t.status is TrackStatus.CONFIRMED
                        and t.track_id not in self._announced_signs):
                    self._announced_signs.add(t.track_id)
                    self.say(self.line("stop_sign"), kind="light", now=now)
            if self.profile.get("step_warning") and depth_frame is not None:
                blocked = any(i.confirmed and i.in_corridor and i.distance_m is not None and i.distance_m < 1.5
                              for i in self.infos)
                if self.step.update(depth_frame, now, blocked):
                    self.say(self.line("step"), urgent=True, kind="step", now=now)
            if self.profile.get("danger_zone"):
                for track_id in self.zone.check(self.tracks, self._frame_shape, now):
                    self.say(self.line("zone"), urgent=True, kind="zone", track_id=track_id, now=now)
        self.handle_pending(image, now)

        target = self.beep_policy.update(self.infos, rel, now) if self.beeps_on else None
        self.beeper.set_target(*target) if target else self.beeper.set_target(None)
        if self.phone is not None:
            self.phone.set_beep(target)
        self._beeping = target is not None
        danger_ids = {i.track_id for i in self.infos
                      if i.confirmed and i.distance_m is not None and i.cls_name in self.profile["classes"]
                      and self.planner.is_danger(i)}
        if self.vibration:
            level = 3 if danger_ids else 0 if target is None else 2 if target[0] >= 3.5 else 1
            self.vibration.set(level, self.caption[0])
        self.metrics.distances(((i.track_id, i.distance_m) for i in self.infos if i.confirmed),
                               float(self.profile["danger_distance_m"]), now)
        stages.add("logic", (time.perf_counter() - t_detect) * 1000.0)

        self.draw(image, danger_ids, fps_counter.tick(), stages)
        return True

    def _object_name(self, cls_name: str) -> str:
        name = OBJECTS[self.lang].get(cls_name, cls_name)
        return name[:1].upper() + name[1:]

    def draw(self, image: np.ndarray, danger_ids: set[int], fps: float, stages: StageTimes) -> None:
        fh, fw = image.shape[:2]
        scale, ox, oy, vw, vh = fit_geometry(fh, fw)  # portrait phones get side bars, never stretched
        if (vw, vh) == (CAM_W, CAM_H):
            view = cv2.resize(image, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)  # AREA costs ~2-5 ms here
        else:
            view = np.full((CAM_H, CAM_W, 3), 16, dtype=np.uint8)
            view[oy:oy + vh, ox:ox + vw] = cv2.resize(image, (vw, vh), interpolation=cv2.INTER_LINEAR)

        def to_view(points):
            return [(int(round(x * scale)) + ox, int(round(y * scale)) + oy) for x, y in points]

        warn = float(self.profile["warn_distance_m"])
        draw_corridor(view, to_view(corridor_polygon(self.profile["corridor"], (fh, fw))), colour=(210, 180, 120),
                      alpha=0.07, thickness=1)
        if self.profile.get("danger_zone"):
            draw_zone(view, to_view(self.zone.pixels((fh, fw))), self.zone.closed, bool(self.zone.inside))

        labels: list[Label] = []
        info_by_id = {i.track_id: i for i in self.infos}
        obstacles = set(self.profile["classes"])
        for t in self.tracks:
            if t.cls_name not in obstacles:
                continue
            bx1, by1, bx2, by2 = t.box
            x1, y1, x2, y2 = (int(round(bx1 * scale)) + ox, int(round(by1 * scale)) + oy,
                              int(round(bx2 * scale)) + ox, int(round(by2 * scale)) + oy)
            x1, x2, y1, y2 = max(ox, x1), min(ox + vw - 1, x2), max(oy, y1), min(oy + vh - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            info = info_by_id.get(t.track_id)
            if t.status is TrackStatus.HIDDEN:
                _dashed_rect(view, (x1, y1), (x2, y2), GREY)
                continue
            if not t.visible:
                continue
            if t.status is not TrackStatus.CONFIRMED:  # uncertain: thin, silent, no label
                cv2.rectangle(view, (x1, y1), (x2, y2), YELLOW, 1, cv2.LINE_AA)
                if self.show_details:
                    labels.append(track_label(x1, y1, f"#{t.track_id} {t.cls_name} {t.conf:.2f}", (40, 120, 140), CAM_H))
                continue
            near = info is not None and info.distance_m is not None and info.distance_m <= warn
            colour = RED if t.track_id in danger_ids else AMBER if near else GREEN
            cv2.rectangle(view, (x1, y1), (x2, y2), colour, 3 if t.track_id in danger_ids else 2, cv2.LINE_AA)
            text = self._object_name(t.cls_name)
            if info is not None and info.distance_m is not None:
                text += f"  {info.distance_m:.1f} m"
                if info.approach_mps >= 0.3:
                    text += "  ↓"
            if self.show_details:
                text += f"   #{t.track_id} {t.conf:.2f}"
            labels.append(track_label(x1, y1, text, colour, CAM_H))
        wall = next((i for i in self.infos if i.track_id == -1), None)
        if wall is not None and wall.distance_m is not None:  # unnamed obstacle (wall): label at the bottom
            colour = RED if -1 in danger_ids else AMBER
            labels.append(track_label(ox + vw // 2 - 70, oy + vh - 70,
                                      f"{self._object_name('obstacle')}  {wall.distance_m:.1f} m", colour, CAM_H))
        if self.traffic.box is not None:
            lb = self.traffic.box
            lx, ly, lcol = draw_light(view, (lb[0] * scale + ox, lb[1] * scale + oy, lb[2] * scale + ox,
                                             lb[3] * scale + oy), self.traffic.colour, 1.0)
            if self.traffic.colour:
                labels.append(track_label(lx, ly, self.ui[self.traffic.colour], lcol, CAM_H))

        rel = self.rel
        if not rel.reliable:
            status = "unreliable"
        elif danger_ids:
            status = "danger"
        elif any(i.confirmed and i.cls_name in obstacles and i.distance_m is not None and i.distance_m <= warn
                 for i in self.infos):
            status = "caution"
        else:
            status = "clear"
        if self.phone is not None:
            self.phone.set_status(status, self.ui[f"status_{status}"])
        nearby = []
        for i in sorted((i for i in self.infos if i.confirmed and i.cls_name in obstacles and i.distance_m is not None),
                        key=lambda i: i.distance_m)[:4]:
            colour = RED if i.track_id in danger_ids else AMBER if i.distance_m <= warn else GREEN
            nearby.append(Nearby(self._object_name(i.cls_name), SIDE_PHRASE[self.lang][i.side], i.distance_m, colour))

        chips: list[tuple[str, tuple[int, int, int]]] = []
        if self.recorder:
            chips.append((self.ui["rec"], RED))
        if self.vibration:
            chips.append((self.ui["phone"], BLUE))
        chips.append((self.ui["mic"], BLUE) if self.sound.enabled else (self.ui["mic_off"], GREY))
        if not self.beeps_on:
            chips.append((self.ui["beeps_off"], GREY))
        if self.gate.eco:
            chips.append((self.ui["eco"], GREY))
        if self.phone is not None and self.phone.walk_speed > 0.2:
            chips.append((f'{self.ui["walking"]} {self.phone.walk_speed:.1f} m/s', GREEN))
        if self.describer.busy:
            chips.append((self.ui["describing"], BLUE))
        if self.voice.state in ("listening", "thinking"):
            chips.append((self.ui[self.voice.state], YELLOW))

        details = ""
        if self.show_details:
            sm = stages.smoothed
            details = ("  ".join(f"{k} {v:.0f} ms" for k, v in sm.items())
                       + f"   rel {rel.score:.2f} {rel.reason}   tracks {len(self.tracks)}"
                       + (f"   depth {self.depth.latest.infer_ms:.0f} ms" if self.depth and self.depth.latest else ""))
        caption, urgent = self.caption
        u = self.ui
        st = HudState(
            mode_label=u[self.mode], industrial=self.mode == "industrial", status=status,
            status_label=u[f"status_{status}"], now_text=caption or u["waiting"], now_urgent=urgent, fps=fps,
            chips=chips, radar=[i for i in self.infos if i.cls_name in obstacles], danger_ids=danger_ids,
            nearby=nearby, recent=list(self.event_log), warn_m=warn,
            danger_m=float(self.profile["danger_distance_m"]), hfov_deg=float(CAMERA["hfov_deg"]),
            sound_db=self.sound.analyzer.db if self.sound.enabled else -120.0, sound_last=self.sound_last,
            keys=[("d", u["k_describe"]), ("t", u["k_read"]), ("v", u["k_voice"]), ("m", u["k_mode"]),
                  ("b", u["k_beeps"]), ("i", u["k_info"]), ("q", u["k_quit"])],
            hint=u["zone_hint"] if self.profile.get("danger_zone") and not self.zone.closed else "",
            titles={"now": u["now"], "radar": u["radar"], "around": u["around"], "recent": u["log"],
                    "nothing": u["nothing"], "sound": u["sound_level"]},
            labels=labels, details=details,
        )
        cv2.imshow(WINDOW_NAME, compose(view, st))

    def report(self, stages: StageTimes, fps: float, dropped: int, now: float) -> None:
        means = stages.pop_window_means()
        counts = {s: sum(t.status is s for t in self.tracks) for s in TrackStatus}
        depth_ms = self.depth.latest.infer_ms if self.depth and self.depth.latest else None
        print(f"[occhio] FPS {fps:5.1f} | " + " | ".join(f"{k} {v:5.1f} ms" for k, v in means.items())
              + (f" | depth {depth_ms:4.0f} ms" if depth_ms is not None else "")
              + f" | rel {self.rel.score:.2f} {self.rel.reason}"
              + f" | tracks C{counts[TrackStatus.CONFIRMED]} U{counts[TrackStatus.UNCERTAIN]}"
              f" H{counts[TrackStatus.HIDDEN]}" + (" | eco" if self.gate.eco else "") + f" | dropped {dropped}",
              flush=True)
        self.metrics.row(now, fps, means, depth_ms, self.rel.score, counts[TrackStatus.CONFIRMED], self.gate.eco)

    def close(self) -> None:
        for part in (self.sound, self.beeper, self.speaker):
            part.stop()
        if self.depth:
            self.depth.stop()
        if self.vibration:
            self.vibration.stop()
        if self.phone is not None:
            self.phone.stop()
        if self.recorder:
            self.recorder.close()
            print(f"[occhio] recorded {self.recorder.written} frames to {self.recorder.path}"
                  + (f" ({self.recorder.dropped} dropped)" if self.recorder.dropped else ""), flush=True)
        summary = self.metrics.summary()
        self.metrics.close()
        if summary:
            print("[occhio] summary: " + ", ".join(f"{k} {v:.1f}" for k, v in summary.items())
                  + (f" -> {self.args.metrics}" if self.args.metrics else ""), flush=True)
        cv2.destroyAllWindows()
