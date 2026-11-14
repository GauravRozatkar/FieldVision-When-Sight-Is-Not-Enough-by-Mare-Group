"""Engine settings shared by both modes (thresholds, model choice).

Per-mode values (classes, distances, voice lines) live in profiles.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CONFIG_DIR.parent

DETECTOR: dict[str, Any] = {
    "model": str(PROJECT_DIR / "yolo11n.pt"),
    # yolo11n: 16 ms/frame on the M5 Air, COCO mAP 39.5 -> webcam / video (keeps 20+ FPS).
    # yolo11s: 34 ms/frame, COCO mAP 47.0 (fewer wrong labels) -> phone mode, where Wi-Fi
    # delivers a frame only every 65-100 ms anyway. Choose with --model n|s.
    "models": {"n": str(PROJECT_DIR / "yolo11n.pt"), "s": str(PROJECT_DIR / "yolo11s.pt")},
    "device": "mps",  # falls back to cpu when MPS is unavailable
    "imgsz": 640,
    "tracker": str(CONFIG_DIR / "bytetrack_occhio.yaml"),
}

TRACKING: dict[str, Any] = {
    "confirm_frames": 5,  # frames seen before a track can be confirmed
    "confirm_mean_conf": 0.6,  # mean confidence needed to confirm
    # YOLO is less sure about small things; with 0.6 a phone on the floor would never be confirmed.
    # They need more frames instead.
    "class_confirm": {c: (0.4, 8) for c in ("backpack", "handbag", "suitcase", "umbrella", "bottle",
                                             "cell phone", "sports ball", "skateboard", "chair", "bench",
                                             "dining table", "potted plant", "fire hydrant", "parking meter",
                                             "cat", "stop sign", "laptop", "tv", "couch", "bed",
                                             "refrigerator", "toilet", "sink", "oven")} | {"chair": (0.3, 8)},  # class: (mean confidence, frames)
    "hidden_ttl_s": 1.5,  # how long a lost confirmed track is kept as "hidden"
    "uncertain_ttl_s": 0.5,  # how long a lost uncertain track is kept before deletion
    "history_len": 30,  # boxes / confidences kept per track
    "velocity_window": 5,  # last N boxes used to estimate velocity for prediction
    "full_box_ratio": 0.9,  # boxes shorter than this x median height are partly occluded:
    #                         ignored when predicting a hidden track's position
    # Re-identification: when ByteTrack gives a new id to someone who just came out
    # from behind an obstacle, hand the new id to the matching hidden track.
    "reid_gate": 1.0,  # max distance from the hidden track's path, in multiples of its median height
    "reid_path_overshoot": 1.5,  # the path runs from the last seen box to 1.5x the predicted move
    "reid_max_height_ratio": 1.6,  # new box height within [1/r, r] x hidden median height
    "reid_min_conf": 0.4,  # weaker new detections are not re-identified
    # Lost memory: after hidden_ttl_s a confirmed track is no longer drawn, but its
    # appearance (torso colour histogram) is remembered so someone who walks away
    # and comes back gets their old id.
    "lost_memory_s": 10.0,  # how long a lost track can still be re-identified
    "appearance_max_dist": 0.4,  # Bhattacharyya distance; same person ~0.1-0.35, others >= 0.5
    "appearance_min_conf": 0.5,  # only confident detections update a track's appearance
    "appearance_ema": 0.2,  # weight of the newest frame in the appearance average
    "reid_window_s": 1.0,  # a new track can match a lost one during its first second
}

CAMERA: dict[str, Any] = {
    "hfov_deg": 70.0,  # horizontal field of view; used for focal length until --calibrate
    "height_m": 1.2,  # camera height above the floor (chest-worn phone ~1.2 m; laptop on a table ~0.75 m)
}

CALIBRATION_FILE = CONFIG_DIR / "calibration.json"  # written by `main.py --calibrate`

DEPTH: dict[str, Any] = {
    "enabled": True,
    "model": "depth-anything/Depth-Anything-V2-Small-hf",
    "device": "mps",
    "every_n": 3,  # run on every 3rd frame; skipped while the previous run is still busy
    "input_size": (448, 252),  # (w, h) multiple of 14; ~30 ms on M5 (518x924 is ~125 ms)
}

SPATIAL: dict[str, Any] = {
    "history": 10,  # distance samples used for the approach-speed fit
    "min_samples": 5,  # fewer samples => approach speed 0
    "max_sample_age_s": 1.5,  # older samples are dropped (track was gone)
    "side_margin": 0.05,  # side changes only 5% of the width past the third boundary
    "approach_ema": 0.3,  # smoothing of the approach speed: one or two odd frames cannot fake a danger
    # Typical real size in metres, for the box-size distance fallback.
    "heights_m": {"person": 1.70, "bicycle": 1.05, "car": 1.50, "motorcycle": 1.20,
                  "bus": 3.00, "truck": 3.00, "train": 3.50, "dog": 0.50, "cat": 0.30,
                  "backpack": 0.50, "handbag": 0.30, "suitcase": 0.65, "umbrella": 0.90, "bottle": 0.25,
                  "cell phone": 0.15, "sports ball": 0.22, "skateboard": 0.12, "chair": 0.90,
                  "bench": 0.85, "dining table": 0.75, "potted plant": 0.70, "fire hydrant": 0.75,
                  "parking meter": 1.40, "traffic light": 1.00, "stop sign": 0.75,
                  "laptop": 0.25, "tv": 0.60, "couch": 0.85, "bed": 0.60, "refrigerator": 1.80,
                  "toilet": 0.75, "sink": 0.90, "oven": 0.90},
    # Small things lying on the floor: distance from where they touch the floor (needs the
    # camera height; the camera is assumed roughly level), because their box height says little.
    "ground_classes": ["cell phone", "bottle", "sports ball", "skateboard", "handbag", "backpack", "suitcase"],
    "person_width_m": 0.45,  # shoulders; used when the person's box is cut by the frame edge
    "edge_margin": 0.02,  # a box within 2% of the frame height from the top/bottom is "cut"
    #                       (YOLO stops a few px short of the edge when the head is out of frame)
    "auto_calibrate_rate": 0.05,  # how fast depth scale follows full-body people (no calibration file)
}

RELIABILITY: dict[str, Any] = {
    "analysis_width": 320,  # frames are downscaled to this width before measuring
    "dark": (20.0, 50.0),  # mean grey: score 0 below the first value, 1 above the second
    "bright": (225.0, 250.0),  # over-exposed: score 1 below the first value, 0 above the second
    "blur": (40.0, 150.0),  # variance of Laplacian at 320 px: 0 below, 1 above (webcam ~600-1900)
    "blocked_std": 10.0,  # grey std below this = lens covered / blank (webcam ~60, hand ~6)
    "ema_alpha": 0.25,  # smoothing; 0.25 at 25+ FPS drops to the threshold in ~0.2 s
    "unreliable_below": 0.4,
    "reliable_above": 0.5,  # hysteresis: back to reliable only above this
    # A well-lit view with almost no edges is not a broken camera: the camera is right up
    # against a plain surface (a wall, a door). Measured on a real iPhone: wall = brightness 133,
    # sharpness 24, contrast 13; a shaky but normal room keeps sharpness ~300+.
    "surface_max_sharpness": 60.0,
    "surface_max_contrast": 25.0,
    "surface_min_light": 0.6,  # light score (0-1): clearly not dark, not over-exposed
}

# A phone in the hand shakes: tolerate more motion blur and react only to blur that lasts
# (a covered lens is still caught within about half a second).
RELIABILITY_PHONE: dict[str, Any] = {"blur": (20.0, 80.0), "ema_alpha": 0.12}

PRIORITY: dict[str, Any] = {
    "cooldown_s": 3.0,  # same track + message type at most once every 3 s
    "repeat_same_text_s": 10.0,  # identical sentence for the same track not repeated sooner
    "danger_repeat_s": 1.5,  # danger skips the cooldown, but is not repeated more often than this
    "danger_still_repeat_s": 6.0,  # unchanged danger (same sentence) is repeated only this often;
    #                                the beeps already say "still close"
    "danger_ttc_s": 2.0,  # time to collision below this => danger
    "unreliable_repeat_s": 5.0,
    "min_closeness_m": 0.3,
    # Calm speech in busy scenes:
    "min_gap_s": 2.5,  # at most one sentence every 2.5 s ...
    "urgent_gap_s": 1.2,  # ... except a new danger or one that came closer, after 1.2 s
    "closer_m": 0.5,  # a danger is re-announced early only when it came at least this much closer
    "away_mps": -0.3,  # moving away faster than this: no new (non-danger) warning
    "crowd_count": 3,  # this many objects within warn distance => one summary sentence
    "crowd_repeat_s": 8.0,
}

SPEECH: dict[str, Any] = {
    "rate_wpm": 200,  # `say -r`
    "max_queue": 2,  # older normal messages are dropped: only the latest matter
}

BEEPS: dict[str, Any] = {
    "enabled": True,
    "samplerate": 44100,
    "freq_hz": 660.0,
    "category_freq_hz": {"vehicle": 990.0, "living": 660.0, "obstacle": 440.0},  # pitch says what it is
    "beep_ms": 45,
    "fade_ms": 12,  # soft fade in/out: a gentle "tip" rather than a hard click
    "volume": 0.12,  # quiet: the user also needs to hear the street
    "min_hz": 0.5,  # beeps per second at warn distance
    "max_hz": 3.0,  # beeps per second at danger distance and closer
    # Beep only while something is coming closer, or for a few seconds after it becomes near;
    # something that just stays there (a person sitting in front of you) goes quiet.
    "quiet_after_s": 3.0,
    "rearm_closer_m": 0.5,  # ... and starts beeping again once it is this much closer
}

TALK: dict[str, Any] = {
    "ttc_warn_s": 3.0,  # "Collisione tra N secondi" below this time to collision
    "clear_after_s": 5.0,  # "Strada libera" after this long without anything dangerous
    "event_log_size": 8,
}

SCENE_LLM: dict[str, Any] = {
    "model_env": "OCCHIO_LLM_MODEL",  # model id read from this environment variable
    "default_model": "claude-opus-5-5",
    "effort": "low",  # one short sentence: low effort keeps latency down
    "max_tokens": 2048,
    "max_image_px": 768,  # longest side of the JPEG sent
    "jpeg_quality": 80,
    "timeout_s": 20.0,
}

TRAFFIC_LIGHT: dict[str, Any] = {
    # HSV ranges (OpenCV: H 0-180). Red wraps around 0.
    "red_hue": ((0, 10), (160, 180)),
    "green_hue": (40, 95),  # pedestrian greens are often blue-ish
    "min_sat": 90,
    "min_val": 120,
    "min_fraction": 0.04,  # lit pixels needed, as a fraction of the box area
    "stable_frames": 6,  # a colour must hold this many frames before it is announced
}

STEP: dict[str, Any] = {
    "strip": 0.15,  # bottom 15% of the depth map vs the 15% strip above it
    "ratio_min": 0.9,  # bottom disparity < 0.9 x above = the floor drops away
    "consecutive": 3,  # depth frames in a row before warning
    "cooldown_s": 5.0,
}

VOICE: dict[str, Any] = {
    "model": "small",  # faster-whisper model; download once: see README
    "compute_type": "int8",
    "samplerate": 16000,
    "record_s": 3.0,  # push-to-talk: 'v' records this long
    "lead_in_s": 0.9,  # wait for "Ti ascolto" to finish before recording
}

ZONE_FILE = CONFIG_DIR / "zone.json"  # restricted zone drawn with the mouse (industrial mode)
ZONE: dict[str, Any] = {
    "cooldown_s": 4.0,  # same person in the zone announced at most this often
}

ECO: dict[str, Any] = {
    "still_after_s": 3.0,  # no motion and nothing tracked for this long => eco mode
    "detect_every": 5,  # in eco mode run the detector on every 5th frame only
    "motion_threshold": 2.5,  # mean abs grey difference (0-255) on a 160x90 copy
}

METRICS: dict[str, Any] = {
    "file": "metrics.csv",
}

SOUND: dict[str, Any] = {
    "enabled": True,
    "samplerate": 16000,
    "block_s": 0.1,  # analyse the microphone in 100 ms blocks
    "loud_db": -30.0,  # dBFS: quieter blocks are ignored (room ~ -55, street car ~ -35..-20)
    "tonal_ratio": 0.25,  # energy share of the strongest 50 Hz band in 250-3500 Hz => "tonal"
    "horn_band": (250.0, 1200.0),  # Hz: car horns and bike bells (fundamental)
    "horn_jump_db": 12.0,  # louder than the background by this much, suddenly
    "horn_db": -25.0,  # and at least this loud (a voice next to the laptop is ~ -30)
    "horn_tonal_ratio": 0.35,  # a horn is a much purer tone than a voice
    "horn_steady_blocks": 3,  # same pitch (within 6%) for 3 blocks = 0.3 s; speech pitch wobbles
    "warmup_s": 2.0,  # learn the room's background before any warning
    "siren_band": (500.0, 1800.0),
    "siren_sweep_hz": 150.0,  # the tone must move at least this much over the window
    "siren_window_s": 1.5,
    "approach_window_s": 2.0,  # level rising this long ...
    "approach_rise_db": 8.0,  # ... by at least this much => something approaching
    "cooldown_s": 6.0,  # per sound type
    "side_db": 4.0,  # stereo mic only: louder on one side by this much => that side
}

VIBRATION: dict[str, Any] = {
    "port": 8765,  # phone opens http://<mac-ip>:8765 on the same Wi-Fi / hotspot
}

LOCAL_VLM: dict[str, Any] = {
    "model": "HuggingFaceTB/SmolVLM-500M-Instruct",  # small vision-language model, runs offline
    "device": "mps",
    "max_new_tokens": 40,
    "max_image_px": 768,
    "load_timeout_s": 90.0,  # a request waits this long for the model to finish loading
}

FREE_SPACE: dict[str, Any] = {
    "flat_rel": 0.04,  # depth changing less than 4% over the window = an upright surface
    "flat_rows": 0.08,  # window: 8% of the image height
    "floor_rel": 0.08,  # the floor recedes by at least 8% over the window
    "stable_frames": 3,  # depth maps in a row before a wall is reported
    "max_m": 4.0,  # farther walls are not interesting
}
