"""Profiles: one engine, two modes.

A profile holds everything that differs between Street and Industrial mode:
the object classes we care about, the warning distances and every user-facing
voice line. Code must take its strings from here, never hard-code them.

Lines are per language ("en" default, "it" with --lang it). Placeholders:
{obj} object name, {dist} distance, {unit} metro / metri (agrees with {dist}),
{side} sinistra / destra / davanti,
{ttc} seconds to collision.

corridor: the walking corridor, a trapezoid in fractions of the frame
(bottom edge = frame bottom). Only objects whose feet (box bottom-centre) are
inside it can be "danger".
"""
from __future__ import annotations

from typing import Any

Profile = dict[str, Any]

LANGUAGES = ("it", "en")
VOICES: dict[str, str] = {"it": "Alice", "en": "Samantha"}  # macOS `say` voices

# Words shared by both modes.
OBJECTS: dict[str, dict[str, str]] = {
    "it": {"person": "persona", "bicycle": "bici", "car": "auto", "motorcycle": "moto",
           "bus": "autobus", "truck": "camion", "dog": "cane", "train": "tram", "cat": "gatto",
           "backpack": "zaino", "handbag": "borsa", "suitcase": "valigia", "umbrella": "ombrello",
           "bottle": "bottiglia", "cell phone": "telefono", "sports ball": "palla", "skateboard": "skateboard",
           "chair": "sedia", "bench": "panchina", "dining table": "tavolo", "potted plant": "vaso",
           "laptop": "portatile", "tv": "schermo", "couch": "divano", "bed": "letto", "refrigerator": "frigorifero",
           "toilet": "WC", "sink": "lavandino", "oven": "forno", "obstacle": "ostacolo",
           "fire hydrant": "idrante", "parking meter": "parchimetro", "stop sign": "segnale di stop",
           "traffic light": "semaforo"},
    "en": {"person": "person", "bicycle": "bike", "car": "car", "motorcycle": "motorbike",
           "bus": "bus", "truck": "truck", "dog": "dog", "train": "tram", "cat": "cat",
           "backpack": "backpack", "handbag": "bag", "suitcase": "suitcase", "umbrella": "umbrella",
           "bottle": "bottle", "cell phone": "phone", "sports ball": "ball", "skateboard": "skateboard",
           "chair": "chair", "bench": "bench", "dining table": "table", "potted plant": "plant pot",
           "laptop": "laptop", "tv": "screen", "couch": "sofa", "bed": "bed", "refrigerator": "fridge",
           "toilet": "toilet", "sink": "sink", "oven": "oven", "obstacle": "obstacle",
           "fire hydrant": "fire hydrant", "parking meter": "parking meter", "stop sign": "stop sign",
           "traffic light": "traffic light"},
}
# What kind of thing it is: decides the beep pitch (vehicles high, people/animals middle, obstacles low).
CATEGORY: dict[str, str] = {
    **{c: "vehicle" for c in ("bicycle", "car", "motorcycle", "bus", "truck", "train")},
    **{c: "living" for c in ("person", "dog", "cat")},
}  # everything else: "obstacle"
SOUND_WHERE: dict[str, dict[str, str]] = {  # appended to sound lines when a stereo mic gives a side
    "it": {"left": " a sinistra", "right": " a destra", "": ""},
    "en": {"left": " on the left", "right": " on the right", "": ""},
}
OBJECTS_PLURAL: dict[str, dict[str, str]] = {
    "it": {"person": "persone", "bicycle": "bici", "car": "auto", "motorcycle": "moto",
           "bus": "autobus", "truck": "camion", "dog": "cani", "train": "tram", "cat": "gatti",
           "backpack": "zaini", "handbag": "borse", "suitcase": "valigie", "umbrella": "ombrelli",
           "bottle": "bottiglie", "cell phone": "telefoni", "sports ball": "palle", "skateboard": "skateboard",
           "chair": "sedie", "bench": "panchine", "dining table": "tavoli", "potted plant": "vasi",
           "fire hydrant": "idranti", "parking meter": "parchimetri", "laptop": "portatili", "tv": "schermi",
           "couch": "divani", "bed": "letti", "refrigerator": "frigoriferi", "toilet": "WC", "sink": "lavandini",
           "oven": "forni", "obstacle": "ostacoli"},
    "en": {"person": "people", "bicycle": "bikes", "car": "cars", "motorcycle": "motorbikes",
           "bus": "buses", "truck": "trucks", "dog": "dogs", "train": "trams", "cat": "cats",
           "backpack": "backpacks", "handbag": "bags", "suitcase": "suitcases", "umbrella": "umbrellas",
           "bottle": "bottles", "cell phone": "phones", "sports ball": "balls", "skateboard": "skateboards",
           "chair": "chairs", "bench": "benches", "dining table": "tables", "potted plant": "plant pots",
           "fire hydrant": "fire hydrants", "parking meter": "parking meters", "laptop": "laptops", "tv": "screens",
           "couch": "sofas", "bed": "beds", "refrigerator": "fridges", "toilet": "toilets", "sink": "sinks",
           "oven": "ovens", "obstacle": "obstacles"},
}
NUMBER_WORDS: dict[str, dict[int, str]] = {  # counts in the instant scene summary (digits above)
    "it": {},
    "en": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"},
}
SIDE_PHRASE: dict[str, dict[str, str]] = {  # where things are, in the instant scene summary
    "it": {"left": "a sinistra", "ahead": "davanti", "right": "a destra"},
    "en": {"left": "on the left", "ahead": "ahead", "right": "on the right"},
}
SIDES: dict[str, dict[str, str]] = {
    "it": {"left": "sinistra", "ahead": "davanti", "right": "destra"},
    "en": {"left": "the left", "ahead": "ahead", "right": "the right"},
}
DECIMAL_SEPARATOR: dict[str, str] = {"it": ",", "en": "."}
UNITS: dict[str, tuple[str, str]] = {"it": ("metro", "metri"), "en": ("metre", "metres")}  # (one, many)

UI_TEXT: dict[str, dict[str, str]] = {  # on-screen labels
    "it": {"street": "STRADA", "industrial": "INDUSTRIALE", "reliable": "AFFIDABILE",
           "unreliable": "NON AFFIDABILE", "eco": "RISPARMIO", "mic": "MIC", "mic_off": "MIC OFF",
           "listening": "TI ASCOLTO...", "thinking": "CAPISCO...", "describing": "DESCRIVO...",
           "radar": "RADAR", "sounds": "SUONI", "log": "RECENTI", "rec": "REC",
           "keys": "d descrivi | t leggi | v voce | m modo | q esci",
           "zone_hint": "clic sinistro: angolo | clic destro: chiudi / cancella",
           "red": "ROSSO", "green": "VERDE", "step": "GRADINO", "phone": "telefono",
           "horn": "clacson", "siren": "sirena", "approach": "rumore in avvicinamento",
           "status_clear": "LIBERO", "status_caution": "ATTENZIONE", "status_danger": "PERICOLO",
           "status_unreliable": "CAMERA NON AFFIDABILE", "now": "ORA", "around": "INTORNO A TE",
           "nothing": "Niente nelle vicinanze", "waiting": "In ascolto della scena...",
           "k_describe": "Descrivi", "k_read": "Leggi", "k_voice": "Voce", "k_mode": "Modo",
           "k_info": "Dettagli", "k_quit": "Esci", "fps": "FPS", "sound_level": "Suono",
           "k_beeps": "Bip", "beeps_off": "BIP OFF",
           "phone_ready": "Pronto", "phone_tap": "Tocca Avvia e punta la fotocamera davanti a te",
           "phone_start": "Avvia", "test_voice": "Prova la voce", "voice_on": "Voce attiva",
           "walking": "CAMMINI"},
    "en": {"street": "STREET", "industrial": "INDUSTRIAL", "reliable": "RELIABLE",
           "unreliable": "UNRELIABLE", "eco": "ECO", "mic": "MIC", "mic_off": "MIC OFF",
           "listening": "LISTENING...", "thinking": "THINKING...", "describing": "DESCRIBING...",
           "radar": "RADAR", "sounds": "SOUNDS", "log": "RECENT", "rec": "REC",
           "keys": "d describe | t read | v voice | m mode | q quit",
           "zone_hint": "left click: corner | right click: close / clear",
           "red": "RED", "green": "GREEN", "step": "STEP", "phone": "phone",
           "horn": "horn", "siren": "siren", "approach": "approaching sound",
           "status_clear": "CLEAR", "status_caution": "CAUTION", "status_danger": "DANGER",
           "status_unreliable": "CAMERA UNRELIABLE", "now": "NOW", "around": "AROUND YOU",
           "nothing": "Nothing nearby", "waiting": "Watching the way ahead...",
           "k_describe": "Describe", "k_read": "Read", "k_voice": "Voice", "k_mode": "Mode",
           "k_info": "Details", "k_quit": "Quit", "fps": "FPS", "sound_level": "Sound",
           "k_beeps": "Beeps", "beeps_off": "BEEPS OFF",
           "phone_ready": "Ready", "phone_tap": "Tap Start and point the camera ahead of you",
           "phone_start": "Start", "test_voice": "Test voice", "voice_on": "Voice on",
           "walking": "WALKING"},
}

PROFILES: dict[str, Profile] = {
    "street": {
        "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "dog", "cat",
                    # things on the pavement
                    "backpack", "handbag", "suitcase", "umbrella", "bottle", "cell phone", "sports ball",
                    "skateboard", "chair", "bench", "dining table", "potted plant", "fire hydrant",
                    "parking meter",
                    # indoors (hallways, offices, the venue)
                    "laptop", "tv", "couch", "bed", "refrigerator", "toilet", "sink", "oven"],
        "signal_classes": ["traffic light", "stop sign"],  # detected and read, never treated as obstacles
        "step_warning": True,
        "danger_zone": False,
        "warn_distance_m": 3.0,
        "danger_distance_m": 1.5,
        "approach_min_mps": 0.3,  # slower than this is "not approaching"
        "fast_mps": 2.5,  # faster than walking: bikes, cars
        "corridor": {"top_y": 0.45, "top_w": 0.30, "bottom_w": 0.90},
        "lang": "en",
        "lines": {
            "ahead": "{obj} davanti, {dist} {unit}",
            "side": "{obj} a {side}, {dist} {unit}",
            "approach": "{obj} in arrivo da {side}",
            "ttc": "Collisione tra {ttc} secondi",
            "ttc_one": "Collisione tra un secondo",
            "clear": "Strada libera",
            "unreliable": "Visibilità scarsa, guida non affidabile",
            "mode": "Modalità strada",
            "describe_failed": "Descrizione non disponibile",
            "light_red": "Semaforo rosso",
            "light_green": "Semaforo verde",
            "step": "Gradino in basso",
            "zone": "Persona in zona pericolosa",
            "read_failed": "Lettura non disponibile",
            "listening": "Ti ascolto",
            "scene_none": "Nessun ostacolo rilevato",
            "stop_sign": "Segnale di stop davanti",
            "read_none": "Nessun testo trovato",
            "read_result": "Scritta: {text}",
            "sound_horn": "Clacson{where}",
            "sound_siren": "Sirena{where}",
            "sound_approach": "Rumore in avvicinamento{where}",
            "command_unknown": "Comando non capito",
            "voice_unavailable": "Comandi vocali non disponibili",
        },
        "lines_en": {
            "ahead": "{obj} ahead, {dist} {unit}",
            "side": "{obj} on {side}, {dist} {unit}",
            "approach": "{obj} coming from {side}",
            "ttc": "Collision in {ttc} seconds",
            "ttc_one": "Collision in one second",
            "clear": "Path clear",
            "unreliable": "Poor visibility, guidance unreliable",
            "mode": "Street mode",
            "describe_failed": "Description not available",
            "light_red": "Red light",
            "light_green": "Green light",
            "step": "Step down ahead",
            "zone": "Person in the danger zone",
            "read_failed": "Reading not available",
            "listening": "Listening",
            "scene_none": "Nothing detected around you",
            "stop_sign": "Stop sign ahead",
            "read_none": "No text found",
            "read_result": "Text: {text}",
            "sound_horn": "Horn{where}",
            "sound_siren": "Siren{where}",
            "sound_approach": "Loud sound approaching{where}",
            "command_unknown": "Command not understood",
            "voice_unavailable": "Voice commands not available",
        },
        "read_prompt": {
            "it": "Read the text a blind person needs from this image: signs, bus numbers and destinations, "
                  "shop names, labels. Answer in one short Italian sentence (max 20 words), most important first, "
                  "e.g. 'Autobus 140 per Piazza Garibaldi'. If there is no readable text, answer 'Nessun testo'.",
            "en": "Read the text a blind person needs from this image: signs, bus numbers and destinations, "
                  "shop names, labels. Answer in one short English sentence (max 20 words), most important first, "
                  "e.g. 'Bus 140 to Piazza Garibaldi'. If there is no readable text, answer 'No text'.",
        },
        "describe_prompt": {
            "it": "You help a blind person walking. In one short Italian sentence (max 20 words) describe "
                  "obstacles, people, vehicles, crossings, traffic lights and signs in front of them. "
                  "Most important first.",
            "en": "You help a blind person walking. In one short English sentence (max 20 words) describe "
                  "obstacles, people, vehicles, crossings, traffic lights and signs in front of them. "
                  "Most important first.",
        },
    },
    "industrial": {
        "classes": ["person", "truck", "car", "bicycle", "motorcycle", "suitcase", "backpack", "chair"],
        "signal_classes": [],
        "step_warning": False,
        "danger_zone": True,  # draw it with the mouse: left click corners, right click close/clear
        "warn_distance_m": 5.0,
        "danger_distance_m": 2.5,
        "approach_min_mps": 0.3,
        "fast_mps": 2.0,
        "corridor": {"top_y": 0.40, "top_w": 0.45, "bottom_w": 1.00},
        "lang": "en",
        "lines": {
            "ahead": "Attenzione: {obj} a {dist} {unit}",
            "side": "Attenzione: {obj} a {side}, {dist} {unit}",
            "approach": "{obj} in arrivo da {side}. Stop.",
            "ttc": "Stop. Collisione tra {ttc} secondi",
            "ttc_one": "Stop. Collisione tra un secondo",
            "clear": "Area libera",
            "unreliable": "Telecamera non affidabile, fermati e verifica",
            "mode": "Modalità industriale",
            "describe_failed": "Descrizione non disponibile",
            "light_red": "Semaforo rosso",
            "light_green": "Semaforo verde",
            "step": "Attenzione, dislivello",
            "zone": "Operatore in zona pericolosa",
            "read_failed": "Lettura non disponibile",
            "listening": "Ti ascolto",
            "scene_none": "Nessun ostacolo rilevato",
            "stop_sign": "Segnale di stop davanti",
            "read_none": "Nessun testo trovato",
            "read_result": "Scritta: {text}",
            "sound_horn": "Clacson{where}",
            "sound_siren": "Sirena{where}",
            "sound_approach": "Rumore in avvicinamento{where}",
            "command_unknown": "Comando non capito",
            "voice_unavailable": "Comandi vocali non disponibili",
        },
        "lines_en": {
            "ahead": "Warning: {obj} at {dist} {unit}",
            "side": "Warning: {obj} on {side}, {dist} {unit}",
            "approach": "{obj} coming from {side}. Stop.",
            "ttc": "Stop. Collision in {ttc} seconds",
            "ttc_one": "Stop. Collision in one second",
            "clear": "Area clear",
            "unreliable": "Camera unreliable, stop and check",
            "mode": "Industrial mode",
            "describe_failed": "Description not available",
            "light_red": "Red light",
            "light_green": "Green light",
            "step": "Warning, level change",
            "zone": "Person in the danger zone",
            "read_failed": "Reading not available",
            "listening": "Listening",
            "scene_none": "Nothing detected around you",
            "stop_sign": "Stop sign ahead",
            "read_none": "No text found",
            "read_result": "Text: {text}",
            "sound_horn": "Horn{where}",
            "sound_siren": "Siren{where}",
            "sound_approach": "Loud sound approaching{where}",
            "command_unknown": "Command not understood",
            "voice_unavailable": "Voice commands not available",
        },
        "read_prompt": {
            "it": "Read the text a worker needs from this image: signs, bus numbers and destinations, "
                  "shop names, labels. Answer in one short Italian sentence (max 20 words), most important first, "
                  "e.g. 'Autobus 140 per Piazza Garibaldi'. If there is no readable text, answer 'Nessun testo'.",
            "en": "Read the text a worker needs from this image: signs, bus numbers and destinations, "
                  "shop names, labels. Answer in one short English sentence (max 20 words), most important first, "
                  "e.g. 'Bus 140 to Piazza Garibaldi'. If there is no readable text, answer 'No text'.",
        },
        "describe_prompt": {
            "it": "You help a worker in a hangar or factory. In one short Italian sentence (max 20 words) "
                  "describe people, vehicles, forklifts, obstacles and hazards in front of them. "
                  "Most important first.",
            "en": "You help a worker in a hangar or factory. In one short English sentence (max 20 words) "
                  "describe people, vehicles, forklifts, obstacles and hazards in front of them. "
                  "Most important first.",
        },
    },
}

DEFAULT_MODE = "street"


def detect_classes(profile: Profile) -> list[str]:
    """Everything the detector should find: obstacles plus signals (traffic lights)."""
    return list(profile["classes"]) + list(profile.get("signal_classes", []))


def get_profile(mode: str) -> Profile:
    """Return the profile for a mode name, or raise ValueError if unknown."""
    try:
        return PROFILES[mode]
    except KeyError:
        raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(PROFILES)}") from None


def lines_for(profile: Profile, lang: str) -> dict[str, str]:
    """The profile's voice lines in `lang` ('it' or 'en')."""
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown language {lang!r}; choose from {LANGUAGES}")
    return profile["lines"] if lang == "it" else profile["lines_en"]


def format_distance(metres: float, lang: str) -> str:
    """Spoken distance: nearest 0.5 m below 3 m, whole metres above ('1,5' / '2' / '4')."""
    value = round(metres * 2) / 2 if metres < 3.0 else float(round(metres))
    value = max(value, 0.5)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}".replace(".", DECIMAL_SEPARATOR[lang])


def render(profile: Profile, lang: str, key: str, cls_name: str = "", dist_m: float | None = None,
           side: str = "ahead", ttc_s: float | None = None) -> str:
    """Build one spoken sentence from a profile line, capitalised."""
    if key == "ttc" and ttc_s is not None and round(ttc_s) <= 1:
        key = "ttc_one"
    dist = format_distance(dist_m, lang) if dist_m is not None else ""
    text = lines_for(profile, lang)[key].format(
        obj=OBJECTS[lang].get(cls_name, cls_name),
        dist=dist,
        unit=UNITS[lang][0 if dist == "1" else 1],
        side=SIDES[lang][side],
        ttc=max(1, round(ttc_s)) if ttc_s is not None else "",
    )
    return text[:1].upper() + text[1:]


def next_mode(mode: str) -> str:
    """The mode after `mode` in PROFILES order (for the 'm' key)."""
    names = list(PROFILES)
    return names[(names.index(mode) + 1) % len(names)]
