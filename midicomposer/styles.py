"""Genre and emotion style profiles used by the planner and rule-based generator.

Labels are open-ended: XMIDI labels are discovered from filenames, so any unknown genre or
emotion is mapped to the closest profile by keyword, falling back to neutral defaults
(with a warning) rather than failing.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenreProfile:
    name: str
    tempo_range: tuple[int, int]
    forms: list[list[tuple[str, int]]]
    progressions: dict[str, list[list[str]]]     # mode -> roman-numeral progressions (1 chord / bar)
    drum_style: str
    bass_style: str
    comp_style: str
    swing: bool = False
    instrumentation: dict[str, int] = field(default_factory=dict)   # generated role -> GM program
    meter: str = "4/4"


_MAJOR_POP = [["I", "V", "vi", "IV"], ["I", "vi", "IV", "V"], ["vi", "IV", "I", "V"], ["I", "IV", "vi", "V"],
              ["IV", "I", "V", "vi"], ["I", "iii", "IV", "V"]]
_MINOR_POP = [["i", "VI", "III", "VII"], ["i", "iv", "VII", "III"], ["i", "VII", "VI", "VII"],
              ["VI", "VII", "i", "i"], ["i", "iv", "VI", "V"]]
_MODAL = {
    "dorian": [["i", "IV", "i", "IV"], ["i", "VII", "IV", "i"], ["i", "ii", "IV", "i"]],
    "mixolydian": [["I", "bVII", "IV", "I"], ["I", "bVII", "I", "IV"], ["I", "v", "bVII", "IV"]],
    "lydian": [["I", "II", "I", "II"], ["I", "II", "vii", "I"]],
    "phrygian": [["i", "bII", "i", "bII"], ["i", "bII", "VII", "i"]],
    "locrian": [["i°", "bII", "i°", "bII"]],
    "harmonic_minor": [["i", "iv", "V", "i"], ["i", "VI", "iv", "V"]],
}

GENRES: dict[str, GenreProfile] = {
    "pop": GenreProfile(
        "pop", (90, 128),
        [[("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("bridge", 8), ("chorus", 8), ("outro", 4)]],
        {"major": _MAJOR_POP, "minor": _MINOR_POP}, "pop", "pop", "pulse8",
        instrumentation={"lead_melody": 73, "chord_bed": 0, "bass_line": 32, "pad": 48, "arpeggio": 25,
                         "counterline": 66, "harmony": 48, "octave_doubles": 48, "high_texture": 11, "low_texture": 42}),
    "rock": GenreProfile(
        "rock", (100, 150),
        [[("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("solo", 8), ("chorus", 8), ("outro", 4)]],
        {"major": [["I", "bVII", "IV", "I"], ["I", "IV", "V", "IV"], ["I", "V", "IV", "IV"], ["I", "bIII", "IV", "I"]],
         "minor": [["i", "VII", "VI", "VII"], ["i", "VI", "VII", "i"], ["i", "iv", "VI", "V"], ["i", "III", "VII", "IV"]]},
        "rock", "pedal8", "pulse8",
        instrumentation={"lead_melody": 56, "chord_bed": 25, "bass_line": 32, "pad": 48, "arpeggio": 0,
                         "counterline": 66, "harmony": 25, "octave_doubles": 48, "high_texture": 11, "low_texture": 42}),
    "jazz": GenreProfile(
        "jazz", (90, 180),
        [[("A", 8), ("A", 8), ("B", 8), ("A", 8)], [("head", 12), ("solo", 12), ("head", 12)]],
        {"major": [["Imaj7", "vi7", "ii7", "V7"], ["ii7", "V7", "Imaj7", "Imaj7"], ["Imaj7", "IV7", "iii7", "VI7"],
                   ["iii7", "VI7", "ii7", "V7"]],
         "minor": [["i7", "iv7", "bVII7", "IIImaj7"], ["iiø7", "V7", "i7", "i7"], ["i7", "VImaj7", "iiø7", "V7"]]},
        "jazz", "walking", "charleston", swing=True,
        instrumentation={"lead_melody": 66, "chord_bed": 0, "bass_line": 32, "walking_bass": 32, "pad": 48,
                         "arpeggio": 26, "counterline": 56, "harmony": 65, "octave_doubles": 56, "high_texture": 11,
                         "low_texture": 42}),
    "country": GenreProfile(
        "country", (80, 130),
        [[("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("outro", 4)]],
        {"major": [["I", "I", "IV", "I"], ["I", "IV", "V", "I"], ["I", "V", "I", "I"], ["IV", "I", "V", "I"]],
         "minor": [["i", "VI", "VII", "i"], ["i", "iv", "VII", "III"]]},
        "country", "boom_chick", "boom_chick",
        instrumentation={"lead_melody": 73, "chord_bed": 25, "bass_line": 32, "pad": 48, "arpeggio": 0,
                         "counterline": 56, "harmony": 25, "octave_doubles": 48, "high_texture": 105, "low_texture": 42}),
    "classical": GenreProfile(
        "classical", (60, 120),
        [[("A", 8), ("B", 8), ("A", 8)], [("A", 8), ("A", 8), ("B", 8), ("B", 8)]],
        {"major": [["I", "IV", "V", "I"], ["I", "ii", "V", "I"], ["I", "vi", "ii", "V"], ["I", "V", "vi", "iii"]],
         "minor": [["i", "iv", "V", "i"], ["i", "ii°", "V", "i"], ["i", "VI", "iv", "V"]]},
        "orchestral", "sustained", "arpeggio",
        instrumentation={"lead_melody": 40, "chord_bed": 0, "bass_line": 42, "pad": 48, "arpeggio": 46,
                         "counterline": 71, "harmony": 41, "octave_doubles": 73, "high_texture": 73, "low_texture": 43}),
    "traditional": GenreProfile(
        "traditional", (70, 120),
        [[("A", 8), ("A", 8), ("B", 8), ("B", 8)]],
        {"major": [["I", "IV", "I", "V"], ["I", "I", "V", "I"], ["I", "vi", "IV", "V"]],
         "minor": [["i", "VII", "i", "V"], ["i", "iv", "i", "V"], ["i", "III", "VII", "i"]]},
        "hand_percussion", "root_fifth", "arpeggio",
        instrumentation={"lead_melody": 73, "chord_bed": 107, "bass_line": 32, "pad": 48, "arpeggio": 104,
                         "counterline": 104, "harmony": 73, "octave_doubles": 73, "high_texture": 9, "low_texture": 42}),
    # Profiles for labels outside XMIDI's set, reached through keyword mapping.
    "electronic": GenreProfile(
        "electronic", (115, 135), [[("intro", 8), ("build", 8), ("drop", 8), ("break", 8), ("drop", 8), ("outro", 8)]],
        {"major": _MAJOR_POP, "minor": _MINOR_POP}, "four_on_floor", "pedal8", "pulse8",
        instrumentation={"lead_melody": 81, "chord_bed": 90, "bass_line": 38, "pad": 89, "arpeggio": 81,
                         "counterline": 80, "harmony": 90, "octave_doubles": 81, "high_texture": 98, "low_texture": 89}),
    "hip_hop": GenreProfile(
        "hip_hop", (80, 100), [[("intro", 4), ("verse", 16), ("hook", 8), ("verse", 16), ("hook", 8)]],
        {"major": _MAJOR_POP, "minor": [["i", "VI", "i", "VII"], ["i", "iv", "i", "v"]]}, "hiphop", "pop", "sustain",
        instrumentation={"lead_melody": 4, "chord_bed": 4, "bass_line": 38, "pad": 89, "arpeggio": 4,
                         "counterline": 11, "harmony": 4, "octave_doubles": 4, "high_texture": 11, "low_texture": 89}),
    "blues": GenreProfile(
        "blues", (70, 130), [[("chorus", 12), ("chorus", 12), ("chorus", 12)]],
        {"major": [["I7", "I7", "I7", "I7", "IV7", "IV7", "I7", "I7", "V7", "IV7", "I7", "V7"]],
         "minor": [["i7", "i7", "i7", "i7", "iv7", "iv7", "i7", "i7", "VI7", "V7", "i7", "V7"]]},
        "jazz", "walking", "charleston", swing=True,
        instrumentation={"lead_melody": 26, "chord_bed": 4, "bass_line": 33, "walking_bass": 32, "pad": 16,
                         "arpeggio": 26, "counterline": 22, "harmony": 26, "octave_doubles": 26, "high_texture": 11,
                         "low_texture": 42}),
    "ambient": GenreProfile(
        "ambient", (60, 90), [[("A", 8), ("B", 8), ("A", 8), ("C", 8)]],
        {"major": [["Imaj7", "IVmaj7", "Imaj7", "IVmaj7"], ["I", "iii", "IV", "I"]],
         "minor": [["i7", "VImaj7", "i7", "VImaj7"]]}, "none", "sustained", "sustain",
        instrumentation={"lead_melody": 79, "chord_bed": 89, "bass_line": 38, "pad": 91, "arpeggio": 46,
                         "counterline": 79, "harmony": 89, "octave_doubles": 79, "high_texture": 98, "low_texture": 89}),
}

GENRE_KEYWORDS = {
    "edm": "electronic", "dance": "electronic", "house": "electronic", "techno": "electronic", "trance": "electronic",
    "electro": "electronic", "synth": "electronic", "hip": "hip_hop", "rap": "hip_hop", "trap": "hip_hop",
    "rnb": "hip_hop", "blues": "blues", "metal": "rock", "punk": "rock", "grunge": "rock", "indie": "rock",
    "folk": "traditional", "celtic": "traditional", "world": "traditional", "ethnic": "traditional",
    "chinese": "traditional", "orchestral": "classical", "baroque": "classical", "romantic_era": "classical",
    "cinematic": "classical", "soundtrack": "classical", "film": "classical", "ambient": "ambient",
    "new_age": "ambient", "chill": "ambient", "lofi": "hip_hop", "swing": "jazz", "bebop": "jazz",
    "bossa": "jazz", "funk": "pop", "soul": "pop", "disco": "electronic", "bluegrass": "country", "western": "country",
}


@dataclass
class EmotionProfile:
    name: str
    valence: float      # -1 (negative) .. 1 (positive)
    arousal: float      # -1 (calm) .. 1 (energetic)
    preferred_mode: str | None = None


EMOTIONS: dict[str, EmotionProfile] = {
    "happy": EmotionProfile("happy", 0.8, 0.5, "major"),
    "exciting": EmotionProfile("exciting", 0.6, 0.9, None),
    "warm": EmotionProfile("warm", 0.6, -0.2, "major"),
    "romantic": EmotionProfile("romantic", 0.5, -0.3, None),
    "funny": EmotionProfile("funny", 0.7, 0.6, "major"),
    "sad": EmotionProfile("sad", -0.7, -0.5, "minor"),
    "angry": EmotionProfile("angry", -0.6, 0.9, "minor"),
    "lazy": EmotionProfile("lazy", 0.2, -0.7, None),
    "quiet": EmotionProfile("quiet", 0.1, -0.8, None),
    "fear": EmotionProfile("fear", -0.8, 0.4, "minor"),
    "magnificent": EmotionProfile("magnificent", 0.5, 0.7, "major"),
}
EMOTION_KEYWORDS = {
    "joy": "happy", "cheer": "happy", "bright": "happy", "upbeat": "exciting", "energetic": "exciting",
    "epic": "magnificent", "grand": "magnificent", "heroic": "magnificent", "calm": "quiet", "peace": "quiet",
    "relax": "lazy", "chill": "lazy", "love": "romantic", "tender": "romantic", "melanch": "sad", "gloom": "sad",
    "dark": "fear", "scary": "fear", "tense": "fear", "anxious": "fear", "aggress": "angry", "rage": "angry",
    "playful": "funny", "humor": "funny", "cozy": "warm", "gentle": "warm",
}


def resolve_genre(label: str | None) -> tuple[GenreProfile, str | None]:
    """Return (profile, warning or None)."""
    if not label:
        return GENRES["pop"], "No genre given; using the pop profile."
    key = label.strip().lower().replace("-", "_").replace(" ", "_")
    if key in GENRES:
        return GENRES[key], None
    for kw, target in GENRE_KEYWORDS.items():
        if kw in key:
            return GENRES[target], f"Genre '{label}' mapped to the '{target}' style profile."
    return GENRES["pop"], f"Genre '{label}' has no style profile; using pop defaults."


def resolve_emotion(label: str | None) -> tuple[EmotionProfile, str | None]:
    if not label:
        return EmotionProfile("neutral", 0.0, 0.0), None
    key = label.strip().lower().replace("-", "_").replace(" ", "_")
    if key in EMOTIONS:
        return EMOTIONS[key], None
    for kw, target in EMOTION_KEYWORDS.items():
        if kw in key:
            return EMOTIONS[target], f"Emotion '{label}' mapped to '{target}'."
    return EmotionProfile(key, 0.0, 0.0), f"Emotion '{label}' unknown; using neutral valence/arousal."


INSTRUMENT_WORDS = {
    "piano": 0, "grand": 0, "epiano": 4, "rhodes": 4, "electric piano": 4, "harpsichord": 6, "organ": 16,
    "accordion": 21, "harmonica": 22, "nylon": 24, "acoustic guitar": 25, "guitar": 25, "jazz guitar": 26,
    "clean guitar": 27, "overdrive": 29, "distortion": 30, "bass": 33, "upright": 32, "acoustic bass": 32,
    "synth bass": 38, "violin": 40, "viola": 41, "cello": 42, "contrabass": 43, "harp": 46, "timpani": 47,
    "strings": 48, "string": 48, "choir": 52, "trumpet": 56, "trombone": 57, "tuba": 58, "horn": 60,
    "brass": 61, "sax": 66, "saxophone": 66, "alto sax": 65, "tenor sax": 66, "oboe": 68, "bassoon": 70,
    "clarinet": 71, "piccolo": 72, "flute": 73, "recorder": 74, "pan flute": 75, "synth lead": 81, "lead": 81,
    "pad": 89, "warm pad": 89, "sitar": 104, "pipa": 104, "banjo": 105, "koto": 107, "guzheng": 107,
    "fiddle": 110, "glockenspiel": 9, "celesta": 8, "music box": 10, "vibraphone": 11, "vibes": 11,
    "marimba": 12, "xylophone": 13, "bells": 14, "steel guitar": 25,
}

PRESETS = {
    "rock band": {"lead_melody": 29, "chord_bed": 30, "bass_line": 34, "arpeggio": 27, "counterline": 27},
    "pop band": {"lead_melody": 81, "chord_bed": 0, "bass_line": 33, "pad": 89, "arpeggio": 25},
    "jazz trio": {"chord_bed": 0, "bass_line": 32, "walking_bass": 32, "lead_melody": 0, "counterline": 0},
    "jazz quartet": {"lead_melody": 66, "chord_bed": 0, "bass_line": 32, "walking_bass": 32, "counterline": 56},
    "string quartet": {"lead_melody": 40, "counterline": 40, "harmony": 41, "chord_bed": 41, "pad": 48, "high_texture": 40,
                       "bass_line": 42, "arpeggio": 41, "low_texture": 42, "octave_doubles": 40},
    "orchestral": {"lead_melody": 73, "counterline": 60, "chord_bed": 48, "pad": 48, "bass_line": 43,
                   "arpeggio": 46, "harmony": 71, "octave_doubles": 40, "high_texture": 9, "low_texture": 42},
    "piano solo": {r: 0 for r in ("lead_melody", "chord_bed", "bass_line", "walking_bass", "arpeggio", "counterline",
                                  "harmony", "octave_doubles", "pad", "high_texture", "low_texture")},
    "country band": {"lead_melody": 110, "chord_bed": 25, "bass_line": 32, "arpeggio": 105, "counterline": 22},
    "electronic": {"lead_melody": 81, "chord_bed": 90, "bass_line": 38, "pad": 89, "arpeggio": 81, "high_texture": 98},
    "chinese traditional": {"lead_melody": 73, "chord_bed": 107, "arpeggio": 104, "counterline": 104, "pad": 48,
                            "bass_line": 32},
}

# Which generated roles can take an instrument from a free-text instrumentation list, in order.
ROLE_PREFERENCES = {
    "lead_melody": [73, 66, 65, 56, 40, 81, 29, 110, 74, 71, 22, 104, 0, 25, 26],
    "chord_bed": [0, 4, 25, 24, 27, 30, 16, 6, 107, 26],
    "bass_line": [33, 32, 34, 38, 43, 42, 58],
    "walking_bass": [32, 33, 43],
    "pad": [48, 89, 52, 61, 16, 49],
    "arpeggio": [46, 25, 24, 104, 107, 0, 27, 6],
    "counterline": [71, 70, 60, 42, 41, 66, 57, 40, 74],
    "harmony": [48, 41, 71, 65, 61],
    "octave_doubles": [48, 40, 73, 61],
    "high_texture": [9, 8, 10, 11, 13, 14, 72],
    "low_texture": [42, 43, 89, 48],
}


def parse_instrumentation(text: str | None) -> tuple[dict[str, int], list[int], list[str]]:
    """Parse free text ('piano, bass, drums, strings' or a preset name).

    Returns (role->program for presets, list of requested programs, warnings).
    """
    if not text or not text.strip():
        return {}, [], []
    t = text.strip().lower()
    if t in PRESETS:
        return dict(PRESETS[t]), [], []
    programs, warnings = [], []
    for part in [p.strip() for p in t.replace(";", ",").replace("/", ",").replace(" and ", ",").split(",") if p.strip()]:
        if part in ("drums", "drum", "drum kit", "percussion"):
            continue
        if part.isdigit():
            programs.append(max(0, min(127, int(part))))
            continue
        match = None
        for word in sorted(INSTRUMENT_WORDS, key=len, reverse=True):
            if word in part:
                match = INSTRUMENT_WORDS[word]
                break
        if match is None:
            warnings.append(f"Unknown instrument '{part}' ignored.")
        else:
            programs.append(match)
    return {}, programs, warnings


def wants_drums(text: str | None) -> bool | None:
    if not text:
        return None
    t = text.lower()
    if any(w in t for w in ("drum", "percussion", "band", "electronic")):
        return True
    return False
