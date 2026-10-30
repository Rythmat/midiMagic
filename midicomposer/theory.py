"""Music-theory primitives: pitch names, keys, scales, chords, GM instruments and ranges.

Everything harmonic in this package is expressed *relative to the tonal root*
(0 = tonic pitch class), so these helpers mostly work in root-relative space.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_NAME_TO_PC = {}
for _i, (_s, _f) in enumerate(zip(PITCH_NAMES, FLAT_NAMES)):
    _NAME_TO_PC[_s.lower()] = _i
    _NAME_TO_PC[_f.lower()] = _i
_NAME_TO_PC.update({"cb": 11, "fb": 4, "e#": 5, "b#": 0})

# Scale intervals (semitones above the tonic).
MODES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
}
MODE_ALIASES = {
    "maj": "major", "ionian": "major", "min": "minor", "m": "minor", "aeolian": "minor",
    "natural_minor": "minor", "harmonic minor": "harmonic_minor",
}
# Which base (major/minor) family each mode belongs to; used for profiles and tokens.
MODE_FAMILY = {
    "major": "major", "lydian": "major", "mixolydian": "major",
    "minor": "minor", "dorian": "minor", "phrygian": "minor", "locrian": "minor",
    "harmonic_minor": "minor",
}

CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
}
TRIAD_OF = {"7": "maj", "maj7": "maj", "6": "maj", "min7": "min", "min6": "min",
            "m7b5": "dim", "dim7": "dim"}

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]


def pc_name(pc: int, prefer_flats: bool = False) -> str:
    return (FLAT_NAMES if prefer_flats else PITCH_NAMES)[pc % 12]


def note_name(midi_pitch: int) -> str:
    """MIDI note number -> scientific pitch name (C4 = 60)."""
    return f"{PITCH_NAMES[midi_pitch % 12]}{midi_pitch // 12 - 1}"


def parse_pitch_class(name: str) -> int:
    key = name.strip().lower().replace("♯", "#").replace("♭", "b")
    if key not in _NAME_TO_PC:
        raise ValueError(f"Unknown pitch name: {name!r}")
    return _NAME_TO_PC[key]


def normalize_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    m = mode.strip().lower().replace("-", "_")
    if not m:
        return None
    m = MODE_ALIASES.get(m, m)
    m = m.replace(" ", "_")
    if m not in MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Known: {sorted(MODES)}")
    return m


_KEY_RE = re.compile(r"^\s*([A-Ga-g](?:#|b|♯|♭)?)\s*(m(?![a-z])|[A-Za-z_ \-]+)?\s*$")


def parse_key(text: str) -> tuple[int, str | None]:
    """Parse 'D', 'Bb minor', 'F#m', 'A dorian', 'Am' -> (pitch_class, mode or None)."""
    m = _KEY_RE.match(text)
    if not m:
        raise ValueError(f"Cannot parse key: {text!r}")
    pc = parse_pitch_class(m.group(1))
    mode_txt = m.group(2)
    mode = normalize_mode(mode_txt) if mode_txt and mode_txt.strip() else None
    return pc, mode


def scale_pcs(mode: str) -> tuple[int, ...]:
    return MODES[normalize_mode(mode) or "major"]


def in_scale(rel_pc: int, mode: str) -> bool:
    return rel_pc % 12 in scale_pcs(mode)


def diatonic_chord(degree: int, mode: str, seventh: bool = False) -> tuple[int, str]:
    """Chord built on scale degree (0-based) of `mode`. Returns (root_rel_pc, quality)."""
    sc = scale_pcs(mode)
    root = sc[degree % 7]
    third = (sc[(degree + 2) % 7] - root) % 12
    fifth = (sc[(degree + 4) % 7] - root) % 12
    if third == 4 and fifth == 7:
        q = "maj"
    elif third == 3 and fifth == 7:
        q = "min"
    elif third == 3 and fifth == 6:
        q = "dim"
    elif third == 4 and fifth == 8:
        q = "aug"
    else:
        q = "maj"
    if seventh:
        sev = (sc[(degree + 6) % 7] - root) % 12
        q = {("maj", 11): "maj7", ("maj", 10): "7", ("min", 10): "min7",
             ("dim", 10): "m7b5", ("dim", 9): "dim7", ("min", 11): "min7"}.get((q, sev), q)
    return root, q


_ROMAN_RE = re.compile(r"^(b|#)?(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)(°|o|\+|7|maj7|m7|ø7|sus4|sus2|6)*$")


def roman_to_chord(numeral: str, mode: str) -> tuple[int, str]:
    """Parse a roman numeral relative to `mode` -> (root_rel_pc, quality).

    Upper-case = major, lower-case = minor, '°' = diminished, '7' adds a seventh.
    'bVII' etc. are chromatic roots relative to the major scale.
    """
    m = _ROMAN_RE.match(numeral)
    if not m:
        raise ValueError(f"Bad roman numeral {numeral!r}")
    accidental, rn = m.group(1), m.group(2)
    suffix = numeral[m.end(2):]
    degree = ROMAN.index(rn.upper())
    if accidental:
        root = (MODES["major"][degree] + (-1 if accidental == "b" else 1)) % 12
    else:
        root = scale_pcs(mode)[degree]
    if "°" in suffix or "o" in suffix:
        q = "dim7" if "7" in suffix else "dim"
    elif "ø" in suffix:
        q = "m7b5"
    elif "+" in suffix:
        q = "aug"
    elif rn.isupper():
        q = "maj7" if "maj7" in suffix else ("7" if "7" in suffix else ("6" if "6" in suffix else "maj"))
    else:
        q = "min7" if "7" in suffix else ("min6" if "6" in suffix else "min")
    if "sus4" in suffix:
        q = "sus4"
    elif "sus2" in suffix:
        q = "sus2"
    return root, q


def chord_pcs(root_rel: int, quality: str) -> list[int]:
    return [(root_rel + i) % 12 for i in CHORD_QUALITIES[quality]]


def chord_label(root_rel: int, quality: str, tonic_pc: int | None = None) -> str:
    """Human label. With tonic_pc, return an absolute name (e.g. 'F#min'), else relative ('+6:min')."""
    if tonic_pc is None:
        return f"{root_rel % 12}:{quality}"
    return f"{PITCH_NAMES[(root_rel + tonic_pc) % 12]}{'' if quality == 'maj' else quality}"


def nearest_in_set(pitch: int, pcs: list[int] | tuple[int, ...], prefer: int = 0) -> int:
    """Nearest MIDI/relative pitch whose pitch class is in `pcs` (prefer: -1 down, +1 up, 0 either)."""
    pcs_set = {p % 12 for p in pcs}
    for d in range(0, 12):
        cands = [pitch + d, pitch - d] if prefer >= 0 else [pitch - d, pitch + d]
        for c in cands:
            if c % 12 in pcs_set:
                return c
    return pitch


# ----------------------------------------------------------------------------- GM instruments

GM_FAMILIES = [
    "piano", "chromatic_percussion", "organ", "guitar", "bass", "strings", "ensemble", "brass",
    "reed", "pipe", "synth_lead", "synth_pad", "synth_effects", "ethnic", "percussive", "sound_effects",
]
GM_NAMES = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet", "Celesta", "Glockenspiel",
    "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ",
    "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ", "Accordion", "Harmonica",
    "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar",
    "Guitar Harmonics", "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)",
    "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola",
    "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs",
    "Voice Oohs", "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2", "Soprano Sax", "Alto Sax",
    "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute",
    "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina", "Lead 1 (square)",
    "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)",
    "Lead 7 (fifths)", "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)",
    "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)",
    "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise", "Breath Noise",
    "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]

# Practical playable ranges (MIDI note numbers), by program. Generous but musically sane.
_FAMILY_RANGES = {
    "piano": (21, 108), "chromatic_percussion": (48, 108), "organ": (24, 96), "guitar": (40, 88),
    "bass": (28, 67), "strings": (28, 103), "ensemble": (28, 100), "brass": (34, 87),
    "reed": (36, 92), "pipe": (55, 108), "synth_lead": (36, 96), "synth_pad": (24, 96),
    "synth_effects": (24, 96), "ethnic": (36, 96), "percussive": (36, 96), "sound_effects": (24, 96),
}
_PROGRAM_RANGES = {
    40: (55, 103), 41: (48, 91), 42: (36, 76), 43: (28, 67), 45: (28, 96), 46: (24, 103), 47: (36, 60),
    56: (54, 86), 57: (40, 77), 58: (28, 58), 59: (54, 86), 60: (34, 77), 64: (56, 88), 65: (49, 81),
    66: (44, 76), 67: (36, 69), 68: (58, 91), 69: (52, 81), 70: (34, 75), 71: (50, 91), 72: (74, 108),
    73: (60, 96), 74: (60, 96), 9: (79, 108), 8: (60, 108), 13: (65, 108), 12: (45, 96), 11: (53, 89),
    14: (60, 77), 105: (48, 84), 110: (55, 100),
}


def gm_family(program: int) -> str:
    return GM_FAMILIES[max(0, min(127, program)) // 8]


def gm_name(program: int, is_drum: bool = False) -> str:
    return "Drum Kit" if is_drum else GM_NAMES[max(0, min(127, program))]


def instrument_range(program: int, is_drum: bool = False) -> tuple[int, int]:
    if is_drum:
        return (27, 87)
    if program in _PROGRAM_RANGES:
        return _PROGRAM_RANGES[program]
    return _FAMILY_RANGES[gm_family(program)]


# ----------------------------------------------------------------------------- GM drums

DRUM = {
    "kick": 36, "kick2": 35, "side_stick": 37, "snare": 38, "clap": 39, "snare2": 40,
    "low_tom": 45, "mid_tom": 47, "high_tom": 50, "floor_tom": 43,
    "closed_hat": 42, "pedal_hat": 44, "open_hat": 46, "crash": 49, "ride": 51, "ride_bell": 53,
    "tambourine": 54, "cowbell": 56, "crash2": 57, "shaker": 70, "brush": 38,
}
DRUM_GROUPS = {
    "kick": {35, 36}, "snare": {37, 38, 39, 40}, "hat": {42, 44, 46}, "ride": {51, 53, 59},
    "crash": {49, 52, 55, 57}, "tom": {41, 43, 45, 47, 48, 50}, "other": set(),
}


def drum_group(note: int) -> str:
    for g, s in DRUM_GROUPS.items():
        if note in s:
            return g
    return "other"


@dataclass(frozen=True)
class KeySpec:
    root_pc: int
    mode: str

    @property
    def name(self) -> str:
        return f"{PITCH_NAMES[self.root_pc]} {self.mode}"
