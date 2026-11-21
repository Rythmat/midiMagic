"""Fixed test sets for the evaluation plan (proposal slide 7).

Test 1 uses melodic lines, test 2 uses text prompts and chord progressions. Everything here is fixed so the
evaluation is repeatable; the melodies are simplified public-domain folk/classical tunes written out by hand.
Notes are (pitch, duration in quarter-note beats), played back to back.
"""
from __future__ import annotations

from midicomposer.midi_io import Note, Score, TimeSignature, TempoChange, Track

C4, D4, E4, F4, Fs4, G4, Gs4, A4, Bb4, B4 = 60, 62, 64, 65, 66, 67, 68, 69, 70, 71
C5, D5, E5, F5 = 72, 74, 76, 77

# ----------------------------------------------------------------------------- test 1: melodic lines

_ODE = [(E4, 1), (E4, 1), (F4, 1), (G4, 1), (G4, 1), (F4, 1), (E4, 1), (D4, 1), (C4, 1), (C4, 1), (D4, 1), (E4, 1),
        (E4, 1.5), (D4, 0.5), (D4, 2)]
_ODE2 = _ODE[:12] + [(D4, 1.5), (C4, 0.5), (C4, 2)]
_TWINKLE_A = [(C4, 1), (C4, 1), (G4, 1), (G4, 1), (A4, 1), (A4, 1), (G4, 2), (F4, 1), (F4, 1), (E4, 1), (E4, 1),
              (D4, 1), (D4, 1), (C4, 2)]
_TWINKLE_B = [(G4, 1), (G4, 1), (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 2)] * 2
_FRERE = ([(F4, 1), (G4, 1), (A4, 1), (F4, 1)] * 2 + [(A4, 1), (Bb4, 1), (C5, 2)] * 2
          + [(C5, .5), (D5, .5), (C5, .5), (Bb4, .5), (A4, 1), (F4, 1)] * 2 + [(F4, 1), (C4, 1), (F4, 2)] * 2)
_GRACE_A = [(G4, 2), (B4, .5), (G4, .5), (B4, 2), (A4, 1), (G4, 2), (E4, 1), (D4, 2), (D4, 1),
            (G4, 2), (B4, .5), (G4, .5), (B4, 2), (A4, 1)]
_GRACE = _GRACE_A + [(D5, 3), (D5, 3)] + _GRACE_A + [(G4, 3), (G4, 3)]
_SAILOR = ([(A4, 1), (A4, .5), (A4, .5), (A4, 1), (A4, .5), (A4, .5), (A4, 1), (D4, 1), (F4, 1), (A4, 1),
            (G4, 1), (G4, .5), (G4, .5), (G4, 1), (G4, .5), (G4, .5), (G4, 1), (C4, 1), (E4, 1), (G4, 1),
            (A4, 1), (A4, .5), (A4, .5), (A4, 1), (A4, .5), (A4, .5), (A4, 1), (B4, 1), (C5, 1), (D5, 1),
            (C5, 1), (A4, 1), (G4, 1), (E4, 1), (D4, 2), (D4, 2)])
_GREEN_A = [(C5, 2), (D5, 1), (E5, 1.5), (F5, .5), (E5, 1), (D5, 2), (B4, 1), (G4, 1.5), (A4, .5), (B4, 1)]
_GREEN = (_GREEN_A + [(C5, 2), (A4, 1), (A4, 1.5), (Gs4, .5), (A4, 1), (B4, 2), (Gs4, 1), (E4, 2), (A4, 1)]
          + _GREEN_A + [(C5, 1.5), (B4, .5), (A4, 1), (Gs4, 1.5), (Fs4, .5), (Gs4, 1), (A4, 3), (A4, 3)])

# name, notes, meter, tempo, expected tonic pitch class, expected mode family
MELODIES = [
    ("ode_to_joy", _ODE + _ODE2, (4, 4), 112, 0, "major"),
    ("twinkle_twinkle", _TWINKLE_A + _TWINKLE_B + _TWINKLE_A, (4, 4), 100, 0, "major"),
    ("frere_jacques", _FRERE, (4, 4), 108, 5, "major"),
    ("amazing_grace", _GRACE, (3, 4), 80, 7, "major"),
    ("drunken_sailor", _SAILOR, (4, 4), 132, 2, "minor"),        # D dorian: a minor-family tonic of D
    ("greensleeves", _GREEN, (3, 4), 90, 9, "minor"),
]


# Moods a listener could reasonably hear in each melody (a reading is "plausible" if its top emotion is one of these).
# Labelled blind, without seeing the reader's outputs or the earlier developer labels; rule: 1-3 moods each.
MELODY_MOODS = {
    "ode_to_joy": {"happy", "magnificent"},
    "twinkle_twinkle": {"happy", "quiet"},
    "frere_jacques": {"happy", "funny"},
    "amazing_grace": {"warm", "quiet", "sad"},
    "drunken_sailor": {"exciting", "funny"},
    "greensleeves": {"sad", "romantic"},
}


def line_to_score(notes, meter=(4, 4), tempo=100, program=73, name="Melody", velocity=88) -> Score:
    out, t = [], 0.0
    for pitch, dur in notes:
        out.append(Note(t, dur * 0.95, pitch, velocity))
        t += dur
    return Score(tracks=[Track(0, name, program, 0, False, out, role="lead")], tempos=[TempoChange(0.0, tempo)],
                 time_signatures=[TimeSignature(0.0, *meter)])


# ----------------------------------------------------------------------------- test 2: prompts and chords

# prompt, expected planner settings (only the fields a reader would agree the prompt states)
PROMPTS = [
    ("A sad, slow piano and cello piece in D minor, 16 bars",
     {"emotion": "sad", "key": "D", "scale_mode": "minor", "bars": 16}),
    ("Upbeat summer pop song with guitar at 128 bpm", {"genre": "pop", "emotion": "exciting", "tempo": 128}),
    ("Epic heroic orchestral theme for a movie trailer", {"genre": "classical", "emotion": "magnificent"}),
    ("A lazy jazz trio tune in Bb for a Sunday morning", {"genre": "jazz", "emotion": "lazy", "key": "A#"}),
    ("Angry heavy rock riff in E minor, fast", {"genre": "rock", "emotion": "angry", "key": "E", "scale_mode": "minor"}),
    ("Warm cozy country ballad in G major", {"genre": "country", "emotion": "warm", "key": "G", "scale_mode": "major"}),
    ("Creepy, tense music for a horror scene", {"emotion": "fear"}),
    ("Romantic love song with strings, 90 bpm", {"emotion": "romantic", "tempo": 90}),
    ("Calm and peaceful meditation music, minimal", {"emotion": "quiet", "density": 0.2}),
    ("A playful, silly cartoon tune in C", {"emotion": "funny", "key": "C"}),
    ("Happy traditional folk dance in A, 32 bars", {"genre": "traditional", "emotion": "happy", "key": "A", "bars": 32}),
    ("Exciting energetic rock anthem with drums", {"genre": "rock", "emotion": "exciting"}),
]

# name, roman-numeral chords (one per bar), key pc, mode, meter, tempo, intended genre/emotion for generation
_DEG = {"major": {"I": (0, "maj"), "ii": (2, "min"), "iii": (4, "min"), "IV": (5, "maj"), "V": (7, "maj"),
                  "vi": (9, "min"), "V7": (7, "dom7"), "ii7": (2, "min7"), "Imaj7": (0, "maj7"), "I7": (0, "dom7"),
                  "IV7": (5, "dom7")},
        "minor": {"i": (0, "min"), "iv": (5, "min"), "v": (7, "min"), "III": (3, "maj"), "VI": (8, "maj"),
                  "VII": (10, "maj"), "V": (7, "maj")}}
_SHAPES = {"maj": (0, 4, 7), "min": (0, 3, 7), "dom7": (0, 4, 7, 10), "min7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11)}

PROGRESSIONS = [
    ("pop_I_V_vi_IV_in_C", ["I", "V", "vi", "IV"] * 4, 0, "major", 110, "pop", "happy"),
    ("jazz_ii_V_I_in_F", ["ii7", "V7", "Imaj7", "Imaj7"] * 4, 5, "major", 120, "jazz", "lazy"),
    ("minor_i_VI_III_VII_in_A", ["i", "VI", "III", "VII"] * 4, 9, "minor", 96, "rock", "sad"),
    ("blues_12_bar_in_G", ["I7", "I7", "I7", "I7", "IV7", "IV7", "I7", "I7", "V7", "IV7", "I7", "V7"], 7, "major", 100,
     "jazz", "warm"),
]


def progression_to_score(chords, key_pc, mode, tempo=100) -> Score:
    notes = []
    for bar, sym in enumerate(chords):
        root, quality = _DEG[mode][sym]
        base = 48 + (key_pc + root) % 12                   # chord root in octave 3
        for iv in _SHAPES[quality]:
            p = base + iv
            notes.append(Note(bar * 4.0, 3.8, p if p >= 52 else p + 12, 76))
    return Score(tracks=[Track(0, "Chords", 0, 0, False, notes, role="chordal")], tempos=[TempoChange(0.0, tempo)],
                 time_signatures=[TimeSignature(0.0, 4, 4)])
