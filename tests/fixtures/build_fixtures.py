"""Build a tiny, deterministic XMIDI-style fixture dataset so tests never need the real XMIDI.

Includes deliberately awkward cases: multi-word labels, mixed case/separators, an exact
duplicate, a transposed duplicate, a corrupted file, an empty file, a drums-only file
(undeterminable root), and a whole-tone file (ambiguous root).

    python tests/fixtures/build_fixtures.py [out_dir]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from midicomposer.midi_io import Note, Score, TempoChange, TimeSignature, Track, save_midi  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "xmidi_mini"

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
PROGS = {"major": [(0, "I"), (5, "IV"), (7, "V"), (0, "I")], "minor": [(0, "i"), (5, "iv"), (7, "V"), (0, "i")]}


def tonal_piece(root_pc: int, mode: str, bars: int = 8, bpm: float = 120.0, seed: int = 0,
                with_drums: bool = True, swing: bool = False) -> Score:
    """A simple but unambiguous tonal piece: melody + block chords + bass (+ drums)."""
    scale = MAJOR if mode == "major" else MINOR
    tonic = 60 + root_pc
    third = 4 if mode == "major" else 3
    melody, chords, bass, drums = [], [], [], []
    prog = PROGS[mode]
    contour = [0, 2, 4, 2, 4, 5, 4, 2, 0, 1, 2, 4, 3, 2, 1, 0]
    for b in range(bars):
        start = b * 4.0
        croot, _ = prog[b % 4]
        q3 = 4 if (mode == "major" or croot == 7) else third   # I IV V / i iv V
        for p in (croot, croot + q3, croot + 7):
            chords.append(Note(start, 4.0, tonic - 12 + p, 70))
        bass.append(Note(start, 2.0, tonic - 24 + croot, 90))
        bass.append(Note(start + 2.0, 2.0, tonic - 24 + croot + 7, 85))
        for i in range(4):
            deg = contour[(b * 4 + i + seed) % len(contour)]
            pitch = tonic + scale[deg % 7] + 12 * (deg // 7)
            if b == bars - 1 and i == 3:
                pitch = tonic
            if swing and i % 2:
                # Swung pair: long-short with the short note on the triplet position.
                melody.append(Note(start + i, 0.667, pitch, 96))
                melody.append(Note(start + i + 0.667, 0.333, tonic + scale[(deg + 1) % 7], 80))
            else:
                melody.append(Note(start + i, 1.0, pitch, 96))
        if with_drums:
            for i in range(4):
                drums.append(Note(start + i, 0.25, 36 if i % 2 == 0 else 38, 100))
                drums.append(Note(start + i + 0.5, 0.25, 42, 70))
    tracks = [Track(0, "lead", 73, 0, False, melody), Track(1, "piano", 0, 1, False, chords),
              Track(2, "bass", 33, 2, False, bass)]
    if with_drums:
        tracks.append(Track(3, "drum", 0, 9, True, drums))
    return Score(tracks=tracks, tempos=[TempoChange(0.0, bpm)], time_signatures=[TimeSignature(0.0, 4, 4)])


def drums_only() -> Score:
    notes = [Note(i * 0.5, 0.25, [36, 42, 38, 42][i % 4], 100) for i in range(32)]
    return Score(tracks=[Track(0, "drum", 0, 9, True, notes)])


def whole_tone() -> Score:
    pcs = [0, 2, 4, 6, 8, 10]
    notes = [Note(i * 1.0, 1.0, 60 + pcs[i % 6], 90) for i in range(48)]
    return Score(tracks=[Track(0, "lead", 0, 0, False, notes)])


SPECS = [
    # (filename, kind, args)
    ("XMIDI_happy_pop_AB12CD34.midi", "tonal", dict(root_pc=2, mode="major")),
    ("XMIDI_happy_pop_ZZ99YY88.midi", "tonal", dict(root_pc=7, mode="major", seed=3)),
    ("XMIDI_sad_classical_QW12ER34.midi", "tonal", dict(root_pc=9, mode="minor", with_drums=False, bpm=72)),
    ("XMIDI_sad_rock_PL09OK87.midi", "tonal", dict(root_pc=4, mode="minor", bpm=96)),
    ("XMIDI_exciting_rock_MN34BV56.midi", "tonal", dict(root_pc=4, mode="major", bpm=140)),
    ("XMIDI_warm_jazz_JH76GF54.midi", "tonal", dict(root_pc=5, mode="major", bpm=110, swing=True)),
    ("XMIDI_Romantic_Country_TY67UI89.MID", "tonal", dict(root_pc=7, mode="major", seed=5)),
    ("XMIDI_very_happy_hip_hop_QA11WS22.midi", "tonal", dict(root_pc=1, mode="minor", seed=2)),
    ("XMIDI_angry_rock_n_roll_ED33RF44.midi", "tonal", dict(root_pc=11, mode="minor", bpm=150)),
    ("XMIDI_quiet-calm_new age_TG55YH66.midi", "tonal", dict(root_pc=3, mode="major", with_drums=False)),
    ("XMIDI_lazy_jazz_UJ77IK88.midi", "tonal", dict(root_pc=10, mode="major", seed=7)),
    ("XMIDI_fear_classical_OL99PM00.midi", "tonal", dict(root_pc=0, mode="minor", with_drums=False, seed=1)),
    ("XMIDI_funny_pop_RF12TG34.midi", "tonal", dict(root_pc=0, mode="major", seed=4)),
    ("XMIDI_magnificent_classical_YH56UJ78.midi", "tonal", dict(root_pc=2, mode="major", with_drums=False, seed=6)),
    ("XMIDI_happy_traditional_IK90OL12.midi", "tonal", dict(root_pc=7, mode="major", seed=8)),
    ("XMIDI_sad_pop_WS34ED56.midi", "tonal", dict(root_pc=6, mode="minor", seed=9)),
    # Leakage cases: exact duplicate (same bytes) and transposed duplicate of the first file.
    ("XMIDI_happy_pop_DUP00001.midi", "copy", dict(of="XMIDI_happy_pop_AB12CD34.midi")),
    ("XMIDI_happy_pop_TRN00001.midi", "tonal", dict(root_pc=5, mode="major")),
    # Problem files.
    ("XMIDI_exciting_pop_DRUMS001.midi", "drums", {}),
    ("XMIDI_fear_jazz_WHOLE001.midi", "wholetone", {}),
    ("XMIDI_sad_jazz_CORRUPT1.midi", "corrupt", {}),
    ("XMIDI_happy_rock_EMPTY001.midi", "empty", {}),
    ("not_an_xmidi_name.mid", "tonal", dict(root_pc=0, mode="major")),
]


def build(out_dir: Path = DEFAULT_OUT) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "nested").mkdir(parents=True)
    for i, (name, kind, args) in enumerate(SPECS):
        # Put a few files in a nested folder to exercise recursive discovery.
        path = out_dir / ("nested" if i % 5 == 4 else "") / name
        if kind == "tonal":
            save_midi(tonal_piece(**args), path)
        elif kind == "copy":
            src = next(out_dir.rglob(args["of"]))
            shutil.copyfile(src, path)
        elif kind == "drums":
            save_midi(drums_only(), path)
        elif kind == "wholetone":
            save_midi(whole_tone(), path)
        elif kind == "corrupt":
            path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0MTrk\xff\xff\xff\xffgarbage")
        elif kind == "empty":
            path.write_bytes(b"")
    return out_dir


if __name__ == "__main__":
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
    print(f"Fixture dataset written to {out}")
