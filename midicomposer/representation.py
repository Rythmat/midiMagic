"""Root-relative, bar/beat-relative musical representation (reversible).

    root_reference_midi = 60 + root_pitch_class          # tonic in MIDI octave 4
    relative_pitch      = original_midi_pitch - root_reference_midi
    absolute_midi_pitch = relative_pitch + target_root_reference_midi

Percussion is never root-normalized: drum notes keep their GM note numbers and have
`relative_pitch = None`, `is_drum = True`.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from .midi_io import BarGrid, KeySignature, Note, Score, TempoChange, TimeSignature, Track
from .root import RootAnalysis, RootUndeterminedError, detect_root
from .theory import PITCH_NAMES, instrument_range, parse_key

STEPS_PER_QUARTER = 12          # grid resolution: covers 16ths and 8th-note triplets


def root_reference_midi(root_pc: int) -> int:
    return 60 + (root_pc % 12)


@dataclass
class NormNote:
    onset: float                    # quarter-note beats from piece start (exact)
    duration: float                 # quarter-note beats
    bar: int
    beat: int                       # metric beat index within the bar
    sub_beat: int                   # grid steps within that metric beat
    position: int                   # grid steps from bar start
    relative_pitch: int | None
    relative_pitch_class: int | None
    relative_octave: int | None
    original_pitch: int             # absolute MIDI pitch / drum note (for debugging + reversibility)
    velocity: int
    track_id: int
    program: int
    role: str | None
    is_drum: bool
    section: str | None = None
    chord: str | None = None


@dataclass
class NormTrack:
    track_id: int
    name: str
    program: int
    channel: int
    is_drum: bool
    role: str | None
    notes: list[NormNote] = field(default_factory=list)
    locked: bool = False
    source: str = "input"


@dataclass
class NormalizedPiece:
    root_name: str
    root_pitch_class: int
    root_reference_midi: int
    mode: str
    root_confidence: float
    root_detection_method: str
    ambiguous: bool
    tracks: list[NormTrack]
    tempos: list[TempoChange]
    time_signatures: list[TimeSignature]
    key_signatures: list[KeySignature]
    ticks_per_beat: int
    sections: list[dict] = field(default_factory=list)

    def condition_tokens(self) -> list[str]:
        """Sequence-level root/mode condition (the root label is not repeated per note)."""
        return [f"<ROOT={self.root_name}>", f"<MODE={self.mode}>"]

    def root_metadata(self) -> dict:
        return {
            "root_name": self.root_name, "root_pitch_class": self.root_pitch_class,
            "root_reference_midi": self.root_reference_midi, "mode": self.mode,
            "root_confidence": self.root_confidence, "root_detection_method": self.root_detection_method,
            "ambiguous": self.ambiguous,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def _timing(grid: BarGrid, onset: float) -> tuple[int, int, int, int]:
    bar, offset = grid.locate(onset)
    position = int(round(offset * STEPS_PER_QUARTER))
    unit = grid.beat_unit(bar)
    beat = int(offset // unit + 1e-9)
    sub_beat = int(round((offset - beat * unit) * STEPS_PER_QUARTER))
    return bar, beat, sub_beat, position


def normalize_score(
    score: Score,
    root: RootAnalysis | None = None,
    root_override: str | int | None = None,
    roles: dict[int, str] | None = None,
    sections: list[dict] | None = None,
) -> NormalizedPiece:
    """Convert a Score into root-relative NormNotes. Raises RootUndeterminedError if the root
    cannot be determined and no override is given (it never silently assumes C)."""
    if root is None or root_override is not None:
        root = detect_root(score, override=root_override)
    if root.root_pitch_class is None:
        raise RootUndeterminedError(
            "Tonal root could not be determined (no pitched content or flat distribution); "
            "supply a manual root override.")
    ref = root_reference_midi(root.root_pitch_class)
    grid = score.bar_grid()
    sec_bounds = [(s["start_beat"], s["end_beat"], s.get("label")) for s in (sections or [])]

    def section_of(t: float) -> str | None:
        for s, e, lab in sec_bounds:
            if s <= t < e:
                return lab
        return None

    tracks: list[NormTrack] = []
    for t in score.tracks:
        role = (roles or {}).get(t.track_id, t.role)
        nt = NormTrack(t.track_id, t.name, t.program, t.channel, t.is_drum, role, locked=t.locked, source=t.source)
        for n in t.sorted_notes():
            bar, beat, sub, pos = _timing(grid, n.start)
            if t.is_drum:
                rel = rel_pc = rel_oct = None
            else:
                rel = n.pitch - ref
                rel_pc, rel_oct = rel % 12, rel // 12
            nt.notes.append(NormNote(
                onset=n.start, duration=n.duration, bar=bar, beat=beat, sub_beat=sub, position=pos,
                relative_pitch=rel, relative_pitch_class=rel_pc, relative_octave=rel_oct,
                original_pitch=n.pitch, velocity=n.velocity, track_id=t.track_id, program=t.program,
                role=role, is_drum=t.is_drum, section=section_of(n.start)))
        tracks.append(nt)
    return NormalizedPiece(
        root_name=PITCH_NAMES[root.root_pitch_class], root_pitch_class=root.root_pitch_class,
        root_reference_midi=ref, mode=root.mode or "major", root_confidence=root.root_confidence,
        root_detection_method=root.root_detection_method, ambiguous=root.ambiguous, tracks=tracks,
        tempos=list(score.tempos), time_signatures=list(score.time_signatures),
        key_signatures=list(score.key_signatures), ticks_per_beat=score.ticks_per_beat,
        sections=list(sections or []))


def _resolve_root_pc(target_root: str | int | None, default: int) -> int:
    if target_root is None:
        return default
    if isinstance(target_root, int):
        return target_root % 12
    return parse_key(str(target_root))[0]


def denormalize(piece: NormalizedPiece, target_root: str | int | None = None,
                octave_offset: int = 0) -> tuple[Score, list[str]]:
    """Map relative pitches back to MIDI: abs = rel + (60 + target_pc) + 12*octave_offset.

    Out-of-MIDI-range notes are moved by octaves (never clipped); each such move is reported.
    Drum notes are copied unchanged.
    """
    target_pc = _resolve_root_pc(target_root, piece.root_pitch_class)
    ref = root_reference_midi(target_pc) + 12 * octave_offset
    issues: list[str] = []
    tracks: list[Track] = []
    for nt in piece.tracks:
        notes = []
        for nn in nt.notes:
            if nt.is_drum or nn.relative_pitch is None:
                pitch = nn.original_pitch
            else:
                pitch = nn.relative_pitch + ref
                if not 0 <= pitch <= 127:
                    moved = pitch
                    while moved > 127:
                        moved -= 12
                    while moved < 0:
                        moved += 12
                    issues.append(f"track {nt.track_id}: pitch {pitch} outside MIDI range moved to {moved} by octave")
                    pitch = moved
            notes.append(Note(nn.onset, nn.duration, pitch, nn.velocity))
        tracks.append(Track(nt.track_id, nt.name, nt.program, nt.channel, nt.is_drum, notes,
                            role=nt.role, locked=nt.locked, source=nt.source))
    key_sigs = []
    if piece.key_signatures or target_root is not None:
        mode = piece.mode if piece.mode in ("major", "minor") else ("minor" if piece.mode in
                                                                     ("dorian", "phrygian", "locrian", "harmonic_minor") else "major")
        name = PITCH_NAMES[target_pc].replace("A#", "Bb").replace("D#", "Eb").replace("G#", "Ab")
        key_sigs = [KeySignature(0.0, name + ("m" if mode == "minor" else ""))]
    score = Score(tracks=tracks, tempos=list(piece.tempos), time_signatures=list(piece.time_signatures),
                  key_signatures=key_sigs, ticks_per_beat=piece.ticks_per_beat)
    return score, issues


def nearest_octave_offset(source_pc: int, target_pc: int) -> int:
    """Octave offset that keeps a transposition within [-6, +5] semitones.

    Denormalization shifts pitches by (target_pc - source_pc) + 12 * offset, where the raw
    difference lies in [-11, 11].
    """
    raw = (target_pc % 12) - (source_pc % 12)
    if raw > 5:
        return -1
    if raw < -6:
        return 1
    return 0


def transpose_score(score: Score, target_root: str | int, source_root: RootAnalysis | None = None,
                    register: str = "nearest") -> tuple[Score, dict]:
    """Transpose via normalize -> denormalize, preserving intervals, rhythm, durations, structure.

    register='canonical' uses the exact spec formula (target reference in octave 4);
    register='nearest' adds an octave offset so the shift stays within a tritone.
    """
    piece = normalize_score(score, root=source_root)
    target_pc = _resolve_root_pc(target_root, piece.root_pitch_class)
    offset = nearest_octave_offset(piece.root_pitch_class, target_pc) if register == "nearest" else 0
    out, issues = denormalize(piece, target_pc, octave_offset=offset)
    for t_out, t_in in zip(out.tracks, score.tracks):
        t_out.locked, t_out.source, t_out.role = t_in.locked, t_in.source, t_in.role
    return out, {"source_root": piece.root_name, "target_root": PITCH_NAMES[target_pc],
                 "semitones": (target_pc - piece.root_pitch_class) + 12 * offset,
                 "octave_offset": offset, "issues": issues}


def fit_track_range(track: Track) -> list[str]:
    """Move a pitched track into its instrument range by whole octaves (keeps intervals), then
    fold remaining outliers by octave. Returns a list of actions/unresolved problems."""
    if track.is_drum or not track.notes:
        return []
    lo, hi = instrument_range(track.program)
    actions: list[str] = []
    pitches = [n.pitch for n in track.notes]

    def inside(shift: int) -> int:
        return sum(lo <= p + shift <= hi for p in pitches)

    best = max(range(-4, 5), key=lambda k: (inside(12 * k), -abs(k)))
    if best != 0 and inside(12 * best) > inside(0):
        for n in track.notes:
            n.pitch += 12 * best
        actions.append(f"track {track.track_id}: shifted {best:+d} octave(s) into instrument range {lo}-{hi}")
    folded = unresolved = 0
    for n in track.notes:
        if lo <= n.pitch <= hi:
            continue
        p = n.pitch
        while p > hi and p - 12 >= 0:
            p -= 12
        while p < lo and p + 12 <= 127:
            p += 12
        if lo <= p <= hi:
            n.pitch = p
            folded += 1
        else:
            unresolved += 1
    if folded:
        actions.append(f"track {track.track_id}: {folded} note(s) folded by octave into range")
    if unresolved:
        actions.append(f"track {track.track_id}: UNRESOLVED {unresolved} note(s) outside range {lo}-{hi}")
    return actions


def median_pitch(notes: list[Note]) -> float:
    return statistics.median([n.pitch for n in notes]) if notes else 60.0
