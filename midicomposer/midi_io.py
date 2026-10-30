"""Beat-based MIDI score model on top of mido.

Times are stored in quarter-note beats (floats derived from integer ticks) so that the
rest of the system works on a bar/beat grid instead of seconds. Tempo maps, time
signatures, key signatures and the source tick resolution are preserved so a file can
be round-tripped without timing drift.
"""
from __future__ import annotations

import copy
import io
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import mido

from .theory import gm_name

DRUM_CHANNEL = 9


class MidiLoadError(Exception):
    """Raised for empty, non-MIDI or corrupted input."""


@dataclass
class Note:
    start: float        # quarter-note beats from the start of the piece
    duration: float     # quarter-note beats
    pitch: int
    velocity: int

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Track:
    track_id: int
    name: str
    program: int
    channel: int
    is_drum: bool
    notes: list[Note] = field(default_factory=list)
    role: str | None = None
    locked: bool = False
    source: str = "input"           # "input" | "generated"

    @property
    def instrument_name(self) -> str:
        return gm_name(self.program, self.is_drum)

    def sorted_notes(self) -> list[Note]:
        return sorted(self.notes, key=lambda n: (n.start, n.pitch))


@dataclass
class TempoChange:
    beat: float
    bpm: float


@dataclass
class TimeSignature:
    beat: float
    numerator: int
    denominator: int

    @property
    def bar_beats(self) -> float:
        """Bar length in quarter-note beats."""
        return self.numerator * 4.0 / self.denominator

    @property
    def label(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass
class KeySignature:
    beat: float
    key: str            # mido notation: 'C', 'F#m', 'Bb', ...


@dataclass
class Score:
    tracks: list[Track] = field(default_factory=list)
    tempos: list[TempoChange] = field(default_factory=lambda: [TempoChange(0.0, 120.0)])
    time_signatures: list[TimeSignature] = field(default_factory=lambda: [TimeSignature(0.0, 4, 4)])
    key_signatures: list[KeySignature] = field(default_factory=list)
    ticks_per_beat: int = 480
    load_warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ basic properties
    def copy(self) -> "Score":
        return copy.deepcopy(self)

    @property
    def pitched_tracks(self) -> list[Track]:
        return [t for t in self.tracks if not t.is_drum]

    def all_notes(self, include_drums: bool = False) -> list[Note]:
        return [n for t in self.tracks if include_drums or not t.is_drum for n in t.notes]

    @property
    def end_beat(self) -> float:
        ends = [n.end for t in self.tracks for n in t.notes]
        return max(ends) if ends else 0.0

    @property
    def initial_bpm(self) -> float:
        return self.tempos[0].bpm if self.tempos else 120.0

    @property
    def initial_time_signature(self) -> TimeSignature:
        return self.time_signatures[0] if self.time_signatures else TimeSignature(0.0, 4, 4)

    def track_by_id(self, track_id: int) -> Track:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        raise KeyError(track_id)

    def next_track_id(self) -> int:
        return max((t.track_id for t in self.tracks), default=-1) + 1

    # ------------------------------------------------------------------ time conversion
    def beat_to_seconds(self, beat: float) -> float:
        tempos = sorted(self.tempos, key=lambda t: t.beat) or [TempoChange(0.0, 120.0)]
        seconds, prev_beat, bpm = 0.0, 0.0, tempos[0].bpm
        for tc in tempos:
            if tc.beat >= beat:
                break
            seconds += (tc.beat - prev_beat) * 60.0 / bpm
            prev_beat, bpm = tc.beat, tc.bpm
        return seconds + (beat - prev_beat) * 60.0 / bpm

    @property
    def duration_seconds(self) -> float:
        return self.beat_to_seconds(self.end_beat)

    def bar_grid(self) -> "BarGrid":
        return BarGrid(self.time_signatures, self.end_beat)


class BarGrid:
    """Maps quarter-note beats to (bar index, offset in bar) using the time-signature map."""

    def __init__(self, time_signatures: list[TimeSignature], end_beat: float, min_bars: int = 1):
        sigs = sorted(time_signatures, key=lambda s: s.beat) or [TimeSignature(0.0, 4, 4)]
        if sigs[0].beat > 0:
            sigs.insert(0, TimeSignature(0.0, sigs[0].numerator, sigs[0].denominator))
        self.bar_starts: list[float] = []
        self.bar_lengths: list[float] = []
        self.bar_sigs: list[TimeSignature] = []
        beat, si = 0.0, 0
        while beat < end_beat - 1e-9 or len(self.bar_starts) < min_bars:
            while si + 1 < len(sigs) and sigs[si + 1].beat <= beat + 1e-9:
                si += 1
            length = sigs[si].bar_beats
            # A time-signature change in the middle of a bar truncates the bar.
            if si + 1 < len(sigs) and sigs[si + 1].beat < beat + length - 1e-9:
                length = sigs[si + 1].beat - beat
            self.bar_starts.append(beat)
            self.bar_lengths.append(length)
            self.bar_sigs.append(sigs[si])
            beat += length
            if len(self.bar_starts) > 100000:
                break

    @property
    def n_bars(self) -> int:
        return len(self.bar_starts)

    def locate(self, beat: float) -> tuple[int, float]:
        i = bisect_right(self.bar_starts, beat + 1e-9) - 1
        i = max(0, i)
        if i >= len(self.bar_starts):
            i = len(self.bar_starts) - 1
        return i, beat - self.bar_starts[i]

    def bar_start(self, bar: int) -> float:
        if bar < len(self.bar_starts):
            return self.bar_starts[bar]
        # Extrapolate with the last bar length.
        extra = bar - len(self.bar_starts) + 1
        return self.bar_starts[-1] + self.bar_lengths[-1] * extra

    def bar_length(self, bar: int) -> float:
        return self.bar_lengths[min(bar, len(self.bar_lengths) - 1)]

    def beat_unit(self, bar: int) -> float:
        """Length of one metric beat (e.g. 0.5 quarter for x/8) in quarter notes."""
        sig = self.bar_sigs[min(bar, len(self.bar_sigs) - 1)]
        return 4.0 / sig.denominator


# ============================================================================ loading


def load_midi(source: str | Path | bytes, name_hint: str = "") -> Score:
    """Parse a MIDI file (path or bytes) into a Score. Raises MidiLoadError on bad input."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        if len(data) == 0:
            raise MidiLoadError("Uploaded file is empty.")
        if not data.startswith(b"MThd"):
            raise MidiLoadError("File is not a Standard MIDI File (missing 'MThd' header).")
        try:
            mf = mido.MidiFile(file=io.BytesIO(data), clip=True)
        except Exception as e:  # mido raises a variety of errors on corrupt data
            raise MidiLoadError(f"Corrupted MIDI data: {e}") from e
    else:
        path = Path(source)
        if not path.exists():
            raise MidiLoadError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise MidiLoadError("File is empty.")
        try:
            mf = mido.MidiFile(str(path), clip=True)
        except Exception as e:
            raise MidiLoadError(f"Corrupted MIDI file {path.name}: {e}") from e
    try:
        return _midifile_to_score(mf)
    except Exception as e:
        raise MidiLoadError(f"Could not interpret MIDI events{(' in ' + name_hint) if name_hint else ''}: {e}") from e


def _midifile_to_score(mf: mido.MidiFile) -> Score:
    tpq = mf.ticks_per_beat or 480
    if mf.type == 2:
        raise MidiLoadError("MIDI type 2 (asynchronous sequences) is not supported.")
    warnings: list[str] = []
    tempos: list[TempoChange] = []
    sigs: list[TimeSignature] = []
    keys: list[KeySignature] = []
    # Per (track index, channel) collected note data.
    groups: dict[tuple[int, int], dict] = {}
    channel_program: dict[int, int] = {}   # global fallback program per channel
    unclosed = zero_len = 0

    for ti, mtrack in enumerate(mf.tracks):
        tick = 0
        track_name = ""
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        local_program: dict[int, int] = {}
        for msg in mtrack:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "set_tempo":
                    tempos.append(TempoChange(tick / tpq, round(mido.tempo2bpm(msg.tempo), 4)))
                elif msg.type == "time_signature":
                    sigs.append(TimeSignature(tick / tpq, msg.numerator, msg.denominator))
                elif msg.type == "key_signature":
                    keys.append(KeySignature(tick / tpq, msg.key))
                elif msg.type == "track_name" and not track_name:
                    track_name = msg.name.strip()
                continue
            if msg.type == "program_change":
                local_program[msg.channel] = msg.program
                channel_program.setdefault(msg.channel, msg.program)
                g = groups.get((ti, msg.channel))
                if g is not None and g["program"] is None:
                    g["program"] = msg.program
                continue
            if msg.type not in ("note_on", "note_off"):
                continue
            ch = msg.channel
            g = groups.setdefault((ti, ch), {"name": track_name, "program": local_program.get(ch), "notes": []})
            if not g["name"]:
                g["name"] = track_name
            key = (ch, msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(key, []).append((tick, msg.velocity))
            else:
                stack = active.get(key)
                if stack:
                    on_tick, vel = stack.pop(0)   # FIFO pairing
                    if tick > on_tick:
                        g["notes"].append(Note(on_tick / tpq, (tick - on_tick) / tpq, msg.note, vel))
                    else:
                        zero_len += 1
        # Close notes left hanging at the end of the track.
        for (ch, pitch), stack in active.items():
            for on_tick, vel in stack:
                if tick > on_tick:
                    groups[(ti, ch)]["notes"].append(Note(on_tick / tpq, (tick - on_tick) / tpq, pitch, vel))
                unclosed += 1

    if unclosed:
        warnings.append(f"{unclosed} note(s) had no note-off and were closed at end of track.")
    if zero_len:
        warnings.append(f"{zero_len} zero-length note(s) were dropped.")

    tracks: list[Track] = []
    for (ti, ch), g in sorted(groups.items()):
        if not g["notes"]:
            continue
        program = g["program"] if g["program"] is not None else channel_program.get(ch, 0)
        is_drum = ch == DRUM_CHANNEL
        notes = sorted(g["notes"], key=lambda n: (n.start, n.pitch))
        name = g["name"] or ("Drums" if is_drum else gm_name(program))
        tracks.append(Track(track_id=len(tracks), name=name, program=program, channel=ch,
                            is_drum=is_drum, notes=notes))

    tempos = _dedupe_by_beat(sorted(tempos, key=lambda t: t.beat)) or [TempoChange(0.0, 120.0)]
    if tempos[0].beat > 0:
        tempos.insert(0, TempoChange(0.0, 120.0))
    sigs = _dedupe_by_beat(sorted(sigs, key=lambda s: s.beat)) or [TimeSignature(0.0, 4, 4)]
    if sigs[0].beat > 0:
        sigs.insert(0, TimeSignature(0.0, 4, 4))
    keys = _dedupe_by_beat(sorted(keys, key=lambda k: k.beat))
    return Score(tracks=tracks, tempos=tempos, time_signatures=sigs, key_signatures=keys,
                 ticks_per_beat=tpq, load_warnings=warnings)


def _dedupe_by_beat(items: list) -> list:
    """Keep the last event at each identical beat position."""
    out: list = []
    for it in items:
        if out and abs(out[-1].beat - it.beat) < 1e-9:
            out[-1] = it
        else:
            out.append(it)
    return out


# ============================================================================ writing


def assign_channels(score: Score) -> list[str]:
    """Give each track a valid channel: drums on 9, melodic tracks on 0-15 except 9.

    When there are more than 15 melodic tracks, tracks sharing a program share a channel;
    anything left over is reported.
    """
    warnings: list[str] = []
    free = [c for c in range(16) if c != DRUM_CHANNEL]
    by_program: dict[int, int] = {}
    for t in score.tracks:
        if t.is_drum:
            t.channel = DRUM_CHANNEL
            continue
        if free:
            t.channel = free.pop(0)
            by_program.setdefault(t.program, t.channel)
        elif t.program in by_program:
            t.channel = by_program[t.program]
            warnings.append(f"Track {t.track_id} shares channel {t.channel} (same program).")
        else:
            t.channel = 15
            warnings.append(f"Track {t.track_id} forced onto channel 15; channel/program conflict.")
    return warnings


def score_to_midifile(score: Score, ticks_per_beat: int | None = None) -> mido.MidiFile:
    tpq = ticks_per_beat or score.ticks_per_beat or 480
    mf = mido.MidiFile(type=1, ticks_per_beat=tpq)

    def to_tick(beat: float) -> int:
        return int(round(beat * tpq))

    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    meta: list[tuple[int, int, mido.MetaMessage]] = []
    for ts in score.time_signatures:
        meta.append((to_tick(ts.beat), 0, mido.MetaMessage("time_signature", numerator=ts.numerator,
                                                            denominator=ts.denominator, time=0)))
    for ks in score.key_signatures:
        try:
            meta.append((to_tick(ks.beat), 1, mido.MetaMessage("key_signature", key=ks.key, time=0)))
        except Exception:
            pass
    for tc in score.tempos:
        meta.append((to_tick(tc.beat), 2, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tc.bpm), time=0)))
    _append_delta(conductor, sorted(meta, key=lambda x: (x[0], x[1])))
    mf.tracks.append(conductor)

    for t in score.tracks:
        mt = mido.MidiTrack()
        mt.append(mido.MetaMessage("track_name", name=t.name[:60] or "Track", time=0))
        ch = DRUM_CHANNEL if t.is_drum else max(0, min(15, t.channel))
        events: list[tuple[int, int, mido.Message]] = []
        if not t.is_drum:
            events.append((0, -1, mido.Message("program_change", channel=ch, program=max(0, min(127, t.program)))))
        for n in t.notes:
            on, off = to_tick(n.start), to_tick(n.end)
            if off <= on:
                off = on + 1
            p = max(0, min(127, n.pitch))
            v = max(1, min(127, int(n.velocity)))
            # Sort key: note-offs before note-ons at the same tick, so repeated pitches re-trigger.
            events.append((on, 1, mido.Message("note_on", channel=ch, note=p, velocity=v)))
            events.append((off, 0, mido.Message("note_off", channel=ch, note=p, velocity=0)))
        _append_delta(mt, sorted(events, key=lambda x: (x[0], x[1])))
        mf.tracks.append(mt)
    return mf


def _append_delta(track: mido.MidiTrack, events: list[tuple[int, int, mido.Message]]) -> None:
    last = 0
    for tick, _, msg in events:
        track.append(msg.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))


def score_to_bytes(score: Score, ticks_per_beat: int | None = None) -> bytes:
    buf = io.BytesIO()
    score_to_midifile(score, ticks_per_beat).save(file=buf)
    return buf.getvalue()


def save_midi(score: Score, path: str | Path, ticks_per_beat: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(score_to_bytes(score, ticks_per_beat))
    return path
