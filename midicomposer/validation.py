"""Pre-export validation and light automatic repair of generated scores.

Locked/anchor tracks are never modified by repair; problems in them are only reported.
"""
from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import mido

from .harmony import ChordSpan, chord_at
from .midi_io import DRUM_CHANNEL, MidiLoadError, Score, Track, assign_channels, load_midi, score_to_bytes
from .representation import STEPS_PER_QUARTER, fit_track_range
from .roles import MONOPHONIC_TRACK_ROLES
from .theory import instrument_range, scale_pcs

MIN_DURATION = 1.0 / STEPS_PER_QUARTER
KEY_THRESHOLD = 0.8
CHROMATIC_ROLES_THRESHOLD = 0.65     # walking bass / counterlines use chromatic approach tones


@dataclass
class Issue:
    code: str
    severity: str                 # error | warning | info
    message: str
    track_id: int | None = None
    count: int = 1


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def add(self, code, severity, message, track_id=None, count=1):
        self.issues.append(Issue(code, severity, message, track_id, count))

    def to_dict(self) -> dict:
        return {"passed": self.passed, "n_errors": len(self.errors), "n_warnings": len(self.warnings),
                "issues": [asdict(i) for i in self.issues], "repairs": self.repairs}


def note_signature(track: Track, shift: int = 0) -> list[tuple]:
    return sorted((round(n.start, 6), round(n.duration, 6), n.pitch + (0 if track.is_drum else shift), n.velocity)
                  for n in track.notes)


# ----------------------------------------------------------------------------- repair


def repair_score(score: Score, locked_ids: set[int] | None = None) -> list[str]:
    """Fix invalid durations, velocities, duplicates, monophonic overlaps, off-grid timing,
    instrument range and channels on unlocked tracks. Returns a list of repair actions."""
    locked_ids = locked_ids or set()
    repairs: list[str] = []
    for t in score.tracks:
        if t.track_id in locked_ids or t.locked:
            continue
        fixed_dur = fixed_vel = quantized = 0
        for n in t.notes:
            # Onsets must sit on the rhythmic grid; durations only need to be valid.
            q = max(0.0, round(n.start * STEPS_PER_QUARTER) / STEPS_PER_QUARTER)
            if abs(q - n.start) > 1e-6:
                quantized += 1
            n.start = q
            if n.duration < MIN_DURATION:
                fixed_dur += 1
                n.duration = MIN_DURATION
            v = max(1, min(127, int(round(n.velocity))))
            if v != n.velocity:
                fixed_vel += 1
                n.velocity = v
        if fixed_dur:
            repairs.append(f"track {t.track_id}: {fixed_dur} too-short/zero/negative durations set to minimum")
        if fixed_vel:
            repairs.append(f"track {t.track_id}: {fixed_vel} velocities clamped to 1-127")
        if quantized:
            repairs.append(f"track {t.track_id}: {quantized} note(s) snapped to the 1/{STEPS_PER_QUARTER}-beat grid")
        # Duplicates (same onset + pitch) - keep the louder/longer one.
        best: dict[tuple, object] = {}
        for n in t.notes:
            k = (round(n.start, 6), n.pitch)
            if k not in best or (n.duration, n.velocity) > (best[k].duration, best[k].velocity):
                best[k] = n
        if len(best) < len(t.notes):
            repairs.append(f"track {t.track_id}: removed {len(t.notes) - len(best)} duplicate note(s)")
            t.notes = list(best.values())
        t.notes.sort(key=lambda n: (n.start, n.pitch))
        # Same-pitch overlaps re-trigger badly in MIDI: truncate the earlier note.
        last_by_pitch: dict[int, object] = {}
        trimmed = 0
        for n in t.notes:
            prev = last_by_pitch.get(n.pitch)
            if prev is not None and prev.start + prev.duration > n.start:
                prev.duration = max(MIN_DURATION, n.start - prev.start)
                trimmed += 1
            last_by_pitch[n.pitch] = n
        if trimmed:
            repairs.append(f"track {t.track_id}: trimmed {trimmed} overlapping same-pitch note(s)")
        if t.role in MONOPHONIC_TRACK_ROLES and not t.is_drum:
            n_fixed = _make_monophonic(t)
            if n_fixed:
                repairs.append(f"track {t.track_id}: resolved {n_fixed} overlap(s) in monophonic {t.role} part")
        if not t.is_drum:
            repairs.extend(fit_track_range(t))
    ch_warn = assign_channels(score)
    repairs.extend(ch_warn)
    return repairs


def _make_monophonic(t: Track) -> int:
    notes = sorted(t.notes, key=lambda n: (n.start, -n.pitch))
    out = []
    fixed = 0
    for n in notes:
        if out and abs(out[-1].start - n.start) < 1e-9:
            fixed += 1          # simultaneous onset: keep the higher note
            continue
        if out and out[-1].start + out[-1].duration > n.start + 1e-9:
            out[-1].duration = max(MIN_DURATION, n.start - out[-1].start)
            fixed += 1
        out.append(n)
    t.notes = out
    return fixed


# ----------------------------------------------------------------------------- validation


def validate_score(score: Score, *, mode: str, chords: list[ChordSpan] | None = None,
                   requested_roles: list[str] | None = None, anchor_reference: dict[int, list[tuple]] | None = None,
                   anchor_shift: int = 0, expected_meter: str | None = None,
                   role_of_track: dict[int, str] | None = None, root_pc_for_key: int | None = None) -> ValidationReport:
    """Validate a score. `root_pc_for_key` is the absolute tonic pitch class; key checks use it."""
    rep = ValidationReport()
    role_of_track = role_of_track or {}
    tracks_by_channel: dict[int, set[int]] = defaultdict(set)
    for t in score.tracks:
        tid = t.track_id
        # Invalid = shorter than one tick (it would vanish or produce a zero-length note on export).
        # Short but positive input notes (e.g. 30-tick grace notes) are legitimate.
        one_tick = 1.0 / max(1, score.ticks_per_beat)
        bad_dur = sum(1 for n in t.notes if n.duration < one_tick - 1e-9)
        if bad_dur:
            rep.add("invalid_duration", "error", f"{bad_dur} note(s) with invalid duration", tid, bad_dur)
        bad_pitch = sum(1 for n in t.notes if not 0 <= n.pitch <= 127)
        if bad_pitch:
            rep.add("pitch_out_of_midi_range", "error", f"{bad_pitch} pitch(es) outside 0-127", tid, bad_pitch)
        bad_vel = sum(1 for n in t.notes if not 1 <= n.velocity <= 127)
        if bad_vel:
            rep.add("invalid_velocity", "error", f"{bad_vel} velocity value(s) outside 1-127", tid, bad_vel)
        keys = [(round(n.start, 6), n.pitch) for n in t.notes]
        dups = len(keys) - len(set(keys))
        if dups:
            rep.add("duplicate_notes", "warning", f"{dups} duplicate note(s)", tid, dups)
        if not 0 <= t.program <= 127:
            rep.add("invalid_program", "error", f"program {t.program} invalid", tid)
        if not 0 <= t.channel <= 15:
            rep.add("invalid_channel", "error", f"channel {t.channel} invalid", tid)
        if t.is_drum and t.channel != DRUM_CHANNEL:
            rep.add("drum_channel", "error", f"drum track on channel {t.channel} (expected 9)", tid)
        if not t.is_drum and t.channel == DRUM_CHANNEL:
            rep.add("pitched_on_drum_channel", "error", "pitched track on percussion channel 9", tid)
        tracks_by_channel[t.channel].add(t.program if not t.is_drum else -1)
        role = t.role or role_of_track.get(tid)     # track role (lead/bass/...) drives mono/key checks
        if role in MONOPHONIC_TRACK_ROLES and not t.is_drum and t.source == "generated":
            ns = sorted(t.notes, key=lambda n: n.start)
            ov = sum(1 for a, b in zip(ns, ns[1:]) if a.end > b.start + 1e-6)
            if ov:
                rep.add("monophonic_overlap", "error", f"{ov} overlap(s) in monophonic {role} part", tid, ov)
        if not t.is_drum and t.notes:
            lo, hi = instrument_range(t.program)
            out = sum(1 for n in t.notes if not lo <= n.pitch <= hi)
            if out:
                sev = "warning" if t.source == "generated" else "info"
                rep.add("instrument_range", sev, f"{out} note(s) outside {t.instrument_name} range {lo}-{hi}", tid, out)
        off_grid = sum(1 for n in t.notes if abs(n.start * STEPS_PER_QUARTER - round(n.start * STEPS_PER_QUARTER)) > 1e-3)
        if off_grid and t.source == "generated":
            rep.add("off_grid", "warning", f"{off_grid} onset(s) off the rhythmic grid", tid, off_grid)
        # Key consistency for generated pitched parts.
        if root_pc_for_key is not None and t.source == "generated" and not t.is_drum and t.notes:
            frac = key_fraction(t, root_pc_for_key, mode, chords)
            thr = CHROMATIC_ROLES_THRESHOLD if role in ("bass", "counterline") else KEY_THRESHOLD
            if frac < thr:
                rep.add("off_key", "warning", f"only {frac:.0%} of note duration fits the key/chords", tid)
    for ch, progs in tracks_by_channel.items():
        if len(progs) > 1:
            rep.add("channel_conflict", "error", f"channel {ch} shared by different programs {sorted(progs)}")
    if expected_meter and score.initial_time_signature.label != expected_meter:
        rep.add("meter_mismatch", "error", f"meter {score.initial_time_signature.label} != {expected_meter}")
    if anchor_reference:
        for tid, sig in anchor_reference.items():
            try:
                t = score.track_by_id(tid)
            except KeyError:
                rep.add("anchor_missing", "error", f"anchor track {tid} missing from output", tid)
                continue
            if note_signature(t, 0) != [(s, d, p + anchor_shift if not t.is_drum else p, v) for s, d, p, v in sig]:
                rep.add("anchor_modified", "error", "locked/anchor track differs from the input", tid)
    for role in requested_roles or []:
        if not any(role_of_track.get(t.track_id) == role and t.notes for t in score.tracks):
            rep.add("missing_role", "error", f"requested role '{role}' produced no notes")
    # Export -> re-parse round trip and raw note-on/off pairing.
    try:
        data = score_to_bytes(score)
        rep.issues.extend(check_note_pairing(data))
        reparsed = load_midi(data)
        n_in = sum(len(t.notes) for t in score.tracks)
        n_out = sum(len(t.notes) for t in reparsed.tracks)
        if n_in != n_out:
            rep.add("reparse_mismatch", "error", f"re-parsed MIDI has {n_out} notes, expected {n_in}")
    except (MidiLoadError, Exception) as e:  # noqa: B014
        rep.add("reparse_failed", "error", f"exported MIDI could not be parsed: {e}")
    return rep


def key_fraction(t: Track, root_pc: int, mode: str, chords: list[ChordSpan] | None) -> float:
    """Share of note duration whose pitch class is in the scale or in the current chord."""
    sc = set(scale_pcs(mode))
    tot = good = 0.0
    for n in t.notes:
        rel = (n.pitch - root_pc) % 12
        ok = rel in sc
        if not ok and chords:
            c = chord_at(chords, n.start)
            ok = c is not None and rel in c.pcs
        tot += n.duration
        good += n.duration if ok else 0.0
    return good / tot if tot else 1.0


def check_note_pairing(data: bytes) -> list[Issue]:
    issues = []
    mf = mido.MidiFile(file=io.BytesIO(data))
    for ti, tr in enumerate(mf.tracks):
        depth: dict[tuple[int, int], int] = defaultdict(int)
        orphan_off = 0
        for msg in tr:
            if msg.type == "note_on" and msg.velocity > 0:
                depth[(msg.channel, msg.note)] += 1
            elif msg.type in ("note_off", "note_on"):
                k = (msg.channel, msg.note)
                if depth[k] <= 0:
                    orphan_off += 1
                else:
                    depth[k] -= 1
        hanging = sum(v for v in depth.values() if v > 0)
        if hanging:
            issues.append(Issue("unpaired_note_on", "error", f"{hanging} note-on(s) without note-off in MIDI track {ti}"))
        if orphan_off:
            issues.append(Issue("orphan_note_off", "error", f"{orphan_off} note-off(s) without note-on in MIDI track {ti}"))
    return issues
