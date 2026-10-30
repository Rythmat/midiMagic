"""Track roles: vocabulary, generated-role mapping, and heuristic role detection."""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .midi_io import Score, Track
from .theory import gm_family

# Roles a track can *have* (analysis, tokens).
TRACK_ROLES = ["lead", "bass", "harmony", "chordal", "high_texture", "low_texture",
               "counterline", "drums", "pad", "full_mix"]

# Roles the user can declare for an uploaded file ("highs"/"lows" are UI aliases).
INPUT_ROLES = ["lead", "bass", "highs", "lows", "chordal", "full_mix"]
INPUT_ROLE_ALIASES = {"highs": "high_texture", "lows": "low_texture", "full mix": "full_mix",
                      "melody": "lead", "chords": "chordal"}

# Roles the user can ask the system to *generate*.
GENERATED_ROLES = ["full_composition", "lead_melody", "chord_bed", "arpeggio", "bass_line", "walking_bass",
                   "octave_doubles", "counterline", "harmony", "drums", "pad", "high_texture", "low_texture"]

# Generated role -> track role used for tokens/analysis.
GEN_TO_TRACK_ROLE = {
    "lead_melody": "lead", "chord_bed": "chordal", "arpeggio": "harmony", "bass_line": "bass",
    "walking_bass": "bass", "octave_doubles": "harmony", "counterline": "counterline", "harmony": "harmony",
    "drums": "drums", "pad": "pad", "high_texture": "high_texture", "low_texture": "low_texture",
}
MONOPHONIC_GEN_ROLES = {"lead_melody", "bass_line", "walking_bass", "counterline"}
MONOPHONIC_TRACK_ROLES = {"lead", "bass", "counterline"}

# Default GM programs per generated role.
DEFAULT_PROGRAMS = {
    "lead_melody": 73, "chord_bed": 0, "arpeggio": 46, "bass_line": 33, "walking_bass": 32,
    "octave_doubles": 48, "counterline": 71, "harmony": 48, "drums": 0, "pad": 89,
    "high_texture": 11, "low_texture": 42,
}

# XMIDI track names are a small canonical set; they are strong role evidence.
NAME_HINTS = {
    "bass": "bass", "drum": "drums", "drums": "drums", "lead": "lead", "pad": "pad", "string": "pad",
    "strings": "pad", "piano": "chordal", "guitar": "chordal", "organ": "chordal", "harp": "harmony",
    "flute": "lead", "sax": "lead", "trumpet": "lead", "violin": "lead", "xylophone": "high_texture",
    "pipa": "lead", "guzheng": "harmony", "tuba": "bass",
}


SUSTAINING = {"strings", "ensemble", "organ", "synth_pad", "synth_effects", "brass", "reed", "pipe", "synth_lead"}


def normalize_input_role(role: str | None) -> str | None:
    if role is None:
        return None
    r = role.strip().lower().replace("-", "_")
    r = INPUT_ROLE_ALIASES.get(r, INPUT_ROLE_ALIASES.get(r.replace("_", " "), r))
    if r not in TRACK_ROLES:
        raise ValueError(f"Unknown input role {role!r}. Choose from {INPUT_ROLES}")
    return r


def expand_generated_roles(roles: list[str], anchor_roles: list[str] | None = None) -> list[str]:
    """Expand 'full_composition' into a concrete role set that complements the anchor."""
    out: list[str] = []
    anchor_roles = anchor_roles or []
    for r in roles:
        r = r.strip().lower().replace("-", "_").replace(" ", "_")
        if r not in GENERATED_ROLES:
            raise ValueError(f"Unknown generated role {r!r}. Choose from {GENERATED_ROLES}")
        if r == "full_composition":
            full = ["lead_melody", "chord_bed", "bass_line", "drums", "pad"]
            if "lead" in anchor_roles or "full_mix" in anchor_roles:
                full = [x for x in full if x != "lead_melody"] + ["counterline"]
            if "bass" in anchor_roles or "full_mix" in anchor_roles:
                full = [x for x in full if x != "bass_line"]
            if "chordal" in anchor_roles or "full_mix" in anchor_roles:
                full = [x for x in full if x not in ("chord_bed",)] + ["arpeggio"]
            if "drums" in anchor_roles:
                full = [x for x in full if x != "drums"]
            for x in full:
                if x not in out:
                    out.append(x)
        elif r not in out:
            out.append(r)
    return out


@dataclass
class TrackStats:
    n_notes: int
    min_pitch: int
    max_pitch: int
    median_pitch: float
    mean_polyphony: float
    mean_duration: float
    notes_per_bar: float
    mean_velocity: float

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def track_stats(track: Track, n_bars: int) -> TrackStats:
    notes = track.notes
    if not notes:
        return TrackStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    onsets: dict[float, int] = {}
    for n in notes:
        k = round(n.start * 12) / 12
        onsets[k] = onsets.get(k, 0) + 1
    return TrackStats(
        n_notes=len(notes), min_pitch=min(n.pitch for n in notes), max_pitch=max(n.pitch for n in notes),
        median_pitch=float(statistics.median(n.pitch for n in notes)),
        mean_polyphony=len(notes) / max(1, len(onsets)),
        mean_duration=statistics.fmean(n.duration for n in notes),
        notes_per_bar=len(notes) / max(1, n_bars),
        mean_velocity=statistics.fmean(n.velocity for n in notes))


def detect_roles(score: Score) -> dict[int, dict]:
    """Heuristic role detection per track. Returns {track_id: {role, confidence, reason, stats}}."""
    n_bars = score.bar_grid().n_bars
    stats = {t.track_id: track_stats(t, n_bars) for t in score.tracks}
    out: dict[int, dict] = {}
    pitched = [t for t in score.tracks if not t.is_drum and t.notes]
    mono_like = [t for t in pitched if stats[t.track_id].mean_polyphony < 1.35]
    top_mono = max(mono_like, key=lambda t: (stats[t.track_id].median_pitch, stats[t.track_id].n_notes), default=None)
    lowest = min(pitched, key=lambda t: stats[t.track_id].median_pitch, default=None)

    for t in score.tracks:
        s = stats[t.track_id]
        fam = gm_family(t.program)
        hint = NAME_HINTS.get(t.name.strip().lower())
        if t.is_drum:
            role, conf, why = "drums", 1.0, "percussion channel"
        elif not t.notes:
            role, conf, why = "lead", 0.0, "empty track"
        elif len(pitched) == 1 and s.mean_polyphony >= 1.6 and s.max_pitch - s.min_pitch >= 24:
            role, conf, why = "full_mix", 0.7, "single polyphonic track with wide range"
        elif fam == "bass" or (s.median_pitch < 48 and s.mean_polyphony < 1.3) or hint == "bass":
            role, conf, why = "bass", 0.9 if fam == "bass" else 0.75, "bass program/low monophonic register"
        elif fam == "synth_pad" or (s.mean_duration >= 2.0 and s.mean_polyphony >= 1.5 and fam in SUSTAINING):
            role, conf, why = "pad", 0.75, "long sustained polyphonic notes on a sustaining instrument"
        elif s.mean_polyphony >= 1.8:
            role, conf, why = "chordal", 0.8, f"polyphony {s.mean_polyphony:.2f}"
        elif t is top_mono and s.notes_per_bar >= 1.0:
            role, conf, why = "lead", 0.75, "highest monophonic line"
        elif s.median_pitch >= 74:
            role, conf, why = "high_texture", 0.6, "high register"
        elif t is lowest and s.median_pitch < 55:
            role, conf, why = "low_texture", 0.6, "lowest register"
        elif s.mean_polyphony >= 1.35:
            role, conf, why = "harmony", 0.55, "light polyphony"
        else:
            role, conf, why = "counterline", 0.5, "secondary monophonic line"
        # Use canonical track names as tie-breaker evidence when the heuristic is unsure.
        if hint and hint != role and conf < 0.7 and not t.is_drum:
            role, conf, why = hint, 0.65, f"track name '{t.name}'"
        out[t.track_id] = {"role": role, "confidence": conf, "reason": why, "stats": s.to_dict()}
    return out
