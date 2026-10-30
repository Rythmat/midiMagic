"""Composition planner: form, sections, chord progression, energy, registers, instrumentation.

Plans are expressed in root-relative space: chords are (semitones above the tonic, quality)
and registers are relative to `ref`, the MIDI reference the output will be denormalized with.
"""
from __future__ import annotations

import random
import re
import zlib
from dataclasses import dataclass, field

import numpy as np

from .harmony import ChordSpan
from .midi_io import BarGrid, TempoChange, TimeSignature
from .roles import DEFAULT_PROGRAMS, GEN_TO_TRACK_ROLE
from .styles import (GENRES, ROLE_PREFERENCES, EmotionProfile, GenreProfile, _MODAL, parse_instrumentation,
                     resolve_emotion, resolve_genre)
from .theory import MODE_FAMILY, PITCH_NAMES, instrument_range, roman_to_chord

SECTION_ENERGY = {"intro": 0.35, "verse": 0.55, "pre": 0.65, "chorus": 0.85, "hook": 0.85, "bridge": 0.6,
                  "solo": 0.75, "outro": 0.4, "build": 0.65, "drop": 0.95, "break": 0.4, "head": 0.7,
                  "A": 0.6, "B": 0.7, "C": 0.65, "D": 0.65}

# Absolute MIDI registers (low, high) per generated role, before instrument-range intersection.
ROLE_REGISTERS = {
    "lead_melody": (62, 84), "chord_bed": (50, 74), "arpeggio": (55, 84), "bass_line": (31, 55),
    "walking_bass": (31, 57), "octave_doubles": (36, 96), "counterline": (50, 72), "harmony": (55, 79),
    "pad": (48, 79), "high_texture": (76, 100), "low_texture": (33, 55),
}


def stable_seed(label: str, seed: int) -> int:
    """Process-independent seed (Python's hash() is salted per process)."""
    return zlib.crc32(f"{label}|{seed}".encode()) & 0xFFFF


@dataclass
class SectionPlan:
    index: int
    name: str
    letter: str
    start_bar: int
    n_bars: int
    energy: float
    progression: list[str] = field(default_factory=list)
    active_roles: list[str] = field(default_factory=list)
    motif_seed: int = 0

    @property
    def end_bar(self) -> int:
        return self.start_bar + self.n_bars

    def to_dict(self) -> dict:
        return dict(self.__dict__, end_bar=self.end_bar)


@dataclass
class CompositionPlan:
    genre: str
    emotion: str
    genre_profile: str
    root_pc: int
    mode: str
    ref: int                                  # MIDI reference used for denormalization
    tempo: float
    tempos: list[TempoChange]
    time_signatures: list[TimeSignature]
    n_bars: int
    sections: list[SectionPlan]
    chords: list[ChordSpan]
    roles: list[str]                          # generated roles
    programs: dict[str, int]
    registers: dict[str, tuple[int, int]]     # relative to ref
    density: float
    swing: bool
    drum_style: str
    bass_style: str
    comp_style: str
    valence: float
    arousal: float
    energy_curve: list[float]
    seed: int
    anchor_onset_profile: list[float] | None = None
    anchor_mode: bool = False
    form_name: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def root_name(self) -> str:
        return PITCH_NAMES[self.root_pc]

    @property
    def meter(self) -> TimeSignature:
        return self.time_signatures[0]

    def grid(self) -> BarGrid:
        bar_beats = self.time_signatures[0].bar_beats
        return BarGrid(self.time_signatures, self.n_bars * bar_beats, min_bars=self.n_bars)

    def section_of_bar(self, bar: int) -> SectionPlan:
        for s in self.sections:
            if s.start_bar <= bar < s.end_bar:
                return s
        return self.sections[-1]

    def to_dict(self) -> dict:
        return {
            "genre": self.genre, "emotion": self.emotion, "genre_profile": self.genre_profile,
            "root_name": self.root_name, "root_pitch_class": self.root_pc, "mode": self.mode,
            "output_reference_midi": self.ref, "tempo": self.tempo, "meter": self.meter.label,
            "n_bars": self.n_bars, "form": self.form_name, "sections": [s.to_dict() for s in self.sections],
            "chords": [c.to_dict(self.root_pc) for c in self.chords], "roles": self.roles,
            "programs": self.programs, "registers_relative": self.registers, "density": self.density,
            "swing": self.swing, "drum_style": self.drum_style, "bass_style": self.bass_style,
            "comp_style": self.comp_style, "valence": self.valence, "arousal": self.arousal,
            "energy_curve": self.energy_curve, "anchor_mode": self.anchor_mode, "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------- form

def parse_form(form: str | None, profile: GenreProfile, rng: random.Random) -> tuple[list[tuple[str, int]], str]:
    if not form or form.strip().lower() in ("auto", "default", ""):
        tpl = rng.choice(profile.forms)
        return list(tpl), f"{profile.name} default"
    f = form.strip().lower().replace("_", "-")
    named = {
        "verse-chorus": [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("outro", 4)],
        "pop": GENRES["pop"].forms[0], "aaba": [("A", 8), ("A", 8), ("B", 8), ("A", 8)],
        "aba": [("A", 8), ("B", 8), ("A", 8)], "ternary": [("A", 8), ("B", 8), ("A", 8)],
        "binary": [("A", 8), ("A", 8), ("B", 8), ("B", 8)], "aabb": [("A", 8), ("A", 8), ("B", 8), ("B", 8)],
        "rondo": [("A", 8), ("B", 8), ("A", 8), ("C", 8), ("A", 8)],
        "strophic": [("A", 8), ("A", 8), ("A", 8), ("A", 8)],
        "through-composed": [("A", 8), ("B", 8), ("C", 8), ("D", 8)],
        "12-bar-blues": [("chorus", 12), ("chorus", 12)], "blues": [("chorus", 12), ("chorus", 12)],
        "head-solo-head": [("head", 12), ("solo", 12), ("head", 12)],
        "build-drop": GENRES["electronic"].forms[0],
    }
    if f in named:
        return list(named[f]), f
    if re.fullmatch(r"[a-z]+", f) and len(f) <= 12 and f.upper() == form.strip().upper() and len(set(f)) <= 6 \
            and all(len(x) == 1 for x in f):
        return [(c.upper(), 8) for c in f], f.upper()
    parts = [p for p in re.split(r"[,\s>|-]+", f) if p]
    if parts:
        out = []
        for p in parts:
            m = re.fullmatch(r"([a-z]+)(?::?(\d+))?", p)
            if m:
                name = m.group(1)
                n = int(m.group(2)) if m.group(2) else (4 if name in ("intro", "outro") else 8)
                out.append((name.upper() if len(name) == 1 else name, n))
        if out:
            return out, "custom"
    return list(profile.forms[0]), f"{profile.name} default"


def fit_form_to_length(form: list[tuple[str, int]], n_bars: int) -> list[tuple[str, int]]:
    """Scale/trim/extend a section template to exactly n_bars, keeping phrase-sized sections."""
    total = sum(n for _, n in form)
    if n_bars <= 0:
        return form
    if n_bars <= 8:
        core = [s for s in form if s[0] not in ("intro", "outro")] or form
        return [(core[0][0], n_bars)]
    if total == n_bars:
        return form
    if total > n_bars:
        out, acc = [], 0
        # Keep intro/outro only if there is room for at least two core sections.
        keep_frame = n_bars >= 24
        core = [s for s in form if keep_frame or s[0] not in ("intro", "outro")]
        for name, n in core:
            if acc >= n_bars:
                break
            if name == "outro" and keep_frame:
                continue
            take = min(n, n_bars - acc)
            out.append((name, take))
            acc += take
        if keep_frame and any(s[0] == "outro" for s in form):
            o = next(n for s, n in form if s == "outro")
            # Make space for the outro by trimming the last core section.
            while out and sum(n for _, n in out) + o > n_bars:
                nm, n = out[-1]
                cut = min(n, sum(x for _, x in out) + o - n_bars)
                out[-1] = (nm, n - cut)
                if out[-1][1] <= 0:
                    out.pop()
            out.append(("outro", n_bars - sum(n for _, n in out)))
        out = [(nm, n) for nm, n in out if n > 0]
        return _merge_tiny(out)
    # Extend: repeat the core (non intro/outro) sections before the outro.
    core = [s for s in form if s[0] not in ("intro", "outro")] or form
    head = [s for s in form if s[0] != "outro"]
    tail = [s for s in form if s[0] == "outro"]
    out = list(head)
    i = 0
    while sum(n for _, n in out) + sum(n for _, n in tail) < n_bars:
        out.append(core[i % len(core)])
        i += 1
    out += tail
    excess = sum(n for _, n in out) - n_bars
    j = len(out) - 1 - len(tail)
    while excess > 0 and j >= 0:
        nm, n = out[j]
        cut = min(excess, n - 2)
        if cut > 0:
            out[j] = (nm, n - cut)
            excess -= cut
        j -= 1
    return _merge_tiny([(nm, n) for nm, n in out if n > 0])


def _merge_tiny(sections: list[tuple[str, int]]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for nm, n in sections:
        if n < 2 and out:
            out[-1] = (out[-1][0], out[-1][1] + n)
        else:
            out.append((nm, n))
    return out


# ----------------------------------------------------------------------------- harmony

def choose_progression(profile: GenreProfile, mode: str, rng: random.Random) -> list[str]:
    if mode in _MODAL:
        return list(rng.choice(_MODAL[mode]))
    fam = MODE_FAMILY.get(mode, "major")
    return list(rng.choice(profile.progressions.get(fam) or profile.progressions["major"]))


def progression_to_chords(progression: list[str], mode: str, start_bar: int, n_bars: int, grid: BarGrid,
                          final_cadence: bool, half_cadence: bool) -> list[ChordSpan]:
    chords: list[ChordSpan] = []
    tonic_q = "maj" if MODE_FAMILY.get(mode, "major") == "major" else "min"
    for i in range(n_bars):
        bar = start_bar + i
        numeral = progression[i % len(progression)]
        root, q = roman_to_chord(numeral, mode)
        if final_cadence and i == n_bars - 1:
            root, q = 0, tonic_q if not q.endswith("7") else ("maj7" if tonic_q == "maj" else "min7")
            if i >= 1:
                pr, pq = roman_to_chord(progression[(i - 1) % len(progression)], mode)
                if (pr, pq) == (0, tonic_q) and chords:
                    chords[-1] = ChordSpan(chords[-1].start, chords[-1].end, 7, "7" if "7" in q else "maj", chords[-1].bar)
        elif half_cadence and i == n_bars - 1 and n_bars >= 4:
            root, q = 7, "7" if q.endswith("7") else "maj"
        s = grid.bar_start(bar)
        chords.append(ChordSpan(s, s + grid.bar_length(bar), root, q, bar))
    return chords


# ----------------------------------------------------------------------------- registers / programs

def assign_programs(roles: list[str], profile: GenreProfile, instrumentation: str | None,
                    warnings: list[str]) -> dict[str, int]:
    preset, requested, w = parse_instrumentation(instrumentation)
    warnings.extend(w)
    programs: dict[str, int] = {}
    pool = list(requested)
    for role in roles:
        if role == "drums":
            programs[role] = 0
            continue
        if role in preset:
            programs[role] = preset[role]
            continue
        if pool:
            prefs = ROLE_PREFERENCES.get(role, [])
            pick = next((p for p in prefs if p in pool), None)
            if pick is not None:
                programs[role] = pick
                if len(pool) > 1:
                    pool.remove(pick)
                continue
        programs[role] = profile.instrumentation.get(role, DEFAULT_PROGRAMS.get(role, 0))
    return programs


def relative_registers(roles: list[str], programs: dict[str, int], ref: int,
                       avoid: tuple[float, float] | None = None, anchor_role: str | None = None) -> dict[str, tuple[int, int]]:
    """Per-role pitch windows relative to `ref`, intersected with instrument ranges.

    `avoid` is the anchor's (p10, p90) absolute range; textural roles are steered around it.
    """
    regs = {}
    for role in roles:
        if role == "drums":
            continue
        lo, hi = ROLE_REGISTERS.get(role, (48, 84))
        ilo, ihi = instrument_range(programs.get(role, 0))
        if avoid is not None and role in ("counterline", "chord_bed", "harmony", "pad") and anchor_role in ("lead", "high_texture"):
            # keep accompaniment below an anchor melody
            hi = min(hi, int(avoid[0]) + 2)
            lo = min(lo, hi - 14)
        if avoid is not None and role == "lead_melody" and anchor_role in ("chordal", "pad", "full_mix"):
            lo = max(lo, int(avoid[1]) - 4)
            hi = max(hi, lo + 16)
        lo2, hi2 = max(lo, ilo), min(hi, ihi)
        if hi2 - lo2 < 12:            # instrument cannot cover the window: use the instrument's middle
            mid = (ilo + ihi) // 2
            lo2, hi2 = max(ilo, mid - 9), min(ihi, mid + 9)
        regs[role] = (lo2 - ref, hi2 - ref)
    return regs


# ----------------------------------------------------------------------------- plan builders

def _density_value(density) -> float:
    if density is None:
        return 0.5
    if isinstance(density, str):
        return {"sparse": 0.2, "low": 0.3, "medium": 0.5, "high": 0.75, "dense": 0.9}.get(density.lower(), 0.5)
    d = float(density)
    return d / 5.0 if d > 1.0 else max(0.0, min(1.0, d))


def plan_from_metadata(*, genre: str | None, emotion: str | None, root_pc: int, mode: str | None,
                       tempo: float | None, meter: str = "4/4", n_bars: int | None = None, density=0.5,
                       instrumentation: str | None = None, form: str | None = None, roles: list[str],
                       seed: int = 0, octave_offset: int = 0) -> CompositionPlan:
    rng = random.Random(seed)
    warnings: list[str] = []
    profile, w = resolve_genre(genre)
    if w:
        warnings.append(w)
    emo, w = resolve_emotion(emotion)
    if w:
        warnings.append(w)
    if mode is None:
        mode = emo.preferred_mode or ("minor" if emo.valence < -0.2 else "major")
    if tempo is None:
        lo, hi = profile.tempo_range
        base = (lo + hi) / 2 + (hi - lo) / 2 * emo.arousal * 0.8
        tempo = float(round(base))
    num, den = _parse_meter(meter)
    sig = TimeSignature(0.0, num, den)
    form_tpl, form_name = parse_form(form, profile, rng)
    if n_bars is None:
        n_bars = sum(n for _, n in form_tpl)
    form_secs = fit_form_to_length(form_tpl, n_bars)
    dens = _density_value(density)
    dens = float(np.clip(dens * (1 + 0.25 * emo.arousal), 0.05, 1.0))
    ref = 60 + root_pc + 12 * octave_offset
    grid = BarGrid([sig], n_bars * sig.bar_beats, min_bars=n_bars)

    sections: list[SectionPlan] = []
    chords: list[ChordSpan] = []
    prog_by_name: dict[str, list[str]] = {}
    bar = 0
    for i, (name, nb) in enumerate(form_secs):
        base_name = name.rstrip("'")
        if base_name not in prog_by_name:
            prog_by_name[base_name] = choose_progression(profile, mode, rng)
        prog = prog_by_name[base_name]
        energy = SECTION_ENERGY.get(base_name, SECTION_ENERGY.get(base_name[:1].upper(), 0.6))
        energy = float(np.clip(energy + 0.2 * emo.arousal, 0.1, 1.0))
        is_last = i == len(form_secs) - 1
        next_is_chorus = i + 1 < len(form_secs) and form_secs[i + 1][0] in ("chorus", "hook", "drop")
        sec_chords = progression_to_chords(prog, mode, bar, nb, grid, final_cadence=is_last,
                                           half_cadence=next_is_chorus and len(prog) != 12)
        letter = base_name if len(base_name) == 1 else base_name[:1].upper()
        sections.append(SectionPlan(i, name, letter, bar, nb, round(energy, 3), prog,
                                    motif_seed=stable_seed(base_name, seed)))
        chords += sec_chords
        bar += nb

    roles = list(roles)
    programs = assign_programs(roles, profile, instrumentation, warnings)
    registers = relative_registers(roles, programs, ref)
    for s in sections:
        s.active_roles = _active_roles(s, roles)
    # A requested role must sound somewhere: if the form silenced it everywhere, enable it throughout.
    for r in roles:
        if not any(r in s.active_roles for s in sections):
            for s in sections:
                s.active_roles.append(r)
    energy_curve = [round(next(s.energy for s in sections if s.start_bar <= b < s.end_bar), 3) for b in range(n_bars)]
    return CompositionPlan(
        genre=genre or profile.name, emotion=emotion or "neutral", genre_profile=profile.name, root_pc=root_pc,
        mode=mode, ref=ref, tempo=float(tempo), tempos=[TempoChange(0.0, float(tempo))], time_signatures=[sig],
        n_bars=n_bars, sections=sections, chords=chords, roles=roles, programs=programs, registers=registers,
        density=dens, swing=profile.swing, drum_style=_drum_style(profile, emo, sig),
        bass_style=_bass_style(profile, emo, roles), comp_style=_comp_style(profile, dens, sig),
        valence=emo.valence, arousal=emo.arousal, energy_curve=energy_curve, seed=seed,
        form_name=form_name, warnings=warnings)


def _active_roles(section: SectionPlan, roles: list[str]) -> list[str]:
    name = section.name.lower()
    if name == "intro":
        keep = {"chord_bed", "pad", "arpeggio", "bass_line", "walking_bass", "high_texture", "low_texture", "drums"}
        return [r for r in roles if r in keep] or roles
    if name == "outro":
        return [r for r in roles if r not in ("counterline", "octave_doubles", "harmony")] or roles
    if name in ("verse", "break"):
        return [r for r in roles if r not in ("octave_doubles", "high_texture")] or roles
    return list(roles)


def _drum_style(profile: GenreProfile, emo: EmotionProfile, sig: TimeSignature) -> str:
    style = profile.drum_style
    if sig.numerator in (3, 6, 9, 12) and style not in ("none",):
        return "triple"
    if style in ("rock", "pop") and emo.arousal < -0.3:
        return "half_time"
    return style


def _bass_style(profile: GenreProfile, emo: EmotionProfile, roles: list[str]) -> str:
    if "walking_bass" in roles:
        return "walking"
    style = profile.bass_style
    if style == "walking":
        return "root_fifth" if emo.arousal < -0.5 else "walking"
    if emo.arousal < -0.4 and style in ("pedal8", "pop"):
        return "sustained"
    return style


def _comp_style(profile: GenreProfile, density: float, sig: TimeSignature) -> str:
    if sig.numerator in (3, 6, 9, 12):
        return "waltz" if sig.numerator == 3 else "arpeggio"
    if density < 0.25:
        return "sustain"
    return profile.comp_style


def _parse_meter(meter: str | None) -> tuple[int, int]:
    if not meter:
        return 4, 4
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(meter))
    if not m:
        raise ValueError(f"Bad meter {meter!r}; expected e.g. '4/4' or '6/8'")
    num, den = int(m.group(1)), int(m.group(2))
    if num < 1 or num > 16 or den not in (1, 2, 4, 8, 16):
        raise ValueError(f"Unsupported meter {meter!r}")
    return num, den


def plan_from_anchor(*, analysis: dict, anchor_chords: list[ChordSpan], tempos: list[TempoChange],
                     time_signatures: list[TimeSignature], grid: BarGrid, root_pc: int, mode: str,
                     genre: str | None, emotion: str | None, n_bars: int | None, density, instrumentation: str | None,
                     roles: list[str], seed: int, anchor_range: tuple[float, float] | None,
                     anchor_role: str | None, octave_offset: int = 0) -> CompositionPlan:
    """Plan around an uploaded MIDI: keep its meter/tempo map, chords and sections."""
    rng = random.Random(seed)
    warnings: list[str] = []
    profile, w = resolve_genre(genre)
    if w and genre:
        warnings.append(w)
    emo, w = resolve_emotion(emotion)
    if w:
        warnings.append(w)
    anchor_bars = analysis["n_bars"]
    n_bars = n_bars or anchor_bars
    sig0 = time_signatures[0]
    chords = [c for c in anchor_chords if c.bar < n_bars]
    if n_bars > anchor_bars:
        # Continue harmony past the anchor by cycling its progression (bar-level).
        per_bar = {}
        for c in anchor_chords:
            per_bar.setdefault(c.bar, (c.root_rel, c.quality))
        cyc = [per_bar.get(b, (0, "maj")) for b in range(anchor_bars)] or [(0, "maj")]
        ext_grid = BarGrid(time_signatures, (n_bars + 1) * sig0.bar_beats, min_bars=n_bars)
        for b in range(anchor_bars, n_bars):
            r, q = cyc[(b - anchor_bars) % len(cyc)]
            s = ext_grid.bar_start(b)
            chords.append(ChordSpan(s, s + ext_grid.bar_length(b), r, q, b))
        warnings.append(f"Requested {n_bars} bars > anchor length {anchor_bars}; harmony continues after the anchor ends.")
    sections = []
    for i, s in enumerate(analysis.get("sections") or [{"label": "A", "start_bar": 0, "end_bar": anchor_bars}]):
        if s["start_bar"] >= n_bars:
            break
        end = min(s["end_bar"], n_bars)
        sections.append(SectionPlan(i, s["label"], s["label"][:1], s["start_bar"], end - s["start_bar"],
                                    float(np.clip(0.45 + 0.5 * s.get("energy", 0.5) + 0.15 * emo.arousal, 0.1, 1.0)),
                                    motif_seed=stable_seed(s["label"].rstrip("'"), seed)))
    if sections and sections[-1].end_bar < n_bars:
        last = sections[-1]
        sections.append(SectionPlan(len(sections), "ext", "X", last.end_bar, n_bars - last.end_bar, last.energy,
                                    motif_seed=rng.randint(0, 0xFFFF)))
    for s in sections:
        s.active_roles = list(roles)
    ref = 60 + root_pc + 12 * octave_offset
    programs = assign_programs(roles, profile, instrumentation, warnings)
    registers = relative_registers(roles, programs, ref, avoid=anchor_range, anchor_role=anchor_role)
    dens = float(np.clip(_density_value(density) * (1 + 0.2 * emo.arousal), 0.05, 1.0))
    tempo = float(tempos[0].bpm) if tempos else 120.0
    energy_curve = [round(next((s.energy for s in sections if s.start_bar <= b < s.end_bar), 0.6), 3) for b in range(n_bars)]
    swing = bool(analysis.get("rhythm", {}).get("swing_ratio", 0) > 0.5) or (profile.swing and genre is not None)
    return CompositionPlan(
        genre=genre or "unspecified", emotion=emotion or "unspecified", genre_profile=profile.name, root_pc=root_pc,
        mode=mode, ref=ref, tempo=tempo, tempos=list(tempos), time_signatures=list(time_signatures), n_bars=n_bars,
        sections=sections, chords=chords, roles=list(roles), programs=programs, registers=registers, density=dens,
        swing=swing, drum_style=_drum_style(profile, emo, sig0), bass_style=_bass_style(profile, emo, roles),
        comp_style=_comp_style(profile, dens, sig0), valence=emo.valence, arousal=emo.arousal,
        energy_curve=energy_curve, seed=seed,
        anchor_onset_profile=analysis.get("rhythm", {}).get("onset_profile_16th"), anchor_mode=True,
        form_name="from anchor sections", warnings=warnings)


def track_role_for(gen_role: str) -> str:
    return GEN_TO_TRACK_ROLE.get(gen_role, gen_role)
