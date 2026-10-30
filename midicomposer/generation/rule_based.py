"""Deterministic rule-based track generator (the always-available fallback engine).

All pitches are root-relative (0 = tonic in octave 4 of the output reference); drums use GM
note numbers. Given the same plan, role and seed, output is identical.
"""
from __future__ import annotations

import bisect
import random
from dataclasses import dataclass

import numpy as np

from ..harmony import ChordSpan, chord_at, consonance
from ..midi_io import BarGrid
from ..planner import CompositionPlan
from ..theory import DRUM, scale_pcs


@dataclass
class RelNote:
    start: float
    duration: float
    pitch: int          # relative pitch, or GM drum note for drums
    velocity: int

    @property
    def end(self) -> float:
        return self.start + self.duration


class Line:
    """Fast 'what is sounding at time t' lookup over a monophonic-ish reference line."""

    def __init__(self, notes: list[RelNote]):
        self.notes = sorted(notes, key=lambda n: (n.start, -n.pitch))
        self.starts = [n.start for n in self.notes]

    def at(self, t: float) -> RelNote | None:
        """Note sounding at t, or one that ended within 0.75 beats (it still governs the harmony)."""
        i = bisect.bisect_right(self.starts, t + 1e-9) - 1
        if i < 0:
            return None
        n = self.notes[i]
        return n if n.end > t + 1e-9 or t - n.end <= 0.75 else None

    def onsets_in(self, a: float, b: float) -> int:
        return bisect.bisect_left(self.starts, b - 1e-9) - bisect.bisect_left(self.starts, a - 1e-9)

    def __bool__(self) -> bool:
        return bool(self.notes)


def skyline(notes: list[RelNote]) -> list[RelNote]:
    """Top voice of polyphonic material (highest pitch per onset, truncated at the next onset)."""
    by_onset: dict[float, RelNote] = {}
    for n in notes:
        k = round(n.start, 4)
        if k not in by_onset or n.pitch > by_onset[k].pitch:
            by_onset[k] = n
    line = [by_onset[k] for k in sorted(by_onset)]
    out = []
    for i, n in enumerate(line):
        end = min(n.end, line[i + 1].start) if i + 1 < len(line) else n.end
        if end > n.start:
            out.append(RelNote(n.start, end - n.start, n.pitch, n.velocity))
    return out


@dataclass
class GenContext:
    plan: CompositionPlan
    grid: BarGrid
    rng: random.Random
    anchor_notes: list[RelNote]            # all pitched anchor notes (relative)
    anchor_line: list[RelNote]             # melody line of the anchor (skyline / lead track)
    anchor_role: str | None
    generated: dict[str, list[RelNote]]
    choices: dict = None                    # per-candidate stylistic choices (pattern variants)
    anchor_tracks: list = None              # [(role, program, is_drum, notes)] per anchor track (neural engine)

    def pick(self, key: str, options: list):
        """Choose a pattern variant once per candidate so it stays consistent across bars."""
        if self.choices is None:
            self.choices = {}
        if key not in self.choices:
            self.choices[key] = self.rng.randrange(len(options))
        return options[self.choices[key] % len(options)]

    @property
    def mode_scale(self) -> tuple[int, ...]:
        return scale_pcs(self.plan.mode)

    def chord(self, t: float) -> ChordSpan:
        c = chord_at(self.plan.chords, t)
        return c if c is not None else ChordSpan(0, 4, 0, "maj", 0)

    def energy(self, bar: int) -> float:
        ec = self.plan.energy_curve
        return ec[min(bar, len(ec) - 1)] if ec else 0.6

    def vel(self, bar: int, accent: float = 0.0, base: float = 0.0) -> int:
        e = self.energy(bar)
        v = 48 + 55 * e + 10 * self.plan.arousal + accent + base + self.rng.gauss(0, 3.0)
        return int(np.clip(v, 18, 124))

    def register(self, role: str) -> tuple[int, int]:
        return self.plan.registers.get(role, (-12, 12))

    def active(self, role: str, bar: int) -> bool:
        return role in self.plan.section_of_bar(bar).active_roles

    def swing(self, t: float) -> float:
        if not self.plan.swing:
            return t
        frac = t - np.floor(t)
        if abs(frac - 0.5) < 1e-6:
            return np.floor(t) + 2.0 / 3.0
        return t

    def reference_line(self) -> Line:
        if self.anchor_line:
            return Line(self.anchor_line)
        if self.generated.get("lead_melody"):
            return Line(self.generated["lead_melody"])
        return Line([])

    def bars(self):
        for b in range(self.plan.n_bars):
            yield b, self.grid.bar_start(b), self.grid.bar_length(b)


# ----------------------------------------------------------------------------- helpers


def tones_in_range(pcs, lo: int, hi: int) -> list[int]:
    s = {p % 12 for p in pcs}
    return [p for p in range(lo, hi + 1) if p % 12 in s]


def nearest(cands: list[int], target: float) -> int:
    return min(cands, key=lambda c: (abs(c - target), c))


def voicing_candidates(pcs: list[int], lo: int, hi: int, n_voices: int) -> list[list[int]]:
    """Close-position voicings (all inversions, all octaves) inside [lo, hi]."""
    pcs = list(dict.fromkeys(p % 12 for p in pcs))[:n_voices]
    out = []
    for base in range(lo, hi - 4):
        if base % 12 not in pcs:
            continue
        v = [base]
        idx = pcs.index(base % 12)
        for k in range(1, len(pcs)):
            pc = pcs[(idx + k) % len(pcs)]
            nxt = v[-1] + ((pc - v[-1]) % 12 or 12)
            v.append(nxt)
        if v[-1] <= hi:
            out.append(v)
    return out


def voice_lead(prev: list[int] | None, cands: list[list[int]], center: float) -> list[int]:
    if not cands:
        return []
    if prev is None:
        return min(cands, key=lambda v: abs(np.mean(v) - center))
    return min(cands, key=lambda v: (sum(abs(a - b) for a, b in zip(sorted(v), sorted(prev)))
                                     + 0.15 * abs(np.mean(v) - center)))


def anchor_hit_positions(ctx: GenContext, bar_len: float, min_weight: float = 0.08) -> list[float]:
    """Strong onset positions (in beats) of the anchor's rhythm, quantized to 8ths."""
    prof = ctx.plan.anchor_onset_profile
    if not prof:
        return []
    prof = np.asarray(prof)
    n = len(prof)
    pos = []
    for i, w in enumerate(prof):
        if w >= min_weight and (i % 2 == 0):
            pos.append(i * bar_len / n)
    return pos


# ----------------------------------------------------------------------------- lead melody

_CELLS = {
    "low": [[2.0], [1.0, 1.0], [1.5, 0.5], [2.0]],
    "mid": [[1.0, 1.0], [1.5, 0.5], [0.5, 0.5, 1.0], [1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [1.0, 1.0]],
    "high": [[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.5, 1.0], [0.5, 0.25, 0.25, 1.0], [0.75, 0.25, 1.0],
             [0.5, 0.5, 1.0], [0.5, 1.0, 0.5]],
}


def _motif_rhythm(rng: random.Random, bar_len: float, density: float) -> list[tuple[float, float]]:
    level = "low" if density < 0.33 else ("mid" if density < 0.66 else "high")
    out, t = [], 0.0
    while t < bar_len - 1e-6:
        span = min(2.0, bar_len - t)
        if span < 2.0:
            cell = [span]
        else:
            cell = rng.choice(_CELLS[level])
        for d in cell:
            out.append((t, d))
            t += d
    return out


def gen_lead_melody(ctx: GenContext) -> list[RelNote]:
    plan, rng = ctx.plan, ctx.rng
    lo, hi = ctx.register("lead_melody")
    scale = list(ctx.mode_scale)
    allowed_weak = set(scale)
    notes: list[RelNote] = []
    center = (lo + hi) / 2
    cur = nearest(tones_in_range([0, 4 if 4 in scale else 3, 7], lo, hi), center)
    motifs: dict[str, dict] = {}
    for sec in plan.sections:
        if "lead_melody" not in sec.active_roles:
            continue
        # Motifs depend on the section label (repeats share a motif) and the candidate's seed.
        mrng = random.Random(sec.motif_seed * 7919 + plan.seed)
        if sec.letter not in motifs:
            bl = ctx.grid.bar_length(sec.start_bar)
            motifs[sec.letter] = {
                "rhythm": [_motif_rhythm(mrng, bl, plan.density) for _ in range(2)],
                "contour": [mrng.choice([-2, -1, -1, 1, 1, 2, 0, 3, -3]) for _ in range(32)],
            }
        motif = motifs[sec.letter]
        for i in range(sec.n_bars):
            bar = sec.start_bar + i
            bs, bl = ctx.grid.bar_start(bar), ctx.grid.bar_length(bar)
            phrase_pos = i % 4
            last_of_phrase = phrase_pos == 3 or i == sec.n_bars - 1
            section_end = i == sec.n_bars - 1
            rhythm = motif["rhythm"][phrase_pos % 2] if phrase_pos < 3 else motif["rhythm"][0][:1]
            if last_of_phrase:
                # Cadence bar: a pickup figure, then a long arrival note.
                rhythm = [(0.0, min(bl, 1.0))] + ([(1.0, bl - 1.0)] if bl > 1.0 else [])
                if rng.random() < 0.5 and bl >= 4:
                    rhythm = [(0.0, 1.0), (1.0, 1.0), (2.0, bl - 2.0)]
            if bl != ctx.grid.bar_length(sec.start_bar):
                rhythm = _motif_rhythm(mrng, bl, plan.density)
            for k, (off, dur) in enumerate(rhythm):
                t = bs + off
                if off >= bl - 1e-6:
                    continue
                dur = min(dur, bl - off)
                # Breathing: occasionally rest at the end of a phrase instead of playing.
                if last_of_phrase and not section_end and k == len(rhythm) - 1 and dur >= 2 and rng.random() < 0.3:
                    dur = dur / 2
                ch = ctx.chord(t)
                strong = (abs(off % 2.0) < 1e-6) or dur >= 1.5
                step = motif["contour"][(i * 4 + k) % len(motif["contour"])]
                if rng.random() < 0.2:
                    step += rng.choice([-1, 1])
                # Steer back toward the centre when far away.
                if cur > hi - 3:
                    step = -abs(step) or -1
                elif cur < lo + 3:
                    step = abs(step) or 1
                target = _scale_step(cur, step, scale)
                if last_of_phrase and k == len(rhythm) - 1:
                    pcs = [0] if (section_end or (i // 4) % 2 == 1) else [p for p in ch.pcs if p != ch.root_rel] or ch.pcs
                    cands = tones_in_range(pcs, lo, hi)
                elif strong:
                    cands = tones_in_range(ch.pcs, lo, hi)
                else:
                    cands = [p for p in range(lo, hi + 1) if p % 12 in allowed_weak | set(ch.pcs)]
                if not cands:
                    cands = list(range(lo, hi + 1))
                pitch = nearest(cands, target)
                ts = ctx.swing(t)
                notes.append(RelNote(ts, max(0.2, dur - (ts - t) - 0.02), pitch,
                                     ctx.vel(bar, accent=6 if strong else -4, base=6)))
                cur = pitch
    return _make_monophonic(notes)


def _scale_step(pitch: int, steps: int, scale: list[int]) -> int:
    if steps == 0:
        return pitch
    p = pitch
    direction = 1 if steps > 0 else -1
    count = 0
    while count < abs(steps):
        p += direction
        if p % 12 in scale:
            count += 1
    return p


def _make_monophonic(notes: list[RelNote]) -> list[RelNote]:
    notes = sorted(notes, key=lambda n: n.start)
    out: list[RelNote] = []
    for n in notes:
        if out and out[-1].end > n.start:
            if n.start - out[-1].start < 0.05:
                continue
            out[-1].duration = max(0.05, n.start - out[-1].start)
        out.append(n)
    return out


# ----------------------------------------------------------------------------- chords / pads / arps


def _comp_offsets(ctx: GenContext, bar_len: float, bar: int) -> list[tuple[float, float]]:
    style, d = ctx.plan.comp_style, ctx.plan.density
    beats = int(round(bar_len))
    # Around an anchor, some candidates follow its rhythm and some use the style pattern;
    # ranking (rhythmic compatibility) decides which works better.
    if ctx.plan.anchor_mode and d >= 0.35 and ctx.pick("comp_follow_anchor", [True, True, False]):
        hits = anchor_hit_positions(ctx, bar_len)
        if len(hits) >= 2:
            hits = sorted(set(hits))
            return [(h, (hits[i + 1] if i + 1 < len(hits) else bar_len) - h) for i, h in enumerate(hits)]
    if style == "sustain" or d < 0.25:
        return [(0.0, bar_len)]
    if style == "charleston" and bar_len >= 4:
        return [(0.0, 0.75), (1.5, 1.0)] if ctx.rng.random() < 0.6 else [(0.0, 1.0), (2.5, 1.0)]
    if style == "boom_chick":
        return [(float(b), 0.75) for b in range(1, beats, 2)]
    if style == "waltz":
        return [(float(b), 0.9) for b in range(1, beats)]
    if style == "pulse8":
        if d >= 0.7:
            return [(i * 0.5, 0.45) for i in range(int(bar_len * 2))]
        if d >= 0.45 and bar_len >= 4:
            return ctx.pick("comp_pulse", [
                [(float(b), 0.9) for b in range(beats)],                       # straight quarters
                [(0.0, 1.4), (1.5, 1.0), (3.0, 0.45), (3.5, 0.45)],          # pop push
                [(0.0, 0.9), (1.0, 0.45), (1.5, 1.4), (3.0, 0.9)],            # syncopated strum
            ])
        if d >= 0.45:
            return [(float(b), 0.9) for b in range(beats)]
        return [(0.0, bar_len / 2), (bar_len / 2, bar_len / 2)]
    if style in ("block", "arpeggio"):
        return [(0.0, bar_len / 2), (bar_len / 2, bar_len / 2)] if d >= 0.4 else [(0.0, bar_len)]
    return [(0.0, bar_len)]


def gen_chord_bed(ctx: GenContext) -> list[RelNote]:
    lo, hi = ctx.register("chord_bed")
    center = (lo + hi) / 2 + ctx.pick("voicing_center", [0, -3, 3])
    notes, prev = [], None
    n_voices = 4 if ctx.plan.genre_profile in ("jazz", "blues") else 3
    for bar, bs, bl in ctx.bars():
        if not ctx.active("chord_bed", bar):
            continue
        for off, dur in _comp_offsets(ctx, bl, bar):
            t = bs + off
            ch = ctx.chord(t)
            pcs = ch.pcs if n_voices == 4 or len(ch.pcs) <= 3 else [ch.pcs[0], ch.pcs[1], ch.pcs[3]]
            voicing = voice_lead(prev, voicing_candidates(pcs, lo, hi, n_voices), center)
            if not voicing:
                continue
            prev = voicing
            # Chord changes inside the hit shorten it.
            nxt = next((c.start for c in ctx.plan.chords if t < c.start < t + dur), None)
            d = (nxt - t) if nxt else dur
            accent = 6 if abs(off) < 1e-6 else -3
            for p in voicing:
                notes.append(RelNote(ctx.swing(t), max(0.1, d - 0.03), p, ctx.vel(bar, accent, base=-8)))
    return notes


def gen_pad(ctx: GenContext) -> list[RelNote]:
    lo, hi = ctx.register("pad")
    notes, prev = [], None
    spans = _merge_chord_spans(ctx.plan.chords)
    for ch in spans:
        bar = ch.bar
        if not ctx.active("pad", bar):
            continue
        pcs = ch.pcs
        # Open voicing: root low, fifth and third above.
        order = [pcs[0], pcs[2] if len(pcs) > 2 else pcs[-1], pcs[1]] + pcs[3:4]
        cands = []
        for root in tones_in_range([order[0]], lo, hi - 12):
            v = [root]
            for pc in order[1:]:
                v.append(v[-1] + ((pc - v[-1]) % 12 or 12))
            if v[-1] <= hi:
                cands.append(v)
        voicing = voice_lead(prev, cands, (lo + hi) / 2) or voice_lead(prev, voicing_candidates(pcs, lo, hi, 3), (lo + hi) / 2)
        if not voicing:
            continue
        prev = voicing
        for p in voicing:
            notes.append(RelNote(ch.start, max(0.25, ch.end - ch.start - 0.05), p, ctx.vel(bar, -2, base=-18)))
    return notes


def _merge_chord_spans(chords: list[ChordSpan]) -> list[ChordSpan]:
    out: list[ChordSpan] = []
    for c in chords:
        if out and (out[-1].root_rel, out[-1].quality) == (c.root_rel, c.quality) and abs(out[-1].end - c.start) < 1e-6 \
                and c.bar - out[-1].bar < 2:
            out[-1] = ChordSpan(out[-1].start, c.end, c.root_rel, c.quality, out[-1].bar)
        else:
            out.append(ChordSpan(c.start, c.end, c.root_rel, c.quality, c.bar))
    return out


def gen_arpeggio(ctx: GenContext) -> list[RelNote]:
    lo, hi = ctx.register("arpeggio")
    d = ctx.plan.density
    step = 0.25 if d >= 0.7 else 0.5
    if ctx.plan.time_signatures[0].numerator in (6, 9, 12):
        step = 0.5
    pattern_kind = ctx.rng.choice(["up", "updown", "broken", "alberti"]) if ctx.plan.genre_profile == "classical" \
        else ctx.rng.choice(["up", "updown", "broken"])
    notes = []
    for bar, bs, bl in ctx.bars():
        if not ctx.active("arpeggio", bar):
            continue
        n = int(round(bl / step))
        for i in range(n):
            t = bs + i * step
            ch = ctx.chord(t)
            tones = tones_in_range(ch.pcs, lo, min(hi, lo + 19))
            if not tones:
                continue
            m = len(tones)
            if pattern_kind == "up":
                p = tones[i % m]
            elif pattern_kind == "updown":
                cyc = tones + tones[-2:0:-1]
                p = cyc[i % len(cyc)]
            elif pattern_kind == "broken":
                idx = [0, 2, 1, 3, 2, 4, 3, 1]
                p = tones[idx[i % len(idx)] % m]
            else:  # alberti: low-high-mid-high
                idx = [0, 2, 1, 2]
                p = tones[idx[i % 4] % min(m, 3)]
            accent = 8 if i == 0 else (2 if (i * step) % 1.0 < 1e-6 else -6)
            notes.append(RelNote(ctx.swing(t), step * 0.95, p, ctx.vel(bar, accent, base=-10)))
    return notes


# ----------------------------------------------------------------------------- bass


def _bass_root(ctx: GenContext, ch: ChordSpan, prev: int | None, lo: int, hi: int) -> int:
    cands = tones_in_range([ch.root_rel], lo, hi)
    target = prev if prev is not None else lo + (hi - lo) * 0.35
    return nearest(cands, target) if cands else lo


def gen_bass_line(ctx: GenContext) -> list[RelNote]:
    if ctx.plan.bass_style == "walking":
        return gen_walking_bass(ctx, role="bass_line")
    lo, hi = ctx.register("bass_line")
    style, d = ctx.plan.bass_style, ctx.plan.density
    notes: list[RelNote] = []
    prev = None
    for bar, bs, bl in ctx.bars():
        if not ctx.active("bass_line", bar):
            continue
        hits: list[tuple[float, float, str]] = []   # (offset, duration, function)
        follow = ctx.plan.anchor_mode and ctx.pick("bass_follow_anchor", [True, False])
        anchor_hits = anchor_hit_positions(ctx, bl, 0.1) if follow else []
        if style == "sustained" or d < 0.2:
            hits = [(0.0, bl, "root")] if bl <= 4 else [(0.0, bl / 2, "root"), (bl / 2, bl / 2, "fifth")]
        elif style == "pedal8":
            hits = [(i * 0.5, 0.45, "root") for i in range(int(bl * 2))]
            if d < 0.5:
                hits = [(float(i), 0.9, "root") for i in range(int(bl))]
        elif style == "boom_chick":
            hits = [(float(b), 0.9, "root" if (b // 2) % 2 == 0 else "fifth") for b in range(0, int(bl), 2)]
        elif style == "root_fifth":
            hits = [(0.0, bl / 2 - 0.1, "root"), (bl / 2, bl / 2 - 0.1, "fifth")]
        else:  # pop: root on 1, syncopated push (variant chosen once per candidate)
            hits = ctx.pick("bass_pop", [
                [(0.0, 1.4, "root"), (1.5, 0.9, "root"), (2.5, 0.45, "fifth"), (3.0, 0.9, "root")],
                [(0.0, 0.9, "root"), (1.0, 0.45, "root"), (1.5, 1.4, "fifth"), (3.0, 0.45, "root"), (3.5, 0.45, "root")],
                [(0.0, 1.9, "root"), (2.0, 0.9, "fifth"), (3.0, 0.9, "root")],
            ]) if bl >= 4 else [(0.0, bl - 0.1, "root")]
            if d < 0.45:
                hits = [(0.0, 1.9, "root"), (2.0, 1.9, "root")] if bl >= 4 else [(0.0, bl - 0.1, "root")]
        if anchor_hits and d >= 0.35 and style not in ("sustained", "pedal8"):
            hits = [(h, (anchor_hits[i + 1] if i + 1 < len(anchor_hits) else bl) - h - 0.05, "root" if i == 0 else
                     ("fifth" if i % 2 else "root")) for i, h in enumerate(anchor_hits)]
            if hits[0][0] > 0:
                hits.insert(0, (0.0, hits[0][0] - 0.05, "root"))
        for off, dur, fn in hits:
            t = bs + off
            ch = ctx.chord(t)
            root = _bass_root(ctx, ch, prev, lo, hi)
            p = root
            if fn == "fifth":
                fifth_pc = ch.pcs[2] if len(ch.pcs) > 2 else ch.root_rel + 7
                cands = tones_in_range([fifth_pc], lo, hi)
                if cands:
                    p = nearest(cands, root + 5)
            notes.append(RelNote(ctx.swing(t), max(0.1, dur), p, ctx.vel(bar, 8 if off == 0 else 0, base=4)))
            prev = p
        # Approach note into a new chord on the last 8th of the bar.
        if d >= 0.55 and style in ("pop", "root_fifth", "boom_chick") and bar + 1 < ctx.plan.n_bars:
            nxt = ctx.chord(bs + bl + 0.01)
            cur_ch = ctx.chord(bs + bl - 0.01)
            if (nxt.root_rel, nxt.quality) != (cur_ch.root_rel, cur_ch.quality) and prev is not None:
                target = _bass_root(ctx, nxt, prev, lo, hi)
                appr = target - 1 if target > prev else target + 1
                if lo <= appr <= hi:
                    t = bs + bl - 0.5
                    notes = [n for n in notes if not (n.start >= t - 1e-6)]
                    for n in notes[-2:]:
                        if n.end > t:
                            n.duration = max(0.1, t - n.start)
                    notes.append(RelNote(t, 0.45, appr, ctx.vel(bar, -4)))
                    prev = appr
    return _make_monophonic(notes)


def gen_walking_bass(ctx: GenContext, role: str = "walking_bass") -> list[RelNote]:
    lo, hi = ctx.register(role) if role in ctx.plan.registers else ctx.register("bass_line")
    scale = list(ctx.mode_scale)
    notes: list[RelNote] = []
    prev = None
    total_beats = int(round(ctx.grid.bar_start(ctx.plan.n_bars - 1) + ctx.grid.bar_length(ctx.plan.n_bars - 1)))
    beat = 0
    while beat < total_beats:
        t = float(beat)
        bar, _ = ctx.grid.locate(t)
        if not ctx.active(role, bar):
            beat += 1
            continue
        ch = ctx.chord(t)
        span_end = min(ch.end, float(total_beats))
        span_beats = max(1, int(round(span_end - t)))
        nxt = ctx.chord(span_end + 0.01) if span_end < total_beats else ch
        root = _bass_root(ctx, ch, prev, lo, hi)
        target = _bass_root(ctx, nxt, root, lo, hi)
        line = [root]
        for k in range(1, span_beats):
            last = line[-1]
            remaining = span_beats - k
            if remaining == 1:
                r = ctx.rng.random()
                if r < 0.5:
                    p = target - 1 if target > last else target + 1        # chromatic approach
                elif r < 0.8:
                    p = _scale_step(target, -1 if target > last else 1, scale)   # diatonic approach
                else:
                    p = target + 7 if target + 7 <= hi else target - 5     # dominant approach
            else:
                direction = 1 if target >= last else -1
                tones = tones_in_range(ch.pcs, lo, hi)
                if ctx.rng.random() < 0.55 and tones:
                    cands = [x for x in tones if 0 < (x - last) * direction <= 5] or tones
                    p = nearest(cands, last + 3 * direction)
                else:
                    p = _scale_step(last, direction, scale)
            p = max(lo, min(hi, p))
            line.append(p)
        for k, p in enumerate(line):
            bt = t + k
            b2, _ = ctx.grid.locate(bt)
            accent = 4 if k == 0 else (2 if (int(bt) % 2 == 1) else -2)
            notes.append(RelNote(bt, 0.92, p, ctx.vel(b2, accent, base=2)))
        prev = line[-1]
        beat += span_beats
    return _make_monophonic(notes)


# ----------------------------------------------------------------------------- lines built on a reference


def gen_counterline(ctx: GenContext) -> list[RelNote]:
    lo, hi = ctx.register("counterline")
    ref = ctx.reference_line()
    scale = set(ctx.mode_scale)
    notes: list[RelNote] = []
    prev_c, prev_r = None, None
    # Sit below a melody, but above a bass/low anchor.
    above = ctx.anchor_role in ("bass", "low_texture")
    if above:
        lo, hi = max(lo, ctx.register("counterline")[0] + 7), hi + 7
    window = 2.0
    for bar, bs, bl in ctx.bars():
        if not ctx.active("counterline", bar):
            continue
        w = 0.0
        while w < bl - 1e-6:
            wl = min(window, bl - w)
            a = bs + w
            busy = ref.onsets_in(a, a + wl) if ref else 0
            if busy >= 3:
                rhythm = [(0.0, wl)]
            elif busy == 2:
                rhythm = [(0.0, wl / 2), (wl / 2, wl / 2)] if ctx.plan.density >= 0.3 else [(0.0, wl)]
            else:
                rhythm = [(0.0, 0.5), (0.5, 0.5), (1.0, 1.0)] if (ctx.plan.density >= 0.5 and wl >= 2) else \
                    [(0.0, wl / 2), (wl / 2, wl / 2)]
            for off, dur in rhythm:
                t = a + off
                ch = ctx.chord(t)
                strong = off == 0.0
                r_note = ref.at(t) if ref else None
                r = r_note.pitch if r_note else None
                pool = tones_in_range(ch.pcs if strong else (scale | set(ch.pcs)), lo, hi)
                best, best_s = None, -1e9
                for c in pool:
                    s = 0.0
                    if r is not None:
                        iv = (r - c) % 12
                        if iv in (3, 4, 8, 9):
                            s += 2.0
                        elif iv in (0, 7):
                            s += 0.8
                        elif iv == 5:
                            s += 0.2
                        else:
                            s -= 2.0 if strong else 0.6
                        if (not above and c > r - 2) or (above and c < r + 2):
                            s -= 3.0
                        if prev_c is not None and prev_r is not None:
                            prev_iv = (prev_r - prev_c) % 12
                            moved = c != prev_c and r != prev_r
                            if moved and prev_iv in (0, 7) and iv == prev_iv:
                                s -= 2.5           # parallel fifths/octaves
                            if moved and np.sign(c - prev_c) == -np.sign(r - prev_r):
                                s += 0.7           # contrary motion
                    if prev_c is not None:
                        leap = abs(c - prev_c)
                        s += 0.8 if 0 < leap <= 2 else (0.3 if leap <= 4 else (-1.0 if leap > 7 else 0.0))
                        if leap == 0:
                            s -= 0.3
                    else:
                        s -= abs(c - (lo + hi) / 2) * 0.05
                    s += ctx.rng.uniform(0, 0.35)
                    if s > best_s:
                        best, best_s = c, s
                if best is None:
                    continue
                notes.append(RelNote(ctx.swing(t), max(0.1, dur - 0.04), best, ctx.vel(bar, 2 if strong else -4, base=-6)))
                prev_c, prev_r = best, r
            w += wl
    return _make_monophonic(notes)


def gen_harmony(ctx: GenContext) -> list[RelNote]:
    """Parallel diatonic thirds (or sixths) under the reference melody."""
    lo, hi = ctx.register("harmony")
    ref = ctx.reference_line()
    scale = list(ctx.mode_scale)
    notes = []
    for n in ref.notes:
        bar, _ = ctx.grid.locate(n.start)
        if not ctx.active("harmony", min(bar, ctx.plan.n_bars - 1)):
            continue
        ch = ctx.chord(n.start)
        third = _scale_step(n.pitch, -2, scale)
        sixth = _scale_step(n.pitch, -5, scale)
        cand = third if (third % 12 in ch.pcs or n.duration < 1.0) else sixth
        while cand > hi:
            cand -= 12
        while cand < lo:
            cand += 12
        if consonance(n.pitch - cand) < 0.5:
            continue
        notes.append(RelNote(n.start, n.duration, cand, max(20, n.velocity - 14)))
    return _make_monophonic(notes)


def gen_octave_doubles(ctx: GenContext) -> list[RelNote]:
    ref = ctx.reference_line()
    if not ref:
        return []
    lo, hi = ctx.register("octave_doubles")
    med = float(np.median([n.pitch for n in ref.notes]))
    shift = 12 if med + 12 + 4 <= hi else -12
    if not (lo <= med + shift <= hi):
        shift = 12 if abs(med + 12 - (lo + hi) / 2) < abs(med - 12 - (lo + hi) / 2) else -12
    out = []
    for n in ref.notes:
        bar, _ = ctx.grid.locate(n.start)
        if bar >= ctx.plan.n_bars:
            continue
        out.append(RelNote(n.start, n.duration, n.pitch + shift, max(20, n.velocity - 12)))
    return out


# ----------------------------------------------------------------------------- textures


def gen_high_texture(ctx: GenContext) -> list[RelNote]:
    lo, hi = ctx.register("high_texture")
    notes = []
    per_bar = 1 if ctx.plan.density < 0.4 else (2 if ctx.plan.density < 0.75 else 4)
    prev = None
    for bar, bs, bl in ctx.bars():
        if not ctx.active("high_texture", bar):
            continue
        step = bl / per_bar
        for i in range(per_bar):
            t = bs + i * step + (0.5 if per_bar == 2 and i == 1 and ctx.rng.random() < 0.5 else 0.0)
            ch = ctx.chord(t)
            tones = tones_in_range([p for p in ch.pcs if p != ch.root_rel] or ch.pcs, lo, hi)
            if not tones:
                continue
            p = nearest(tones, prev + ctx.rng.choice([-3, 3, 5, -5]) if prev is not None else (lo + hi) / 2)
            prev = p
            notes.append(RelNote(t, min(step, 2.0), p, ctx.vel(bar, -4, base=-16)))
    return notes


def gen_low_texture(ctx: GenContext) -> list[RelNote]:
    lo, hi = ctx.register("low_texture")
    notes = []
    for ch in _merge_chord_spans(ctx.plan.chords):
        if not ctx.active("low_texture", ch.bar):
            continue
        root = tones_in_range([ch.root_rel], lo, hi)
        if not root:
            continue
        r = nearest(root, lo + 7)
        notes.append(RelNote(ch.start, max(0.25, ch.end - ch.start - 0.05), r, ctx.vel(ch.bar, -6, base=-20)))
        fifth = r + 7
        if fifth <= hi:
            notes.append(RelNote(ch.start, max(0.25, ch.end - ch.start - 0.05), fifth, ctx.vel(ch.bar, -8, base=-24)))
    return notes


# ----------------------------------------------------------------------------- drums


def gen_drums(ctx: GenContext) -> list[RelNote]:
    style, d, rng = ctx.plan.drum_style, ctx.plan.density, ctx.rng
    if style == "none":
        style = "orchestral"
    notes: list[RelNote] = []
    kick_bias = anchor_hit_positions(ctx, ctx.grid.bar_length(0), 0.12) if ctx.plan.anchor_mode else []

    def hit(t, note, accent=0, base=0, dur=0.2):
        b, _ = ctx.grid.locate(t)
        notes.append(RelNote(t, dur, note, ctx.vel(min(b, ctx.plan.n_bars - 1), accent, base)))

    for bar, bs, bl in ctx.bars():
        if not ctx.active("drums", bar):
            continue
        sec = ctx.plan.section_of_bar(bar)
        first = bar == sec.start_bar
        last = bar == sec.end_bar - 1 and sec.n_bars >= 4
        e = ctx.energy(bar)
        beats = int(round(bl))
        eighths = [i * 0.5 for i in range(int(bl * 2))]
        if first and e >= 0.5 and style not in ("orchestral",):
            hit(bs, DRUM["crash"], 10)
        if style in ("rock", "pop", "country", "half_time", "hiphop", "four_on_floor"):
            hat_sub = 0.25 if (d >= 0.75 and style in ("pop", "hiphop")) else 0.5
            hat_note = DRUM["ride"] if (style == "rock" and sec.name in ("chorus", "solo") and e > 0.8) else DRUM["closed_hat"]
            if style == "country":
                hat_note = DRUM["shaker"] if rng.random() < 0.5 else DRUM["closed_hat"]
            t = 0.0
            while t < bl - 1e-6:
                if not (first and t == 0 and e >= 0.5):
                    hit(bs + ctx.swing(t), hat_note, 4 if t % 1.0 == 0 else -8, base=-14)
                t += hat_sub
            if style == "four_on_floor":
                kicks = [float(b) for b in range(beats)]
                snares = [float(b) for b in range(1, beats, 2)]
            elif style == "half_time":
                kicks, snares = [0.0], [bl / 2] if bl >= 4 else [float(beats - 1)]
            elif style == "hiphop":
                kicks = [0.0, 0.75, 2.5] if bl >= 4 else [0.0]
                snares = [1.0, 3.0] if bl >= 4 else [float(beats - 1)]
            else:
                kicks = list(ctx.pick("kick", [[0.0, 2.0], [0.0, 2.0, 2.5], [0.0, 1.5, 2.0], [0.0, 0.5, 2.0]])) \
                    if bl >= 4 else [0.0]
                if d < 0.4:
                    kicks = kicks[:2]
                snares = [float(b) for b in range(1, beats, 2)]
            if kick_bias and d >= 0.4:
                kicks = sorted(set(kicks[:1] + [k for k in kick_bias if k not in snares][:3]))
            for k in kicks:
                hit(bs + k, DRUM["kick"], 8 if k == 0 else 0)
            for s in snares:
                if last and s >= bl - 1.0 - 1e-6:
                    continue
                hit(bs + s, DRUM["side_stick"] if (style == "country" and e < 0.6) else DRUM["snare"], 6)
            if last:
                _fill(ctx, bs, bl, notes, hit, rng)
        elif style == "jazz":
            for b in range(beats):
                hit(bs + b, DRUM["ride"], 2 if b % 2 else -2, base=-10)
                if b % 2 == 1:
                    hit(bs + b + 2.0 / 3.0, DRUM["ride"], -10, base=-14)
                    hit(bs + b, DRUM["pedal_hat"], -6, base=-12)
            if rng.random() < 0.5 + d * 0.3:
                hit(bs + rng.choice(eighths[1:]) + (1 / 6 if ctx.plan.swing else 0), DRUM["snare"], -18, base=-14)
            hit(bs, DRUM["kick"], -20, base=-16)
            if last:
                hit(bs + bl - 1 + 2 / 3, DRUM["snare"], -2)
        elif style == "triple":
            hit(bs, DRUM["kick"], 6)
            step = 0.5
            t = 0.0
            while t < bl - 1e-6:
                hit(bs + t, DRUM["closed_hat"], 0 if t == 0 else -8, base=-14)
                t += step
            backbeat = bl / 2 if ctx.plan.time_signatures[0].numerator in (6, 12) else 1.0
            hit(bs + backbeat, DRUM["snare"] if e > 0.5 else DRUM["side_stick"], 2)
            if ctx.plan.time_signatures[0].numerator == 3:
                hit(bs + 2.0, DRUM["side_stick"], -6)
        elif style == "hand_percussion":
            hit(bs, DRUM["low_tom"], 6)
            hit(bs + bl / 2, DRUM["high_tom"], 0)
            for t in eighths:
                if t % 1.0:
                    hit(bs + t, DRUM["shaker"], -10, base=-16)
            if d >= 0.5:
                hit(bs + bl - 0.5, DRUM["mid_tom"], -4)
        else:  # orchestral: soft bass drum and cymbal swells at section starts
            if first:
                hit(bs, DRUM["crash2"], -6, dur=2.0)
            hit(bs, DRUM["kick2"], -12, base=-10)
            if e > 0.7:
                hit(bs + bl / 2, DRUM["kick2"], -16, base=-12)
    # Final hit on the very last downbeat.
    end = ctx.grid.bar_start(ctx.plan.n_bars - 1)
    if style not in ("orchestral",):
        hit(end, DRUM["crash"], 6)
    return _dedupe_drums(notes)


def _fill(ctx, bs, bl, notes, hit, rng):
    start = bs + bl - 1.0
    seq = [DRUM["snare"], DRUM["snare"], DRUM["high_tom"], DRUM["mid_tom"], DRUM["low_tom"], DRUM["floor_tom"]]
    if rng.random() < 0.5:
        for i in range(4):
            hit(start + i * 0.25, seq[min(i + rng.randint(0, 2), len(seq) - 1)], -2 + i * 2)
    else:
        hit(start, DRUM["snare"], 0)
        hit(start + 0.5, DRUM["snare"], 4)
        hit(start + 0.75, DRUM["snare"], 6)


def _dedupe_drums(notes: list[RelNote]) -> list[RelNote]:
    seen = {}
    for n in notes:
        k = (round(n.start, 3), n.pitch)
        if k not in seen or n.velocity > seen[k].velocity:
            seen[k] = n
    return sorted(seen.values(), key=lambda n: (n.start, n.pitch))


GENERATORS = {
    "lead_melody": gen_lead_melody, "chord_bed": gen_chord_bed, "pad": gen_pad, "arpeggio": gen_arpeggio,
    "bass_line": gen_bass_line, "walking_bass": gen_walking_bass, "counterline": gen_counterline,
    "harmony": gen_harmony, "octave_doubles": gen_octave_doubles, "high_texture": gen_high_texture,
    "low_texture": gen_low_texture, "drums": gen_drums,
}

# Roles that read other generated parts must run after them.
GENERATION_ORDER = ["lead_melody", "chord_bed", "pad", "arpeggio", "bass_line", "walking_bass", "counterline",
                    "harmony", "octave_doubles", "high_texture", "low_texture", "drums"]


def order_roles(roles: list[str]) -> list[str]:
    return sorted(roles, key=lambda r: GENERATION_ORDER.index(r) if r in GENERATION_ORDER else 99)


def generate_role(role: str, ctx: GenContext) -> list[RelNote]:
    if role not in GENERATORS:
        raise ValueError(f"No rule-based generator for role {role!r}")
    end = ctx.grid.bar_start(ctx.plan.n_bars - 1) + ctx.grid.bar_length(ctx.plan.n_bars - 1)
    notes = [n for n in GENERATORS[role](ctx) if n.start < end - 1e-6]
    for n in notes:
        if n.end > end:
            n.duration = max(0.05, end - n.start)
    return notes
