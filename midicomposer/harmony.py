"""Chord inference in root-relative space.

A single Viterbi decoder handles chordal material (template fit), bass lines (root on
the bass note) and bare melodies (chord-tone coverage of strong beats). Emissions are
weighted by the anchor's role; transitions encode a light functional-harmony prior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .midi_io import BarGrid, Note
from .theory import CHORD_QUALITIES, MODE_FAMILY, chord_label, chord_pcs, scale_pcs

CANDIDATE_QUALITIES = ["maj", "min", "dim", "7", "sus4", "min7", "maj7"]


@dataclass
class ChordSpan:
    start: float            # quarter-note beats
    end: float
    root_rel: int           # semitones above the tonic (0-11)
    quality: str
    bar: int

    @property
    def pcs(self) -> list[int]:
        return chord_pcs(self.root_rel, self.quality)

    def label(self, tonic_pc: int | None = None) -> str:
        return chord_label(self.root_rel, self.quality, tonic_pc)

    def to_dict(self, tonic_pc: int | None = None) -> dict:
        d = asdict(self)
        d["relative_label"] = self.label()
        if tonic_pc is not None:
            d["name"] = self.label(tonic_pc)
        return d


def _diatonic_set(mode: str) -> set[tuple[int, str]]:
    sc = scale_pcs(mode)
    out = set()
    for deg in range(7):
        root = sc[deg]
        third = (sc[(deg + 2) % 7] - root) % 12
        fifth = (sc[(deg + 4) % 7] - root) % 12
        sev = (sc[(deg + 6) % 7] - root) % 12
        tri = {(4, 7): "maj", (3, 7): "min", (3, 6): "dim", (4, 8): "aug"}.get((third, fifth))
        if tri:
            out.add((root, tri))
            sev_q = {("maj", 11): "maj7", ("maj", 10): "7", ("min", 10): "min7", ("dim", 10): None}.get((tri, sev))
            if sev_q:
                out.add((root, sev_q))
        out.add((root, "sus4"))
    if MODE_FAMILY.get(mode, "major") == "minor":
        out.add((7, "maj"))   # harmonic-minor dominant
        out.add((7, "7"))
    return out


def _candidates(mode: str) -> list[tuple[int, str, float]]:
    dia = _diatonic_set(mode)
    out = []
    for r in range(12):
        for q in CANDIDATE_QUALITIES:
            prior = 0.12 if (r, q) in dia else (-0.05 if q in ("maj", "min", "7") else -0.2)
            if q in ("sus4", "maj7", "min7"):
                prior -= 0.03   # prefer plain triads unless evidence is strong
            out.append((r, q, prior))
    return out


def _transition(prev: tuple[int, str], cur: tuple[int, str], at_barline: bool) -> float:
    if prev == cur:
        return 0.0
    cost = -0.04 if at_barline else -0.25
    interval = (cur[0] - prev[0]) % 12
    if interval == 5:            # root motion down a fifth (V->I, ii->V, ...)
        cost += 0.06
    elif interval in (2, 10, 9, 3):
        cost += 0.02
    return cost


def window_histograms(notes: list[Note], tonic_pc: int, starts: list[float], ends: list[float],
                      strong_beats: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-window relative pitch-class weights and per-window bass pitch class (-1 if none)."""
    n = len(starts)
    H = np.zeros((n, 12))
    bass = np.full(n, -1)
    bass_pitch = np.full(n, 10 ** 6)
    if not notes:
        return H, bass
    idx_sorted = sorted(range(n), key=lambda i: starts[i])
    srt = [starts[i] for i in idx_sorted]
    import bisect
    for note in notes:
        rel_pc = (note.pitch - tonic_pc) % 12
        i0 = bisect.bisect_right(srt, note.start) - 1
        i0 = max(0, i0)
        for j in range(i0, n):
            w = idx_sorted[j]
            s, e = starts[w], ends[w]
            if s >= note.end:
                break
            ov = min(e, note.end) - max(s, note.start)
            if ov <= 0:
                continue
            weight = ov * (0.5 + note.velocity / 254.0)
            if strong_beats and abs(note.start - s) < 1e-6:
                weight *= 1.5       # notes that start with the window are structurally stronger
            H[w, rel_pc] += weight
            if note.pitch < bass_pitch[w]:
                bass_pitch[w] = note.pitch
                bass[w] = rel_pc
    return H, bass


def infer_chords(notes: list[Note], tonic_pc: int, mode: str, grid: BarGrid, n_bars: int | None = None,
                 anchor_role: str = "chordal", half_bar: bool = True) -> list[ChordSpan]:
    """Infer a chord progression (root-relative) from pitched notes."""
    n_bars = n_bars or grid.n_bars
    starts, ends, bars, barline = [], [], [], []
    for b in range(n_bars):
        s, L = grid.bar_start(b), grid.bar_length(b)
        if half_bar and L >= 4.0 - 1e-9:
            starts += [s, s + L / 2]
            ends += [s + L / 2, s + L]
            bars += [b, b]
            barline += [True, False]
        else:
            starts.append(s)
            ends.append(s + L)
            bars.append(b)
            barline.append(True)
    H, bass = window_histograms(notes, tonic_pc, starts, ends)
    cands = _candidates(mode)
    bass_w = {"bass": 0.45, "low_texture": 0.35, "chordal": 0.2, "full_mix": 0.2, "pad": 0.2}.get(anchor_role, 0.05)
    melody_like = anchor_role in ("lead", "counterline", "high_texture", "harmony")

    E = np.zeros((len(starts), len(cands)))
    for w in range(len(starts)):
        tot = H[w].sum()
        for ci, (r, q, prior) in enumerate(cands):
            if tot <= 0:
                E[w, ci] = prior + (0.3 if (r, q) == (0, MODE_FAMILY.get(mode, "major")[:3]) else 0.0)
                continue
            pcs = chord_pcs(r, q)
            inside = sum(H[w, p] for p in pcs) / tot
            # Penalise template tones that are absent (only for dense material).
            missing = sum(1 for p in pcs if H[w, p] <= 1e-9) / len(pcs)
            fit = inside - (0.15 if melody_like else 0.35) * missing
            if bass[w] >= 0 and bass[w] == r:
                fit += bass_w
            elif bass[w] >= 0 and bass[w] in pcs:
                fit += bass_w * 0.3
            E[w, ci] = fit + prior
    # Viterbi
    nW, nC = E.shape
    if nW == 0:
        return []
    keys = [(r, q) for r, q, _ in cands]
    tonic_q = "maj" if MODE_FAMILY.get(mode, "major") == "major" else "min"
    start_bonus = np.array([0.08 if k == (0, tonic_q) else 0.0 for k in keys])
    D = E[0] + start_bonus
    back = np.zeros((nW, nC), dtype=int)
    T_bar = np.array([[_transition(a, b, True) for b in keys] for a in keys])
    T_mid = np.array([[_transition(a, b, False) for b in keys] for a in keys])
    for w in range(1, nW):
        T = T_bar if barline[w] else T_mid
        M = D[:, None] + T
        back[w] = M.argmax(axis=0)
        D = M.max(axis=0) + E[w]
    D = D + np.array([0.06 if k == (0, tonic_q) else 0.0 for k in keys])
    path = [int(D.argmax())]
    for w in range(nW - 1, 0, -1):
        path.append(int(back[w, path[-1]]))
    path.reverse()
    spans: list[ChordSpan] = []
    for w, ci in enumerate(path):
        r, q = keys[ci]
        if spans and (spans[-1].root_rel, spans[-1].quality) == (r, q) and not barline[w]:
            spans[-1].end = ends[w]
        else:
            spans.append(ChordSpan(starts[w], ends[w], r, q, bars[w]))
    return spans


def chord_at(chords: list[ChordSpan], beat: float) -> ChordSpan | None:
    for c in chords:
        if c.start - 1e-9 <= beat < c.end - 1e-9:
            return c
    return chords[-1] if chords else None


def consonance(interval: int) -> float:
    """1.0 for perfect/imperfect consonances, 0.5 for the fourth, 0 for dissonances."""
    ic = interval % 12
    if ic in (0, 3, 4, 7, 8, 9):
        return 1.0
    if ic == 5:
        return 0.5
    return 0.0


QUALITY_SET = set(CHORD_QUALITIES)
