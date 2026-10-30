"""Candidate scoring and ranking.

Separate components (all in [0, 1]):
  quality  : validity, key consistency, harmonic compatibility, rhythmic compatibility,
             structural coherence, instrument range, density appropriateness
  genre    : classifier probability (if a trained classifier is available) + style heuristics
  emotion  : classifier probability (if available) + valence/arousal heuristics
  novelty  : diversity w.r.t. the other candidates and non-copying of the anchor
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from .analyzer import onset_profile
from .classifiers import ClassifierScorer
from .features import score_features
from .harmony import chord_at, consonance
from .midi_io import Score, Track
from .planner import CompositionPlan
from .root import RootAnalysis
from .styles import resolve_genre
from .theory import MODE_FAMILY, instrument_range
from .validation import ValidationReport, key_fraction

QUALITY_WEIGHTS = {"validity": 2.0, "key_consistency": 1.0, "harmonic_compatibility": 1.5,
                   "rhythmic_compatibility": 1.0, "structure": 0.75, "range": 0.75, "density": 0.75}
TOTAL_WEIGHTS = {"quality": 3.0, "genre": 0.75, "emotion": 0.75, "novelty": 0.5}

# Expected notes per bar at density 0.5 per generated role (pitched notes; chords count each voice).
ROLE_DENSITY = {"lead_melody": 4.5, "chord_bed": 9, "pad": 3.5, "arpeggio": 8, "bass_line": 4, "walking_bass": 4,
                "counterline": 3, "harmony": 4, "octave_doubles": 4, "high_texture": 2, "low_texture": 1.5,
                "drums": 14}
COPY_ROLES = {"octave_doubles", "harmony"}


@dataclass
class CandidateInfo:
    score: Score
    generated_ids: list[int]
    anchor_ids: list[int]
    role_of_track: dict[int, str]          # track id -> generated role (or anchor role)
    validation: ValidationReport
    signature: set = field(default_factory=set)


def note_set_signature(score: Score, ids: list[int]) -> set:
    sig = set()
    for t in score.tracks:
        if t.track_id in ids:
            for n in t.notes:
                sig.add((t.track_id if t.is_drum else -1, round(n.start * 4), n.pitch % 12 if not t.is_drum else n.pitch))
    return sig


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


class _Sounding:
    """Index of notes for 'what sounds at time t' queries."""

    def __init__(self, notes):
        self.notes = sorted(notes, key=lambda n: n.start)
        self.starts = [n.start for n in self.notes]

    def at(self, t: float, max_back: float = 8.0):
        i = bisect.bisect_right(self.starts, t + 1e-9)
        out = []
        j = i - 1
        while j >= 0 and self.starts[j] >= t - max_back:
            n = self.notes[j]
            if n.start <= t + 1e-9 < n.start + n.duration:
                out.append(n.pitch)
            j -= 1
        return out


def harmonic_compatibility(gen_tracks: list[Track], ref_notes, plan: CompositionPlan) -> float:
    if not gen_tracks:
        return 1.0
    idx = _Sounding(ref_notes)
    cons_tot = cons_w = 0.0
    chord_hits = chord_tot = 0.0
    for t in gen_tracks:
        for n in t.notes:
            strong = abs((n.start % 2.0)) < 1e-6
            w = 2.0 if strong else 1.0
            others = idx.at(n.start + 1e-4)
            if others:
                c = np.mean([consonance(n.pitch - o) for o in others])
                cons_tot += w * c
                cons_w += w
            if strong:
                ch = chord_at(plan.chords, n.start)
                if ch is not None:
                    chord_tot += 1
                    chord_hits += ((n.pitch - plan.root_pc) % 12) in ch.pcs
    cons = cons_tot / cons_w if cons_w else 1.0
    fit = chord_hits / chord_tot if chord_tot else 1.0
    return float(0.6 * cons + 0.4 * fit)


def rhythmic_compatibility(gen_tracks: list[Track], anchor_tracks: list[Track], plan: CompositionPlan,
                           score: Score) -> float:
    grid = score.bar_grid()
    gen_notes = [n for t in gen_tracks for n in t.notes]
    if not gen_notes:
        return 0.0
    if anchor_tracks:
        anc = [n for t in anchor_tracks for n in t.notes]
        pa, pg = onset_profile(anc, grid), onset_profile(gen_notes, grid)
        cos = float(np.dot(pa, pg) / max(1e-9, np.linalg.norm(pa) * np.linalg.norm(pg)))
        gen_on = {round(n.start * 4) for n in gen_notes}
        strong = [round(n.start * 4) for n in anc if abs(n.start % 2.0) < 1e-6]
        cover = sum(1 for s in strong if s in gen_on) / len(strong) if strong else 1.0
        return 0.5 * cos + 0.5 * cover
    # Metadata mode: bass/kick alignment and on-grid onsets.
    on_grid = np.mean([abs(n.start * 12 - round(n.start * 12)) < 1e-3 for n in gen_notes])
    kick = {round(n.start * 4) for t in gen_tracks if t.is_drum for n in t.notes if n.pitch in (35, 36)}
    bass = [round(n.start * 4) for t in gen_tracks if t.role == "bass" for n in t.notes]
    align = sum(1 for b in bass if b in kick or b % 4 == 0) / len(bass) if bass else 1.0
    return float(0.5 * on_grid + 0.5 * align)


def structural_coherence(gen_tracks: list[Track], plan: CompositionPlan, score: Score) -> float:
    """Repeated sections should be recognisably similar (not identical, not unrelated)."""
    grid = plan.grid()
    groups: dict[str, list] = {}
    for s in plan.sections:
        groups.setdefault(s.name.rstrip("'"), []).append(s)
    sims = []
    for secs in groups.values():
        if len(secs) < 2:
            continue
        a, b = secs[0], secs[1]
        n = min(a.n_bars, b.n_bars)
        for t in gen_tracks:
            sa = {(round((x.start - grid.bar_start(a.start_bar)) * 4), x.pitch % 12) for x in t.notes
                  if grid.bar_start(a.start_bar) <= x.start < grid.bar_start(a.start_bar + n)}
            sb = {(round((x.start - grid.bar_start(b.start_bar)) * 4), x.pitch % 12) for x in t.notes
                  if grid.bar_start(b.start_bar) <= x.start < grid.bar_start(b.start_bar + n)}
            if sa or sb:
                sims.append(_jaccard(sa, sb))
    rep = 1.0 - min(1.0, abs(float(np.mean(sims)) - 0.75) / 0.75) if sims else 0.75
    # Section endings land on chord tones.
    ends_ok = ends = 0
    for s in plan.sections:
        end_beat = grid.bar_start(s.end_bar - 1) + grid.bar_length(s.end_bar - 1)
        for t in gen_tracks:
            if t.is_drum or t.role not in ("lead", "bass", "counterline"):
                continue
            last = [n for n in t.notes if n.start < end_beat]
            if last:
                n = max(last, key=lambda x: x.start)
                ch = chord_at(plan.chords, n.start)
                ends += 1
                ends_ok += ch is not None and ((n.pitch - plan.root_pc) % 12) in ch.pcs
    cad = ends_ok / ends if ends else 1.0
    return float(0.6 * rep + 0.4 * cad)


def density_score(gen_tracks: list[Track], role_of_track: dict[int, str], plan: CompositionPlan) -> float:
    vals = []
    for t in gen_tracks:
        role = role_of_track.get(t.track_id)
        exp = ROLE_DENSITY.get(role, 4) * (0.5 + plan.density)
        act = len(t.notes) / max(1, plan.n_bars)
        if role in COPY_ROLES:
            continue
        vals.append(math.exp(-abs(math.log(max(act, 0.05) / exp)) * 0.8))
    return float(np.mean(vals)) if vals else 1.0


def range_score(gen_tracks: list[Track]) -> float:
    tot = ok = 0
    for t in gen_tracks:
        if t.is_drum:
            continue
        lo, hi = instrument_range(t.program)
        tot += len(t.notes)
        ok += sum(lo <= n.pitch <= hi for n in t.notes)
    return ok / tot if tot else 1.0


def genre_emotion_scores(score: Score, plan: CompositionPlan, classifier: ClassifierScorer | None,
                         root: RootAnalysis) -> dict:
    out = {"classifier": None}
    profile, _ = resolve_genre(plan.genre)
    # Heuristic genre fit.
    lo, hi = profile.tempo_range
    tempo_fit = 1.0 if lo <= plan.tempo <= hi else max(0.0, 1 - min(abs(plan.tempo - lo), abs(plan.tempo - hi)) / 40)
    has_drums = any(t.is_drum and t.notes for t in score.tracks)
    drum_fit = 1.0 if has_drums == (profile.drum_style not in ("none", "orchestral")) else 0.6
    out["genre_heuristic"] = round(0.5 * tempo_fit + 0.5 * drum_fit, 4)
    # Heuristic emotion fit: mode vs valence, loudness/density vs arousal.
    fam = MODE_FAMILY.get(plan.mode, "major")
    mode_fit = 1.0 if (plan.valence < -0.2 and fam == "minor") or (plan.valence > 0.2 and fam == "major") else (
        0.7 if abs(plan.valence) <= 0.2 else 0.35)
    vels = [n.velocity for t in score.tracks for n in t.notes]
    loud = (np.mean(vels) - 40) / 70 if vels else 0.5
    arousal_fit = 1 - min(1.0, abs(loud - (0.5 + 0.4 * plan.arousal)))
    out["emotion_heuristic"] = round(0.6 * mode_fit + 0.4 * arousal_fit, 4)
    g_cls = e_cls = None
    if classifier is not None:
        probs = classifier.predict(score_features(score, root))
        out["classifier"] = {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:3]) for k, v in probs.items()}
        if plan.genre in probs["genre"]:
            g_cls = probs["genre"][plan.genre] / max(probs["genre"].values())
        if plan.emotion in probs["emotion"]:
            e_cls = probs["emotion"][plan.emotion] / max(probs["emotion"].values())
    out["genre_classifier"] = None if g_cls is None else round(float(g_cls), 4)
    out["emotion_classifier"] = None if e_cls is None else round(float(e_cls), 4)
    out["genre"] = out["genre_heuristic"] if g_cls is None else 0.5 * out["genre_heuristic"] + 0.5 * g_cls
    out["emotion"] = out["emotion_heuristic"] if e_cls is None else 0.5 * out["emotion_heuristic"] + 0.5 * e_cls
    return out


def score_candidate(cand: CandidateInfo, plan: CompositionPlan, classifier: ClassifierScorer | None,
                    others: list[set], anchor_signature: set | None = None) -> dict:
    s = cand.score
    gen = [t for t in s.tracks if t.track_id in cand.generated_ids]
    anchor = [t for t in s.tracks if t.track_id in cand.anchor_ids]
    pitched_gen = [t for t in gen if not t.is_drum]
    ref_notes = [n for t in (anchor if anchor else gen) if not t.is_drum for n in t.notes]
    comps = {}
    v = cand.validation
    comps["validity"] = 1.0 if v.passed else max(0.0, 1 - 0.25 * len(v.errors))
    fr = [key_fraction(t, plan.root_pc, plan.mode, plan.chords) for t in pitched_gen if t.notes]
    comps["key_consistency"] = float(np.mean(fr)) if fr else 1.0
    if anchor:
        comps["harmonic_compatibility"] = harmonic_compatibility(pitched_gen, ref_notes, plan)
    else:
        # Each generated part against all the other generated parts.
        vals = []
        for t in pitched_gen:
            others_notes = [n for u in pitched_gen if u is not t for n in u.notes]
            vals.append(harmonic_compatibility([t], others_notes, plan))
        comps["harmonic_compatibility"] = float(np.mean(vals)) if vals else 1.0
    comps["rhythmic_compatibility"] = rhythmic_compatibility(gen, anchor, plan, s)
    comps["structure"] = structural_coherence(gen, plan, s)
    comps["range"] = range_score(gen)
    comps["density"] = density_score(gen, cand.role_of_track, plan)
    quality = sum(QUALITY_WEIGHTS[k] * comps[k] for k in QUALITY_WEIGHTS) / sum(QUALITY_WEIGHTS.values())
    root = RootAnalysis(plan.root_pc, plan.root_name, plan.mode, MODE_FAMILY.get(plan.mode, "major"), 1.0,
                        "plan", False)
    ge = genre_emotion_scores(s, plan, classifier, root)
    sig = cand.signature
    nov = 1.0 - max((_jaccard(sig, o) for o in others), default=0.0)
    if anchor_signature:
        copy_like = [t.track_id for t in gen if cand.role_of_track.get(t.track_id) not in COPY_ROLES]
        if copy_like:
            nov = min(nov, 1.0 - 0.5 * _jaccard(note_set_signature(s, copy_like), anchor_signature))
    comps["novelty"] = float(nov) if others or anchor_signature else 1.0
    total = (TOTAL_WEIGHTS["quality"] * quality + TOTAL_WEIGHTS["genre"] * ge["genre"]
             + TOTAL_WEIGHTS["emotion"] * ge["emotion"] + TOTAL_WEIGHTS["novelty"] * comps["novelty"]) / sum(TOTAL_WEIGHTS.values())
    if not v.passed:
        total *= 0.5        # invalid candidates always rank below valid ones of similar quality
    return {"total": round(float(total), 4), "quality": round(float(quality), 4), "genre": round(float(ge["genre"]), 4),
            "emotion": round(float(ge["emotion"]), 4), "novelty": round(comps["novelty"], 4),
            "components": {k: round(float(x), 4) for k, x in comps.items()},
            "genre_detail": {k: ge[k] for k in ("genre_heuristic", "genre_classifier")},
            "emotion_detail": {k: ge[k] for k in ("emotion_heuristic", "emotion_classifier")},
            "classifier_top": ge["classifier"]}
