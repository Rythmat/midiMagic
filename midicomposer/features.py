"""Fixed-length, key-invariant feature vectors for genre/emotion classifiers and ranking."""
from __future__ import annotations

import math

import numpy as np

from .analyzer import onset_profile, swing_ratio, syncopation
from .midi_io import Score
from .root import RootAnalysis
from .theory import GM_FAMILIES, MODE_FAMILY, drum_group, gm_family

DRUM_GROUP_ORDER = ["kick", "snare", "hat", "ride", "crash", "tom", "other"]

FEATURE_NAMES = (
    [f"rel_pc_{i}" for i in range(12)]
    + ["pitch_mean", "pitch_std", "pitch_range", "log_notes_per_bar", "polyphony",
       "log_dur_mean", "log_dur_std", "vel_mean", "vel_std", "log_tempo", "default_tempo",
       "has_drums", "log_drum_per_bar"]
    + [f"drum_{g}" for g in DRUM_GROUP_ORDER]
    + [f"fam_{f}" for f in GM_FAMILIES]
    + [f"onset_{i}" for i in range(16)]
    + ["syncopation", "swing", "mode_major", "mode_minor", "mode_modal", "n_tracks"]
)
N_FEATURES = len(FEATURE_NAMES)


def score_features(score: Score, root: RootAnalysis | None) -> np.ndarray:
    f = np.zeros(N_FEATURES, dtype=np.float32)
    grid = score.bar_grid()
    nb = max(1, grid.n_bars)
    pitched = [n for t in score.tracks if not t.is_drum for n in t.notes]
    drums = [n for t in score.tracks if t.is_drum for n in t.notes]
    i = 0
    if root is not None and root.root_pitch_class is not None and pitched:
        h = np.zeros(12)
        for n in pitched:
            h[(n.pitch - root.root_pitch_class) % 12] += min(n.duration, 8.0)
        f[0:12] = h / max(h.sum(), 1e-9)
    i = 12
    if pitched:
        p = np.array([n.pitch for n in pitched], dtype=float)
        d = np.array([max(n.duration, 1e-3) for n in pitched])
        v = np.array([n.velocity for n in pitched], dtype=float)
        onsets = len({round(n.start * 12) for n in pitched})
        f[i:i + 11] = [p.mean() / 127, p.std() / 24, (p.max() - p.min()) / 88, math.log1p(len(pitched) / nb) / 5,
                       len(pitched) / max(onsets, 1) / 4, (np.log2(d).mean() + 4) / 8, np.log2(d).std() / 4,
                       v.mean() / 127, v.std() / 64, 0, 0]
    bpm = score.initial_bpm
    f[i + 9] = math.log(max(bpm, 20)) / math.log(240)
    f[i + 10] = 1.0 if abs(bpm - 119) < 1.5 or abs(bpm - 120) < 0.01 else 0.0
    i += 11
    f[i] = 1.0 if drums else 0.0
    f[i + 1] = math.log1p(len(drums) / nb) / 5
    i += 2
    if drums:
        dg = np.zeros(len(DRUM_GROUP_ORDER))
        for n in drums:
            dg[DRUM_GROUP_ORDER.index(drum_group(n.pitch))] += 1
        f[i:i + 7] = dg / dg.sum()
    i += 7
    fam = np.zeros(16)
    for t in score.tracks:
        if not t.is_drum:
            fam[GM_FAMILIES.index(gm_family(t.program))] += len(t.notes)
    if fam.sum() > 0:
        f[i:i + 16] = fam / fam.sum()
    i += 16
    prof = onset_profile([n for t in score.tracks for n in t.notes], grid)
    if len(prof) == 16:
        f[i:i + 16] = prof
    else:   # resample other meters onto 16 bins
        f[i:i + 16] = np.interp(np.linspace(0, len(prof) - 1, 16), np.arange(len(prof)), prof) if len(prof) > 1 else 0
    i += 16
    f[i] = syncopation(prof)
    f[i + 1] = swing_ratio([n for t in score.tracks for n in t.notes])
    if root is not None and root.mode:
        fam_m = MODE_FAMILY.get(root.mode, "major")
        f[i + 2] = 1.0 if fam_m == "major" else 0.0
        f[i + 3] = 1.0 if fam_m == "minor" else 0.0
        f[i + 4] = 0.0 if root.mode in ("major", "minor") else 1.0
    f[i + 5] = len(score.tracks) / 10
    return f
