"""MIDI analyzer: root/mode, tempo/meter, roles, ranges, density, rhythm, chords, sections, energy."""
from __future__ import annotations

import numpy as np

from .harmony import ChordSpan, infer_chords
from .midi_io import BarGrid, Note, Score
from .representation import STEPS_PER_QUARTER
from .roles import detect_roles, normalize_input_role
from .root import RootAnalysis, detect_root, detect_section_roots
from .theory import PITCH_NAMES, gm_name


def onset_profile(notes: list[Note], grid: BarGrid, bins_per_quarter: int = 4) -> np.ndarray:
    """Normalized histogram of onset positions within the bar (16 bins for 4/4 at 16ths)."""
    bar_len = grid.bar_length(0)
    nb = max(1, int(round(bar_len * bins_per_quarter)))
    h = np.zeros(nb)
    for n in notes:
        _, off = grid.locate(n.start)
        h[min(nb - 1, int(round(off * bins_per_quarter)) % nb)] += 1
    s = h.sum()
    return h / s if s > 0 else h


def swing_ratio(notes: list[Note]) -> float:
    """Share of off-beat onsets on the swung (triplet) position vs. the straight 8th position."""
    straight = swung = 0
    for n in notes:
        frac = n.start % 1.0
        step = int(round(frac * STEPS_PER_QUARTER)) % STEPS_PER_QUARTER
        if step == 6:
            straight += 1
        elif step == 8:
            swung += 1
    tot = straight + swung
    return swung / tot if tot >= 8 else 0.0


def syncopation(profile: np.ndarray) -> float:
    """Fraction of onsets that fall off the quarter-note beat."""
    if profile.sum() <= 0:
        return 0.0
    on_beat = profile[::4].sum()
    return float(1.0 - on_beat / profile.sum())


def bar_features(score: Score, grid: BarGrid, tonic_pc: int) -> np.ndarray:
    """Per-bar feature matrix used for section segmentation."""
    nb = grid.n_bars
    F = np.zeros((nb, 16))
    for t in score.tracks:
        for n in t.notes:
            b, _ = grid.locate(n.start)
            if b >= nb:
                continue
            if not t.is_drum:
                F[b, (n.pitch - tonic_pc) % 12] += n.duration
                F[b, 12] += n.pitch
                F[b, 13] += 1
            else:
                F[b, 15] += 1
            F[b, 14] += n.velocity
    cnt = np.maximum(F[:, 13], 1)
    F[:, 12] = F[:, 12] / cnt / 127.0
    F[:, 14] = F[:, 14] / np.maximum(F[:, 13] + F[:, 15], 1) / 127.0
    F[:, :12] = F[:, :12] / np.maximum(F[:, :12].sum(axis=1, keepdims=True), 1e-9)
    dens = F[:, 13] + F[:, 15]
    F[:, 13] = dens / max(dens.max(), 1)
    F[:, 15] = F[:, 15] / max(F[:, 15].max(), 1)
    return F


def energy_curve(score: Score, grid: BarGrid) -> list[float]:
    nb = grid.n_bars
    e = np.zeros(nb)
    for t in score.tracks:
        for n in t.notes:
            b, _ = grid.locate(n.start)
            if b < nb:
                e[b] += n.velocity / 127.0
    if e.max() > 0:
        e = e / e.max()
    return [round(float(x), 3) for x in e]


def detect_sections(score: Score, grid: BarGrid, tonic_pc: int, chords: list[ChordSpan] | None = None,
                    min_len: int = 4) -> list[dict]:
    """Segment into phrase-aligned sections via feature novelty, then label by similarity (A, B, A', ...)."""
    nb = grid.n_bars
    if nb <= 2 * min_len:
        return [_section(grid, 0, nb, "A", 0)]
    F = bar_features(score, grid, tonic_pc)
    novelty = {}
    for b in range(min_len, nb - min_len + 1, min_len):
        before = F[max(0, b - min_len):b].mean(axis=0)
        after = F[b:b + min_len].mean(axis=0)
        novelty[b] = float(np.linalg.norm(before - after))
    vals = np.array(list(novelty.values()))
    thr = vals.mean() + 0.5 * vals.std() if len(vals) > 1 else 1e9
    bounds = [0]
    for b in sorted(novelty):
        if novelty[b] >= thr and b - bounds[-1] >= min_len:
            bounds.append(b)
    # Avoid very long sections: split at 8-bar multiples if a section exceeds 16 bars.
    refined = []
    for i, s in enumerate(bounds):
        e = bounds[i + 1] if i + 1 < len(bounds) else nb
        refined.append(s)
        x = s + 16
        while e - x >= min_len:
            refined.append(x)
            x += 8 if (e - x) >= 8 + min_len else (e - x)
    bounds = sorted(set(refined))
    sections = []
    labels: list[tuple[str, np.ndarray]] = []
    letters = "ABCDEFGHIJ"
    for i, s in enumerate(bounds):
        e = bounds[i + 1] if i + 1 < len(bounds) else nb
        feat = F[s:e].mean(axis=0)
        label = None
        for lab, f in labels:
            if np.linalg.norm(f - feat) < 0.12:
                label = lab + "'"
                break
        if label is None:
            label = letters[min(len(labels), len(letters) - 1)]
            labels.append((label, feat))
        energy = float(F[s:e, 13].mean())
        sections.append(_section(grid, s, e, label, i, energy))
    return sections


def _section(grid: BarGrid, s: int, e: int, label: str, idx: int, energy: float = 0.5) -> dict:
    return {"index": idx, "label": label, "start_bar": s, "end_bar": e, "n_bars": e - s,
            "start_beat": grid.bar_start(s), "end_beat": grid.bar_start(e) if e < grid.n_bars else
            grid.bar_start(e - 1) + grid.bar_length(e - 1), "energy": round(energy, 3)}


def analyze_score(score: Score, root_override: str | int | None = None, input_role: str | None = None,
                  include_chords: bool = True) -> dict:
    grid = score.bar_grid()
    root: RootAnalysis = detect_root(score, override=root_override)
    roles = detect_roles(score)
    input_role_n = normalize_input_role(input_role) if input_role else None
    warnings = list(score.load_warnings)

    # Apply the user's declared input role.
    assigned = {}
    pitched_tracks = [t for t in score.tracks if not t.is_drum and t.notes]
    for t in score.tracks:
        r = roles[t.track_id]["role"]
        if input_role_n and input_role_n != "full_mix" and not t.is_drum:
            r = input_role_n
        assigned[t.track_id] = r
        t.role = r
    if input_role_n and input_role_n != "full_mix" and len(pitched_tracks) > 1:
        warnings.append(f"Input role '{input_role_n}' applied to all {len(pitched_tracks)} pitched tracks; "
                        "choose 'full_mix' to keep per-track detected roles.")

    if root.root_pitch_class is None:
        warnings.append("Root could not be determined; choose a manual root before generating.")
    elif root.ambiguous:
        cands = ", ".join(f"{c['root_name']} {c['mode']}" for c in root.top_candidates[:3])
        warnings.append(f"Root detection is ambiguous (confidence {root.root_confidence:.2f}); "
                        f"top candidates: {cands}. Consider a manual override.")
    if len(score.tempos) > 1:
        warnings.append(f"Tempo map has {len(score.tempos)} tempo changes; generation follows the map.")
    if len(score.time_signatures) > 1:
        warnings.append(f"{len(score.time_signatures)} time-signature changes present.")

    pitched_notes = [n for t in score.tracks if not t.is_drum for n in t.notes]
    all_notes = [n for t in score.tracks for n in t.notes]
    prof = onset_profile(all_notes, grid)
    chords: list[ChordSpan] = []
    sections: list[dict] = []
    tonic = root.root_pitch_class
    if tonic is not None and pitched_notes:
        anchor_role = "full_mix" if len(pitched_tracks) > 1 else (assigned.get(pitched_tracks[0].track_id, "chordal")
                                                                   if pitched_tracks else "chordal")
        if include_chords:
            chords = infer_chords(pitched_notes, tonic, root.mode or "major", grid, anchor_role=anchor_role)
        sections = detect_sections(score, grid, tonic, chords)
        sections = detect_section_roots(score, sections, root)
    else:
        sections = [_section(grid, 0, grid.n_bars, "A", 0)]

    tracks = []
    for t in score.tracks:
        info = roles[t.track_id]
        tracks.append({
            "track_id": t.track_id, "name": t.name, "program": t.program, "instrument": gm_name(t.program, t.is_drum),
            "channel": t.channel, "is_drum": t.is_drum, "n_notes": len(t.notes),
            "detected_role": info["role"], "role_confidence": info["confidence"], "role_reason": info["reason"],
            "assigned_role": assigned[t.track_id],
            "pitch_range": None if t.is_drum or not t.notes else [info["stats"]["min_pitch"], info["stats"]["max_pitch"]],
            "stats": info["stats"],
        })
    ts = score.initial_time_signature
    return {
        "root": root.to_dict(),
        "tempo_bpm": round(score.initial_bpm, 2),
        "tempo_map": [{"beat": t.beat, "bpm": t.bpm} for t in score.tempos],
        "meter": ts.label,
        "time_signatures": [{"beat": s.beat, "meter": s.label} for s in score.time_signatures],
        "key_signatures": [{"beat": k.beat, "key": k.key} for k in score.key_signatures],
        "duration_seconds": round(score.duration_seconds, 2),
        "total_beats": round(score.end_beat, 3),
        "n_bars": grid.n_bars,
        "ticks_per_beat": score.ticks_per_beat,
        "tracks": tracks,
        "instrumentation": sorted({gm_name(t.program, t.is_drum) for t in score.tracks}),
        "note_density_per_bar": round(len(all_notes) / max(1, grid.n_bars), 3),
        "rhythm": {"onset_profile_16th": [round(float(x), 4) for x in prof],
                   "syncopation": round(syncopation(prof), 3), "swing_ratio": round(swing_ratio(all_notes), 3)},
        "chords": [c.to_dict(tonic) for c in chords],
        "sections": sections,
        "energy_curve": energy_curve(score, grid),
        "warnings": warnings,
        "root_names": PITCH_NAMES,
    }
