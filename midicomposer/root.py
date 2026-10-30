"""Global tonal-root (tonic) detection.

"Root" here means the global tonal centre of the piece, not per-chord roots.

Evidence, in order of precedence:
  1. A MIDI key-signature event, accepted only if it agrees with the pitch content.
  2. A duration- and velocity-weighted pitch-class histogram matched against
     Krumhansl-Kessler major/minor key profiles.
  3. Supporting evidence: a bass-line histogram (lowest sounding pitch over time) and
     the final bass note (cadence). Neither is ever used on its own.
  4. Modal refinement (dorian, mixolydian, lydian, phrygian) from characteristic degrees.

Ambiguous results are flagged (low confidence + top candidates); the detector never
silently falls back to C. Files with no pitched content get `root_pitch_class=None`.
"""
from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

from .midi_io import KeySignature, Note, Score
from .theory import MODE_FAMILY, PITCH_NAMES, normalize_mode, parse_key, parse_pitch_class

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

W_ALL, W_BASS, W_FINAL = 0.75, 0.20, 0.05
DEFAULT_AMBIGUITY_THRESHOLD = 0.40
MAX_NOTE_WEIGHT_BEATS = 8.0     # cap very long sustains so one pedal note can't dominate


class RootUndeterminedError(ValueError):
    """Raised when a root is required but could not be determined and no override was given."""


@dataclass
class RootAnalysis:
    root_pitch_class: int | None
    root_name: str | None
    mode: str | None
    mode_family: str | None
    root_confidence: float
    root_detection_method: str
    ambiguous: bool
    requires_override: bool = False
    top_candidates: list[dict] = field(default_factory=list)
    key_signature: str | None = None
    key_signature_agrees: bool | None = None
    evidence: dict = field(default_factory=dict)

    @property
    def root_reference_midi(self) -> int | None:
        return None if self.root_pitch_class is None else 60 + self.root_pitch_class

    @property
    def key_name(self) -> str | None:
        return None if self.root_name is None else f"{self.root_name} {self.mode}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["root_reference_midi"] = self.root_reference_midi
        d["key_name"] = self.key_name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RootAnalysis":
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})


# ----------------------------------------------------------------------------- histograms


def pitch_class_histogram(notes: Iterable[Note]) -> np.ndarray:
    """Duration x velocity weighted pitch-class histogram (pitched notes only)."""
    h = np.zeros(12)
    for n in notes:
        h[n.pitch % 12] += min(n.duration, MAX_NOTE_WEIGHT_BEATS) * (n.velocity / 127.0)
    return h


def bass_histogram(notes: list[Note]) -> tuple[np.ndarray, int | None]:
    """Histogram of the lowest sounding pitch class over time, plus the final bass pitch class."""
    h = np.zeros(12)
    if not notes:
        return h, None
    events = []
    for i, n in enumerate(notes):
        events.append((n.start, 1, i))
        events.append((n.end, 0, i))
    events.sort()
    heap: list[tuple[int, int]] = []
    alive: set[int] = set()
    prev_t = events[0][0]
    for t, kind, i in events:
        if t > prev_t and alive:
            while heap and heap[0][1] not in alive:
                heapq.heappop(heap)
            if heap:
                h[heap[0][0] % 12] += min(t - prev_t, MAX_NOTE_WEIGHT_BEATS)
        prev_t = t
        if kind == 1:
            alive.add(i)
            heapq.heappush(heap, (notes[i].pitch, i))
        else:
            alive.discard(i)
    # Final bass: lowest pitch among notes starting at the last onset time.
    last_start = max(n.start for n in notes)
    final = min((n.pitch for n in notes if abs(n.start - last_start) < 1e-6), default=None)
    # Prefer the lowest note sounding at the very end (covers held bass under a final melody note).
    end_t = max(n.end for n in notes)
    sounding_end = [n.pitch for n in notes if n.end >= end_t - 0.5 and n.start < end_t]
    if sounding_end:
        final = min(sounding_end)
    return h, (None if final is None else final % 12)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def profile_scores(hist: np.ndarray) -> dict[tuple[int, str], float]:
    return {(r, m): _corr(hist, np.roll(KK_MAJOR if m == "major" else KK_MINOR, r))
            for r in range(12) for m in ("major", "minor")}


def _parse_mido_key(key: str) -> tuple[int, str]:
    minor = key.endswith("m")
    pc = parse_pitch_class(key[:-1] if minor else key)
    return pc, "minor" if minor else "major"


def _refine_mode(hist: np.ndarray, root: int, family: str) -> tuple[str, dict]:
    tot = hist.sum()
    if tot <= 0:
        return family, {}
    h = hist / tot
    deg = lambda i: float(h[(root + i) % 12])  # noqa: E731
    info = {}
    if family == "major":
        if deg(10) > 1.5 * deg(11) and deg(10) > 0.04:
            info["reason"] = "flat seventh dominates leading tone"
            return "mixolydian", info
        if deg(6) > 1.5 * deg(5) and deg(6) > 0.04:
            info["reason"] = "raised fourth dominates natural fourth"
            return "lydian", info
        return "major", info
    if deg(9) > 1.5 * deg(8) and deg(9) > 0.04:
        info["reason"] = "natural sixth dominates flat sixth"
        return "dorian", info
    if deg(1) > 1.5 * deg(2) and deg(1) > 0.04:
        info["reason"] = "flat second dominates natural second"
        return "phrygian", info
    return "minor", info


# ----------------------------------------------------------------------------- detection


def detect_root(
    source: Score | list[Note],
    key_signatures: list[KeySignature] | None = None,
    override: str | int | None = None,
    ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD,
) -> RootAnalysis:
    """Detect the global tonal root of a score (or a list of pitched notes).

    `override` may be a pitch-class int, a root name ('D', 'Bb') or a key ('D minor').
    """
    if isinstance(source, Score):
        notes = [n for t in source.tracks if not t.is_drum for n in t.notes]
        key_signatures = source.key_signatures if key_signatures is None else key_signatures
    else:
        notes = list(source)
    key_signatures = key_signatures or []

    hist = pitch_class_histogram(notes)
    bass_h, final_bass = bass_histogram(notes)
    evidence = {
        "pitch_class_histogram": [round(float(x), 4) for x in hist],
        "bass_histogram": [round(float(x), 4) for x in bass_h],
        "final_bass_pitch_class": final_bass,
        "n_pitched_notes": len(notes),
    }

    if hist.sum() <= 0 or np.std(hist / max(hist.sum(), 1e-12)) < 1e-9:
        undetermined = RootAnalysis(
            root_pitch_class=None, root_name=None, mode=None, mode_family=None, root_confidence=0.0,
            root_detection_method="undetermined", ambiguous=True, requires_override=True,
            top_candidates=[], evidence={**evidence, "reason": "no pitched notes or flat pitch-class distribution"},
        )
        return _apply_override(undetermined, override, {}) if override is not None else undetermined

    s_all = profile_scores(hist)
    s_bass = profile_scores(bass_h) if bass_h.sum() > 0 else None
    combined: dict[tuple[int, str], float] = {}
    for k, v in s_all.items():
        if s_bass is not None:
            fin = 1.0 if final_bass is not None and final_bass == k[0] else 0.0
            combined[k] = W_ALL * v + W_BASS * s_bass[k] + W_FINAL * fin
        else:
            combined[k] = v
    ranked = sorted(combined.items(), key=lambda kv: -kv[1])
    best_key, best_score = ranked[0]

    # --- key-signature evidence (accepted only if it agrees with the content)
    ks_label, ks_agrees, method = None, None, "pitch_profile"
    winner = best_key
    if key_signatures:
        ks_label = key_signatures[0].key
        try:
            ks_pc, ks_mode = _parse_mido_key(ks_label)
            if ks_mode == "major":
                consistent = [(ks_pc, "major"), ((ks_pc + 9) % 12, "minor")]
            else:
                consistent = [(ks_pc, "minor"), ((ks_pc + 3) % 12, "major")]
            # C major is the default in many sequencers, so demand stronger agreement.
            tolerance = 0.03 if (ks_pc, ks_mode) == (0, "major") else 0.08
            cbest = max(consistent, key=lambda k: combined[k])
            declared = consistent[0]
            if combined[declared] >= combined[cbest] - 0.05:
                cbest = declared
            ks_agrees = combined[cbest] >= best_score - tolerance
            if ks_agrees:
                winner, method = cbest, "key_signature+pitch_profile"
            else:
                method = "pitch_profile (key signature rejected)"
        except ValueError:
            ks_agrees = False

    root, family = winner
    win_score = combined[winner]
    other_root_best = max(v for (r, m), v in combined.items() if r != root)
    margin = win_score - other_root_best
    fit = float(np.clip((s_all[winner] - 0.35) / 0.45, 0.0, 1.0))
    sep = float(np.clip(margin / 0.10, 0.0, 1.0))
    confidence = 0.4 * fit + 0.6 * sep
    if ks_agrees:
        confidence = min(1.0, confidence + 0.15)
    tie = margin < 1e-6
    ambiguous = tie or confidence < ambiguity_threshold

    mode, mode_info = _refine_mode(hist, root, family)
    evidence.update({
        "profile_fit": round(s_all[winner], 4),
        "bass_fit": None if s_bass is None else round(s_bass[winner], 4),
        "margin_over_next_root": round(margin, 4),
        "mode_refinement": mode_info,
    })
    top = []
    seen = set()
    for (r, m), v in ranked:
        if (r, m) in seen:
            continue
        seen.add((r, m))
        top.append({"root_pitch_class": r, "root_name": PITCH_NAMES[r], "mode": m, "score": round(v, 4)})
        if len(top) >= 5:
            break

    result = RootAnalysis(
        root_pitch_class=int(root), root_name=PITCH_NAMES[root], mode=mode, mode_family=family,
        root_confidence=round(confidence, 4),
        root_detection_method=method + ("+bass" if s_bass is not None else ""),
        ambiguous=bool(ambiguous), requires_override=bool(tie),
        top_candidates=top if ambiguous else top[:3],
        key_signature=ks_label, key_signature_agrees=ks_agrees, evidence=evidence,
    )
    if override is not None:
        return _apply_override(result, override, combined)
    return result


def _apply_override(result: RootAnalysis, override: str | int, combined: dict) -> RootAnalysis:
    if isinstance(override, int):
        pc, mode = override % 12, None
    else:
        pc, mode = parse_key(str(override))
    if mode is None:
        if combined:
            mode = "major" if combined[(pc, "major")] >= combined[(pc, "minor")] else "minor"
        else:
            mode = result.mode or "major"
    mode = normalize_mode(mode)
    return RootAnalysis(
        root_pitch_class=pc, root_name=PITCH_NAMES[pc], mode=mode, mode_family=MODE_FAMILY[mode],
        root_confidence=1.0, root_detection_method="manual_override", ambiguous=False,
        requires_override=False, top_candidates=result.top_candidates,
        key_signature=result.key_signature, key_signature_agrees=result.key_signature_agrees,
        evidence={**result.evidence, "detected_before_override": {
            "root_name": result.root_name, "mode": result.mode,
            "root_confidence": result.root_confidence, "ambiguous": result.ambiguous}},
    )


def detect_section_roots(score: Score, sections: list[dict], global_root: RootAnalysis,
                         min_confidence: float = 0.6) -> list[dict]:
    """Optional local-key analysis per section. Adds `section_root` when a confident local
    tonic differs from the global one (i.e. a modulation); the global root is kept."""
    out = []
    pitched = [n for t in score.tracks if not t.is_drum for n in t.notes]
    for sec in sections:
        s, e = sec["start_beat"], sec["end_beat"]
        notes = [n for n in pitched if s <= n.start < e]
        entry = dict(sec)
        if len(notes) >= 8:
            local = detect_root(notes, key_signatures=[])
            if (local.root_pitch_class is not None and not local.ambiguous
                    and local.root_confidence >= min_confidence
                    and local.root_pitch_class != global_root.root_pitch_class):
                entry["section_root"] = {"root_name": local.root_name, "root_pitch_class": local.root_pitch_class,
                                         "mode": local.mode, "root_confidence": local.root_confidence}
        out.append(entry)
    return out
