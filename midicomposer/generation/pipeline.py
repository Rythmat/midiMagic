"""End-to-end generation: analysis -> plan -> candidates -> validate/repair -> rank -> MIDI + manifest.

Nothing here touches the disk. Callers get a GenerationState holding the MIDI bytes, manifest and
everything needed to regenerate tracks or sections; it lives only as long as they keep it.
"""
from __future__ import annotations

import copy
import hashlib
import random
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import __version__
from ..analyzer import analyze_score
from ..classifiers import load_scorer
from ..harmony import ChordSpan
from ..midi_io import MidiLoadError, Score, load_midi, score_to_bytes
from ..planner import CompositionPlan, plan_from_anchor, plan_from_metadata, track_role_for
from ..representation import (STEPS_PER_QUARTER, NormalizedPiece, NormNote, NormTrack, denormalize,
                              nearest_octave_offset, normalize_score)
from ..ranking import CandidateInfo, note_set_signature, score_candidate
from ..roles import GEN_TO_TRACK_ROLE, expand_generated_roles, normalize_input_role
from ..root import RootAnalysis
from ..theory import PITCH_NAMES, gm_name, normalize_mode, parse_key
from ..validation import note_signature, repair_score, validate_score
from .rule_based import GenContext, RelNote, generate_role, order_roles, skyline

DEFAULT_CLASSIFIER = "checkpoints/classifiers.pt"
DEFAULT_MODEL = "checkpoints/composer_dev.pt"
MAX_VARIATIONS = 8
MAX_BARS = 256


class GenerationError(ValueError):
    """User-facing generation error (bad input, undeterminable root, ...)."""


@dataclass
class GenerationRequest:
    mode: str = "metadata"                    # "metadata" | "midi"
    genre: str | None = "pop"
    emotion: str | None = "happy"
    key: str | None = "C"                     # root for metadata mode, e.g. "D" or "D minor"
    scale_mode: str | None = None
    tempo: float | None = None
    meter: str | None = "4/4"
    bars: int | None = None
    density: float = 0.5
    instrumentation: str | None = None
    form: str | None = None
    input_role: str | None = None
    generated_roles: list[str] = field(default_factory=lambda: ["full_composition"])
    target_key: str | None = None
    root_override: str | None = None
    n_variations: int = 3
    seed: int = 0
    engine: str = "rule"                      # rule | neural | hybrid
    track_roles: dict[int, str] | None = None
    locked_track_ids: list[int] | None = None
    classifier_path: str | None = DEFAULT_CLASSIFIER
    model_path: str | None = DEFAULT_MODEL

    def validate(self) -> None:
        if self.mode not in ("metadata", "midi"):
            raise GenerationError("mode must be 'metadata' or 'midi'")
        if self.engine not in ("rule", "neural", "hybrid"):
            raise GenerationError("engine must be rule, neural or hybrid")
        if not 1 <= int(self.n_variations) <= MAX_VARIATIONS:
            raise GenerationError(f"n_variations must be 1..{MAX_VARIATIONS}")
        if self.bars is not None and not 1 <= int(self.bars) <= MAX_BARS:
            raise GenerationError(f"bars must be 1..{MAX_BARS}")
        if self.tempo is not None and not 20 <= float(self.tempo) <= 300:
            raise GenerationError("tempo must be within 20-300 BPM")
        if not self.generated_roles:
            raise GenerationError("choose at least one generated role")
        if self.scale_mode:
            try:
                self.scale_mode = normalize_mode(self.scale_mode)
            except ValueError as e:
                raise GenerationError(str(e)) from e


@dataclass
class CandidateState:
    index: int
    seed: int
    score: Score
    rel_tracks: dict[int, list[RelNote]]
    role_of_track: dict[int, str]             # generated role for generated tracks, track role for anchors
    generated_ids: list[int]
    anchor_ids: list[int]
    engine_used: dict[str, str]
    scores: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    midi: bytes = b""
    plan: CompositionPlan | None = None       # each candidate has its own plan (progressions, motifs)


@dataclass
class GenerationState:
    job_id: str
    request: GenerationRequest
    plan: CompositionPlan
    analysis: dict | None
    anchor_rel: dict[int, list[RelNote]]
    anchor_meta: dict[int, dict]
    anchor_reference: dict[int, list[tuple]]
    anchor_shift: int
    anchor_role: str | None
    candidates: list[CandidateState]
    input_name: str | None = None
    input_sha256: str | None = None
    warnings: list[str] = field(default_factory=list)
    parent_job_id: str | None = None
    created_at: str = ""
    manifest: dict = field(default_factory=dict)

    def midi_bytes(self, candidate: int = 0) -> bytes:
        return self.candidates[candidate].midi

    def view(self) -> dict:
        """JSON-friendly summary for the API/UI (includes compact note data for a piano roll)."""
        cands = []
        for i, c in enumerate(self.candidates):
            cands.append({
                "index": i, "rank": i + 1, "seed": c.seed, "scores": c.scores,
                "validation": c.validation, "engine_by_role": c.engine_used,
                "tracks": [{"track_id": t.track_id, "name": t.name, "role": c.role_of_track.get(t.track_id, t.role),
                            "program": t.program, "instrument": gm_name(t.program, t.is_drum), "channel": t.channel,
                            "is_drum": t.is_drum, "source": t.source, "locked": t.locked, "n_notes": len(t.notes),
                            "notes": [[round(n.start, 3), round(n.duration, 3), n.pitch, n.velocity] for n in t.notes]}
                           for t in c.score.tracks],
            })
        plan = self.plan
        return {"job_id": self.job_id, "parent_job_id": self.parent_job_id, "status": "done",
                "analysis": self.analysis, "plan": plan.to_dict(), "candidates": cands, "best_index": 0,
                "warnings": self.warnings, "manifest": self.manifest,
                "tempo": plan.tempo, "meter": plan.meter.label, "n_bars": plan.n_bars,
                "total_beats": plan.grid().bar_start(plan.n_bars - 1) + plan.grid().bar_length(plan.n_bars - 1)}


# ----------------------------------------------------------------------------- engines


class _Engines:
    def __init__(self, req: GenerationRequest, warnings: list[str]):
        self.neural = None
        if req.engine in ("neural", "hybrid"):
            from .neural import load_neural_generator
            self.neural = load_neural_generator(req.model_path)
            if self.neural is None:
                warnings.append(f"Neural engine requested but no trained checkpoint at {req.model_path!r}; "
                                "using the rule-based generator instead.")
        self.mode = req.engine

    def generate(self, role: str, ctx: GenContext, variation: int) -> tuple[list[RelNote], str]:
        use_neural = self.neural is not None and (self.mode == "neural" or (self.mode == "hybrid" and variation % 2 == 1))
        if use_neural and self.neural.supports(role):
            try:
                notes = self.neural.generate(role, ctx)
                if notes:
                    return notes, f"neural:{self.neural.version}"
            except Exception as e:  # fall back but record why
                ctx.plan.warnings.append(f"neural generation failed for {role}: {e}; used rule-based fallback")
        return generate_role(role, ctx), "rule"


# ----------------------------------------------------------------------------- helpers


def _norm_track(track_id: int, name: str, program: int, channel: int, is_drum: bool, role: str | None,
                notes: list[RelNote], grid, source: str, locked: bool) -> NormTrack:
    nt = NormTrack(track_id, name, program, channel, is_drum, role, locked=locked, source=source)
    for n in sorted(notes, key=lambda x: (x.start, x.pitch)):
        bar, off = grid.locate(n.start)
        pos = int(round(off * STEPS_PER_QUARTER))
        unit = grid.beat_unit(bar)
        beat = int(off // unit + 1e-9)
        rel = None if is_drum else n.pitch
        nt.notes.append(NormNote(
            onset=n.start, duration=n.duration, bar=bar, beat=beat,
            sub_beat=int(round((off - beat * unit) * STEPS_PER_QUARTER)), position=pos, relative_pitch=rel,
            relative_pitch_class=None if rel is None else rel % 12, relative_octave=None if rel is None else rel // 12,
            original_pitch=n.pitch if is_drum else 0, velocity=int(n.velocity), track_id=track_id, program=program,
            role=role, is_drum=is_drum))
    return nt


def _assemble(state_plan: CompositionPlan, anchor_rel, anchor_meta, gen_rel: dict[int, list[RelNote]],
              gen_meta: dict[int, dict], octave_offset: int) -> tuple[Score, list[str]]:
    plan = state_plan
    grid = plan.grid()
    tracks = []
    for tid, notes in anchor_rel.items():
        m = anchor_meta[tid]
        tracks.append(_norm_track(tid, m["name"], m["program"], m["channel"], m["is_drum"], m["role"], notes, grid,
                                  "input", True))
    for tid, notes in gen_rel.items():
        m = gen_meta[tid]
        tracks.append(_norm_track(tid, m["name"], m["program"], 0, m["is_drum"], m["track_role"], notes, grid,
                                  "generated", m.get("locked", False)))
    piece = NormalizedPiece(
        root_name=PITCH_NAMES[plan.root_pc], root_pitch_class=plan.root_pc, root_reference_midi=60 + plan.root_pc,
        mode=plan.mode, root_confidence=1.0, root_detection_method="plan", ambiguous=False, tracks=tracks,
        tempos=plan.tempos, time_signatures=plan.time_signatures, key_signatures=[], ticks_per_beat=480)
    score, issues = denormalize(piece, target_root=plan.root_pc, octave_offset=octave_offset)
    return score, issues


def _gen_meta(roles: list[str], plan: CompositionPlan, first_id: int) -> dict[int, dict]:
    meta = {}
    for i, role in enumerate(roles):
        prog = plan.programs.get(role, 0)
        meta[first_id + i] = {"role": role, "track_role": track_role_for(role), "program": prog,
                              "is_drum": role == "drums",
                              "name": f"{role.replace('_', ' ').title()} ({gm_name(prog, role == 'drums')})"}
    return meta


def _ctx(plan: CompositionPlan, seed: int, state: "GenerationState", generated=None) -> GenContext:
    pitched = [n for tid, ns in state.anchor_rel.items() if not state.anchor_meta[tid]["is_drum"] for n in ns]
    tracks = [(m["role"], m["program"], m["is_drum"], state.anchor_rel[tid]) for tid, m in state.anchor_meta.items()]
    return GenContext(plan=plan, grid=plan.grid(), rng=random.Random(seed), anchor_notes=pitched,
                      anchor_line=getattr(state, "_anchor_line", []), anchor_role=state.anchor_role,
                      generated=dict(generated or {}), anchor_tracks=tracks)


def _repair_and_validate(state: GenerationState, cand: CandidateState) -> None:
    """Repair unlocked tracks once, then validate. Anchor/locked tracks are never repaired."""
    plan = cand.plan
    locked = set(cand.anchor_ids) | {t.track_id for t in cand.score.tracks if t.locked}
    repairs = repair_score(cand.score, locked_ids=locked)
    for t in cand.score.tracks:
        r = cand.role_of_track.get(t.track_id)
        if r is not None:
            t.role = GEN_TO_TRACK_ROLE.get(r, r)
    # Chords are tonic-relative; validation compares (pitch - root_pc) % 12 against them.
    report = validate_score(
        cand.score, mode=plan.mode, chords=plan.chords,
        requested_roles=[cand.role_of_track[tid] for tid in cand.generated_ids],
        anchor_reference=state.anchor_reference or None, anchor_shift=state.anchor_shift,
        expected_meter=plan.meter.label, role_of_track=cand.role_of_track, root_pc_for_key=plan.root_pc)
    report.repairs = repairs
    cand.validation = report.to_dict()
    cand._report = report  # type: ignore[attr-defined]
    cand._signature = note_set_signature(cand.score, cand.generated_ids)  # type: ignore[attr-defined]


def _score(state: GenerationState, cand: CandidateState, others: list[set], classifier) -> None:
    info = CandidateInfo(cand.score, cand.generated_ids, cand.anchor_ids, cand.role_of_track,
                         cand._report, cand._signature)  # type: ignore[attr-defined]
    anchor_sig = note_set_signature(cand.score, cand.anchor_ids) if cand.anchor_ids else None
    cand.scores = score_candidate(info, cand.plan, classifier, others, anchor_sig)


def _finalize_candidate(state: GenerationState, cand: CandidateState, others: list[set], classifier) -> None:
    _repair_and_validate(state, cand)
    _score(state, cand, others, classifier)


# ----------------------------------------------------------------------------- main entry


def run_generation(req: GenerationRequest, input_bytes: bytes | None = None, input_name: str | None = None,
                   job_id: str | None = None, progress=None) -> GenerationState:
    """`progress(message)` is called between steps; it may raise to abandon the generation."""
    t0 = time.time()
    req.validate()
    job_id = job_id or uuid.uuid4().hex[:12]
    warnings: list[str] = []
    say = progress or (lambda msg: None)

    analysis = None
    anchor_rel: dict[int, list[RelNote]] = {}
    anchor_meta: dict[int, dict] = {}
    anchor_reference: dict[int, list[tuple]] = {}
    anchor_shift = 0
    anchor_role = None
    anchor_line: list[RelNote] = []
    input_sha = None
    octave_offset = 0

    if req.mode == "midi":
        if not input_bytes:
            raise GenerationError("MIDI mode requires an uploaded MIDI file.")
        input_sha = hashlib.sha256(input_bytes).hexdigest()
        say("analyzing input MIDI")
        try:
            score = load_midi(input_bytes)
        except MidiLoadError as e:
            raise GenerationError(str(e)) from e
        if not any(t.notes for t in score.tracks):
            raise GenerationError("The uploaded MIDI contains no notes.")
        try:
            in_role = normalize_input_role(req.input_role) if req.input_role else None
        except ValueError as e:
            raise GenerationError(str(e)) from e
        analysis = analyze_score(score, root_override=req.root_override, input_role=in_role)
        root = RootAnalysis.from_dict(analysis["root"])
        if root.root_pitch_class is None:
            raise GenerationError("The tonal root of the input could not be determined (e.g. drums only). "
                                  "Choose a manual root override.")
        warnings += analysis["warnings"]
        if req.track_roles:
            for t in score.tracks:
                r = req.track_roles.get(t.track_id, req.track_roles.get(str(t.track_id)))  # type: ignore[arg-type]
                if r:
                    t.role = normalize_input_role(r)
        src_pc = root.root_pitch_class
        mode = req.scale_mode or root.mode or "major"
        if req.scale_mode and req.scale_mode != root.mode:
            warnings.append(f"Requested mode '{req.scale_mode}' differs from the input's detected mode "
                            f"'{root.mode}'; generated parts follow the requested mode.")
        target_pc = parse_key(req.target_key)[0] if req.target_key else src_pc
        octave_offset = nearest_octave_offset(src_pc, target_pc)
        anchor_shift = (target_pc - src_pc) + 12 * octave_offset
        piece = normalize_score(score, root=root)
        for nt, t in zip(piece.tracks, score.tracks):
            notes = [RelNote(n.onset, n.duration, n.original_pitch if nt.is_drum else n.relative_pitch, n.velocity)
                     for n in nt.notes]
            anchor_rel[t.track_id] = notes
            anchor_meta[t.track_id] = {"name": t.name, "program": t.program, "channel": t.channel,
                                       "is_drum": t.is_drum, "role": t.role}
            anchor_reference[t.track_id] = note_signature(t)
        pitched_anchor = [n for tid, ns in anchor_rel.items() if not anchor_meta[tid]["is_drum"] for n in ns]
        anchor_roles = [anchor_meta[tid]["role"] for tid in anchor_rel]
        lead_ids = [tid for tid in anchor_rel if anchor_meta[tid]["role"] == "lead"]
        anchor_line = skyline([n for tid in lead_ids for n in anchor_rel[tid]]) if lead_ids else skyline(pitched_anchor)
        pitched_roles = [anchor_meta[tid]["role"] for tid in anchor_rel if not anchor_meta[tid]["is_drum"]]
        anchor_role = in_role or (pitched_roles[0] if len(set(pitched_roles)) == 1 else "full_mix")
        try:
            roles = expand_generated_roles(req.generated_roles, anchor_roles + ([in_role] if in_role else []))
        except ValueError as e:
            raise GenerationError(str(e)) from e
        out_ref = 60 + target_pc + 12 * octave_offset
        abs_pitches = [n.pitch + out_ref for n in pitched_anchor]
        anchor_range = (float(np.percentile(abs_pitches, 10)), float(np.percentile(abs_pitches, 90))) if abs_pitches else None
        # The analysed chords describe the anchor's actual harmony, so they are kept even if the
        # user requests a different mode. Anchor relative pitches are relative to the source tonic;
        # denormalizing with the output reference gives original + anchor_shift (a uniform transposition).
        chords = [ChordSpan(c["start"], c["end"], c["root_rel"], c["quality"], c["bar"]) for c in analysis["chords"]]

        def make_plan(seed: int) -> CompositionPlan:
            return plan_from_anchor(
                analysis=analysis, anchor_chords=chords, tempos=score.tempos, time_signatures=score.time_signatures,
                grid=score.bar_grid(), root_pc=target_pc, mode=mode, genre=req.genre, emotion=req.emotion,
                n_bars=req.bars, density=req.density, instrumentation=req.instrumentation, roles=roles, seed=seed,
                anchor_range=anchor_range, anchor_role=anchor_role, octave_offset=octave_offset)
        plan = make_plan(req.seed)
    else:
        say("planning composition")
        try:
            root_pc, key_mode = parse_key(req.key or "C")
        except ValueError as e:
            raise GenerationError(str(e)) from e
        mode = req.scale_mode or key_mode
        from ..styles import resolve_genre, wants_drums
        try:
            roles = expand_generated_roles(req.generated_roles)
        except ValueError as e:
            raise GenerationError(str(e)) from e
        no_drums = (wants_drums(req.instrumentation) is False
                    or (not req.instrumentation and resolve_genre(req.genre)[0].name in ("classical", "ambient")))
        if no_drums and "drums" in roles and "drums" not in req.generated_roles:
            roles.remove("drums")

        def make_plan(seed: int) -> CompositionPlan:
            # Each variation gets its own plan (progressions, motifs) from its own seed.
            return plan_from_metadata(genre=req.genre, emotion=req.emotion, root_pc=root_pc, mode=mode,
                                      tempo=req.tempo, meter=req.meter or "4/4", n_bars=req.bars, density=req.density,
                                      instrumentation=req.instrumentation, form=req.form, roles=roles, seed=seed)
        try:
            plan = make_plan(req.seed)
        except ValueError as e:
            raise GenerationError(str(e)) from e
    warnings += plan.warnings

    state = GenerationState(job_id=job_id, request=req, plan=plan, analysis=analysis, anchor_rel=anchor_rel,
                            anchor_meta=anchor_meta, anchor_reference=anchor_reference, anchor_shift=anchor_shift,
                            anchor_role=anchor_role, candidates=[], input_name=input_name,
                            input_sha256=input_sha, warnings=warnings, created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    state._octave_offset = octave_offset  # type: ignore[attr-defined]
    state._anchor_line = anchor_line      # type: ignore[attr-defined]
    engines = _Engines(req, warnings)
    classifier = load_scorer(req.classifier_path)
    if req.classifier_path and classifier is None:
        warnings.append("No trained genre/emotion classifier found; genre/emotion scores use heuristics only.")

    first_id = (max(anchor_rel) + 1) if anchor_rel else 0
    meta = _gen_meta(order_roles(plan.roles), plan, first_id)
    for v in range(req.n_variations):
        seed = req.seed + 7919 * v
        plan_v = plan if v == 0 else make_plan(seed)
        ctx = _ctx(plan_v, seed, state)
        gen_rel: dict[int, list[RelNote]] = {}
        engine_used: dict[str, str] = {}
        for tid, m in meta.items():
            say(f"candidate {v + 1}/{req.n_variations}: {m['role'].replace('_', ' ')}")
            notes, used = engines.generate(m["role"], ctx, v)
            ctx.generated[m["role"]] = notes
            gen_rel[tid] = notes
            engine_used[m["role"]] = used
        score, issues = _assemble(plan_v, anchor_rel, anchor_meta, gen_rel, meta, octave_offset)
        warnings.extend(i for i in issues if i not in warnings)
        role_of_track = {tid: m["role"] for tid, m in meta.items()}
        role_of_track.update({tid: anchor_meta[tid]["role"] for tid in anchor_rel})
        cand = CandidateState(v, seed, score, gen_rel, role_of_track, list(meta), list(anchor_rel), engine_used,
                              plan=plan_v)
        _repair_and_validate(state, cand)
        state.candidates.append(cand)
    say("ranking candidates")
    # Novelty depends on all candidates, so scoring happens once every candidate exists.
    all_sigs = [c._signature for c in state.candidates]  # type: ignore[attr-defined]
    for i, c in enumerate(state.candidates):
        _score(state, c, [s for j, s in enumerate(all_sigs) if j != i], classifier)
    state.candidates.sort(key=lambda c: -c.scores["total"])
    state.plan = state.candidates[0].plan
    return _finish(state, classifier, engines, time.time() - t0)


def _finish(state: GenerationState, classifier, engines, seconds: float) -> GenerationState:
    for c in state.candidates:
        c.midi = score_to_bytes(c.score)
    state.manifest = build_manifest(state, classifier, engines, seconds)
    return state


def build_manifest(state: GenerationState, classifier, engines, seconds: float) -> dict:
    plan, req = state.plan, state.request
    model_info = {"package_version": __version__, "engine_requested": req.engine, "rule_based_version": "rules-1"}
    if engines is not None and getattr(engines, "neural", None) is not None:
        model_info["neural_checkpoint"] = engines.neural.describe()
    if classifier is not None:
        model_info["classifier"] = {"path": classifier.path, "test_metrics": classifier.metrics.get("test")}
    root_info = {"output_root_name": plan.root_name, "output_root_pitch_class": plan.root_pc,
                 "output_reference_midi": plan.ref, "mode": plan.mode}
    if state.analysis:
        r = state.analysis["root"]
        root_info.update({"input_root_name": r["root_name"], "input_root_pitch_class": r["root_pitch_class"],
                          "input_root_reference_midi": r["root_reference_midi"], "input_mode": r["mode"],
                          "input_root_confidence": r["root_confidence"],
                          "input_root_detection_method": r["root_detection_method"], "input_root_ambiguous": r["ambiguous"],
                          "transposition_semitones": state.anchor_shift})
    return {
        "job_id": state.job_id, "parent_job_id": state.parent_job_id, "created_at": state.created_at,
        "generation_seconds": round(seconds, 2),
        "input_conditions": asdict(req), "input_file": {"name": state.input_name, "sha256": state.input_sha256},
        "root": root_info, "model": model_info,
        "plan": plan.to_dict(),
        "candidates": [{
            "rank": i + 1, "seed": c.seed, "scores": c.scores,
            "engine_by_role": c.engine_used,
            "output_roles": [{"track_id": t.track_id, "name": t.name, "role": c.role_of_track.get(t.track_id, t.role),
                              "program": t.program, "instrument": gm_name(t.program, t.is_drum), "channel": t.channel,
                              "source": t.source, "locked": t.locked, "n_notes": len(t.notes)} for t in c.score.tracks],
            "validation": c.validation} for i, c in enumerate(state.candidates)],
        "warnings": state.warnings,
        "notes": ["Rule-based engine output is deterministic for a given request and seed.",
                  "Pitches were generated root-relative and denormalized with abs = rel + output_reference_midi."],
    }


# ----------------------------------------------------------------------------- regeneration


def _derived_state(parent: GenerationState, cand: CandidateState, job_id: str | None) -> GenerationState:
    """A new job sharing the parent's read-only input data, with its own copy of one candidate."""
    st = copy.copy(parent)
    st.job_id, st.parent_job_id = job_id or uuid.uuid4().hex[:12], parent.job_id
    st.candidates = [copy.deepcopy(cand)]
    st.warnings = list(parent.warnings)
    st.plan = st.candidates[0].plan
    st.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    return st


def _rebuild_candidate(st: GenerationState, cand: CandidateState, gen_rel: dict[int, list[RelNote]]) -> None:
    meta = {}
    old = {t.track_id: t for t in cand.score.tracks}
    for tid in cand.generated_ids:
        t = old[tid]
        role = cand.role_of_track[tid]
        meta[tid] = {"role": role, "track_role": track_role_for(role), "program": t.program, "is_drum": t.is_drum,
                     "name": t.name, "locked": t.locked}
    score, _ = _assemble(cand.plan, st.anchor_rel, st.anchor_meta, gen_rel, meta,
                         getattr(st, "_octave_offset", 0))
    cand.score = score
    cand.rel_tracks = gen_rel


def regenerate_track(parent: GenerationState, candidate_index: int, track_id: int, seed: int | None = None,
                     job_id: str | None = None, progress=None) -> GenerationState:
    t0 = time.time()
    say = progress or (lambda msg: None)
    if not 0 <= candidate_index < len(parent.candidates):
        raise GenerationError("candidate index out of range")
    cand = parent.candidates[candidate_index]
    if track_id not in cand.generated_ids:
        raise GenerationError("Only generated tracks can be regenerated; input/anchor tracks are locked.")
    if any(t.track_id == track_id and t.locked for t in cand.score.tracks):
        raise GenerationError("Track is locked; unlock it before regenerating.")
    st = _derived_state(parent, cand, job_id)
    c = st.candidates[0]
    seed = seed if seed is not None else c.seed + 104729
    role = c.role_of_track[track_id]
    say(f"regenerating {role.replace('_', ' ')}")
    others = {c.role_of_track[tid]: notes for tid, notes in c.rel_tracks.items() if tid != track_id}
    ctx = _ctx(c.plan, seed, st, others)
    engines = _Engines(st.request, st.warnings)
    notes, used = engines.generate(role, ctx, 1 if st.request.engine == "hybrid" else 0)
    gen_rel = dict(c.rel_tracks)
    gen_rel[track_id] = notes
    _rebuild_candidate(st, c, gen_rel)
    c.engine_used[role] = used
    c.seed = seed
    classifier = load_scorer(st.request.classifier_path)
    _finalize_candidate(st, c, [], classifier)
    st.warnings.append(f"Regenerated track {track_id} ({role}) with seed {seed} from job {parent.job_id}.")
    return _finish(st, classifier, engines, time.time() - t0)


def regenerate_section(parent: GenerationState, candidate_index: int, section_index: int | None = None,
                       start_bar: int | None = None, end_bar: int | None = None, track_ids: list[int] | None = None,
                       seed: int | None = None, job_id: str | None = None, progress=None) -> GenerationState:
    t0 = time.time()
    say = progress or (lambda msg: None)
    if not 0 <= candidate_index < len(parent.candidates):
        raise GenerationError("candidate index out of range")
    plan = parent.candidates[candidate_index].plan
    if section_index is not None:
        if not 0 <= section_index < len(plan.sections):
            raise GenerationError("section index out of range")
        sec = plan.sections[section_index]
        start_bar, end_bar = sec.start_bar, sec.end_bar
    if start_bar is None or end_bar is None or not 0 <= start_bar < end_bar <= plan.n_bars:
        raise GenerationError("give a valid section_index or start_bar/end_bar range")
    cand = parent.candidates[candidate_index]
    st = _derived_state(parent, cand, job_id)
    c = st.candidates[0]
    locked = {t.track_id for t in c.score.tracks if t.locked}
    targets = [tid for tid in (track_ids or c.generated_ids) if tid in c.generated_ids and tid not in locked]
    if not targets:
        raise GenerationError("No unlocked generated tracks to regenerate in this section.")
    seed = seed if seed is not None else c.seed + 15485863
    grid = c.plan.grid()
    a = grid.bar_start(start_bar)
    b = grid.bar_start(end_bar - 1) + grid.bar_length(end_bar - 1)
    engines = _Engines(st.request, st.warnings)
    gen_rel = {tid: list(ns) for tid, ns in c.rel_tracks.items()}
    for tid in sorted(targets, key=lambda t: order_roles(list(c.role_of_track.values())).index(c.role_of_track[t])):
        role = c.role_of_track[tid]
        say(f"regenerating {role.replace('_', ' ')}")
        others = {c.role_of_track[o]: ns for o, ns in gen_rel.items() if o != tid}
        ctx = _ctx(c.plan, seed + tid, st, others)
        fresh, used = engines.generate(role, ctx, 1 if st.request.engine == "hybrid" else 0)
        kept = []
        for n in gen_rel[tid]:
            if n.start >= a - 1e-9 and n.start < b - 1e-9:
                continue
            if n.start < a and n.end > a:
                n = RelNote(n.start, a - n.start, n.pitch, n.velocity)
            kept.append(n)
        new = [RelNote(n.start, min(n.duration, b - n.start), n.pitch, n.velocity) for n in fresh if a - 1e-9 <= n.start < b - 1e-9]
        gen_rel[tid] = sorted(kept + new, key=lambda n: (n.start, n.pitch))
        c.engine_used[role] = used
    _rebuild_candidate(st, c, gen_rel)
    classifier = load_scorer(st.request.classifier_path)
    _finalize_candidate(st, c, [], classifier)
    st.warnings.append(f"Regenerated bars {start_bar + 1}-{end_bar} of tracks {targets} with seed {seed} "
                       f"from job {parent.job_id}.")
    return _finish(st, classifier, engines, time.time() - t0)


def set_track_lock(state: GenerationState, candidate_index: int, track_id: int, locked: bool) -> None:
    cand = state.candidates[candidate_index]
    if track_id in cand.anchor_ids and not locked:
        raise GenerationError("Input/anchor tracks are always locked.")
    for t in cand.score.tracks:
        if t.track_id == track_id:
            t.locked = locked
            return
    raise GenerationError(f"track {track_id} not found")
