"""Planner, rule-based generation, pipeline (both modes), regeneration, roles and analysis."""
import json

import pytest

from build_fixtures import drums_only, tonal_piece, whole_tone
from midicomposer.analyzer import analyze_score
from midicomposer.generation.pipeline import (GenerationError, GenerationRequest, regenerate_section,
                                              regenerate_track, run_generation, set_track_lock)
from midicomposer.midi_io import Score, load_midi, score_to_bytes
from midicomposer.planner import fit_form_to_length, parse_form, plan_from_metadata
from midicomposer.roles import detect_roles, expand_generated_roles
from midicomposer.root import detect_root
from midicomposer.styles import GENRES
from midicomposer.theory import parse_key, roman_to_chord

NO_CLS = dict(classifier_path=None)   # tests must not depend on locally trained checkpoints


def lead_only(root=2, mode="major", bars=8):
    s = tonal_piece(root, mode, bars=bars)
    return Score(tracks=[s.tracks[0]], tempos=s.tempos, time_signatures=s.time_signatures)


# --------------------------------------------------------------------------- theory / planner

def test_roman_numerals():
    assert roman_to_chord("V7", "minor") == (7, "7")
    assert roman_to_chord("bVII", "major") == (10, "maj")
    assert roman_to_chord("iiø7", "minor") == (2, "m7b5")
    assert roman_to_chord("vi", "major") == (9, "min")


def test_parse_key():
    assert parse_key("Bb minor") == (10, "minor")
    assert parse_key("F#m") == (6, "minor")
    assert parse_key("D") == (2, None)
    with pytest.raises(ValueError):
        parse_key("H major")


@pytest.mark.parametrize("bars", [4, 8, 16, 24, 32, 50, 64])
def test_form_fits_requested_length(bars):
    for g in GENRES.values():
        for tpl in g.forms:
            secs = fit_form_to_length(list(tpl), bars)
            assert sum(n for _, n in secs) == bars


def test_forms_are_genre_aware():
    import random
    jazz, _ = parse_form(None, GENRES["jazz"], random.Random(0))
    classical, _ = parse_form(None, GENRES["classical"], random.Random(0))
    assert "chorus" not in [n for n, _ in jazz] and "verse" not in [n for n, _ in classical]
    custom, name = parse_form("intro:4, verse, chorus, outro:2", GENRES["pop"], random.Random(0))
    assert custom == [("intro", 4), ("verse", 8), ("chorus", 8), ("outro", 2)]


def test_plan_from_metadata_basics():
    p = plan_from_metadata(genre="pop", emotion="sad", root_pc=2, mode=None, tempo=None, n_bars=32,
                           roles=["lead_melody", "chord_bed", "bass_line", "drums"], seed=1)
    assert p.mode == "minor"                      # sad -> minor when no mode is given
    assert p.n_bars == 32 and sum(s.n_bars for s in p.sections) == 32
    assert len(p.chords) == 32 and p.chords[-1].root_rel == 0     # ends on the tonic
    assert p.ref == 62
    for role, (lo, hi) in p.registers.items():
        assert hi - lo >= 12


def test_unknown_labels_map_with_warning():
    p = plan_from_metadata(genre="Synthwave", emotion="Nostalgic", root_pc=0, mode="major", tempo=100, n_bars=8,
                           roles=["chord_bed"], seed=0)
    assert p.genre_profile == "electronic"
    assert any("Nostalgic" in w for w in p.warnings)


def test_expand_full_composition_complements_anchor():
    roles = expand_generated_roles(["full_composition"], ["lead"])
    assert "lead_melody" not in roles and "counterline" in roles and "bass_line" in roles
    assert expand_generated_roles(["full_composition"], ["bass"]).count("bass_line") == 0


# --------------------------------------------------------------------------- roles / analysis

def test_role_detection(tonal_piece):
    roles = {k: v["role"] for k, v in detect_roles(tonal_piece(0, "major")).items()}
    assert roles == {0: "lead", 1: "chordal", 2: "bass", 3: "drums"}


def test_analysis_fields():
    a = analyze_score(tonal_piece(9, "minor", bars=16))
    assert a["root"]["root_name"] == "A" and a["root"]["mode"] == "minor"
    assert a["meter"] == "4/4" and a["tempo_bpm"] == 120 and a["n_bars"] == 16
    assert len(a["chords"]) >= 16 and a["chords"][0]["name"] in ("Amin", "Amin7")
    assert a["sections"] and len(a["energy_curve"]) == 16
    assert len(a["rhythm"]["onset_profile_16th"]) == 16


def test_analysis_flags_ambiguity():
    a = analyze_score(whole_tone())
    assert a["root"]["ambiguous"] and any("ambiguous" in w for w in a["warnings"])


# --------------------------------------------------------------------------- metadata mode

@pytest.mark.parametrize("genre,emotion,key,meter", [
    ("pop", "happy", "D", "4/4"), ("jazz", "lazy", "Bb", "4/4"), ("classical", "sad", "A minor", "3/4"),
    ("rock", "angry", "E", "4/4"), ("traditional", "warm", "G dorian", "6/8")])
def test_metadata_generation_valid(genre, emotion, key, meter):
    state = run_generation(GenerationRequest(genre=genre, emotion=emotion, key=key, meter=meter, bars=16,
                                             n_variations=2, seed=4, **NO_CLS))
    best = state.view()["candidates"][0]
    assert best["validation"]["passed"], best["validation"]["issues"]
    midi = load_midi(state.midi_bytes())
    assert midi.initial_time_signature.label == meter
    assert detect_root(midi).root_name == ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][parse_key(key)[0]]
    roles = {t["role"] for t in best["tracks"]}
    assert {"lead_melody", "bass_line"} & roles or {"lead_melody", "walking_bass"} & roles
    drums = [t for t in midi.tracks if t.is_drum]
    assert all(t.channel == 9 for t in drums)


def test_generation_is_deterministic():
    req = dict(genre="pop", emotion="happy", key="F", bars=16, n_variations=2, seed=11, **NO_CLS)
    a = run_generation(GenerationRequest(**req))
    b = run_generation(GenerationRequest(**req))
    assert a.midi_bytes() == b.midi_bytes()


def test_variations_differ_and_are_ranked():
    state = run_generation(GenerationRequest(genre="pop", key="C", bars=16, n_variations=3, seed=2, **NO_CLS))
    candidates = state.view()["candidates"]
    totals = [c["scores"]["total"] for c in candidates]
    assert totals == sorted(totals, reverse=True)
    assert len({state.midi_bytes(i) for i in range(3)}) == 3
    for key in ("quality", "genre", "emotion", "novelty"):
        assert 0 <= candidates[0]["scores"][key] <= 1


def test_requested_roles_present():
    roles = ["lead_melody", "chord_bed", "arpeggio", "bass_line", "counterline", "harmony", "octave_doubles",
             "pad", "high_texture", "low_texture", "drums"]
    state = run_generation(GenerationRequest(genre="pop", key="E", bars=8, generated_roles=roles, n_variations=1, **NO_CLS))
    c = state.view()["candidates"][0]
    assert c["validation"]["passed"], c["validation"]["issues"]
    got = {t["role"]: t["n_notes"] for t in c["tracks"]}
    assert all(got.get(role, 0) > 0 for role in roles), got


def test_manifest_contents():
    state = run_generation(GenerationRequest(genre="jazz", emotion="warm", key="Eb", bars=8, n_variations=2, seed=9, **NO_CLS))
    m = json.loads(json.dumps(state.manifest))       # plain JSON, as served by the API
    assert m["input_conditions"]["genre"] == "jazz" and m["input_conditions"]["seed"] == 9
    assert m["root"]["output_root_name"] == "D#" and m["root"]["output_reference_midi"] == 63
    assert m["model"]["engine_requested"] == "rule" and "package_version" in m["model"]
    assert [c["seed"] for c in m["candidates"]] and all(c["output_roles"] for c in m["candidates"])


def test_invalid_requests_rejected():
    with pytest.raises(GenerationError):
        run_generation(GenerationRequest(n_variations=0))
    with pytest.raises(GenerationError):
        run_generation(GenerationRequest(generated_roles=["kazoo_solo"]))
    with pytest.raises(GenerationError):
        run_generation(GenerationRequest(meter="7/3"))
    with pytest.raises(GenerationError):
        run_generation(GenerationRequest(mode="midi"))


# --------------------------------------------------------------------------- MIDI mode

def _anchor_notes(track):
    return [(n.start, n.duration, n.pitch, n.velocity) for n in track.sorted_notes()]


def test_melody_conditioned_accompaniment():
    anchor = lead_only(2)
    state = run_generation(GenerationRequest(mode="midi", input_role="lead", genre="pop", emotion="happy",
                                             generated_roles=["chord_bed", "bass_line", "counterline", "drums"],
                                             n_variations=2, seed=1, **NO_CLS),
                           input_bytes=score_to_bytes(anchor))
    r = state.view()
    best = r["candidates"][0]
    assert best["validation"]["passed"], best["validation"]["issues"]
    assert r["analysis"]["root"]["root_name"] == "D"
    out = load_midi(state.midi_bytes())
    assert _anchor_notes(out.tracks[0]) == _anchor_notes(anchor.tracks[0])    # anchor unchanged
    assert out.initial_time_signature.label == "4/4" and abs(out.initial_bpm - 120) < 1e-6
    assert best["scores"]["components"]["harmonic_compatibility"] > 0.7
    assert best["scores"]["components"]["key_consistency"] > 0.85
    assert detect_root(out).root_name == "D"


def test_bass_conditioned_harmony_with_target_key():
    s = tonal_piece(2, "major")
    bass = Score(tracks=[s.tracks[2]], tempos=s.tempos, time_signatures=s.time_signatures)
    state = run_generation(GenerationRequest(mode="midi", input_role="bass", target_key="A",
                                             generated_roles=["chord_bed", "counterline"], n_variations=1, **NO_CLS),
                           input_bytes=score_to_bytes(bass))
    best = state.view()["candidates"][0]
    assert best["validation"]["passed"], best["validation"]["issues"]
    out = load_midi(state.midi_bytes())
    src = [n.pitch for n in bass.tracks[0].sorted_notes()]
    dst = [n.pitch for n in out.tracks[0].sorted_notes()]
    shift = state.manifest["root"]["transposition_semitones"]
    assert shift == -5 and [d - p for p, d in zip(src, dst)] == [shift] * len(src)  # D -> A, nearest register
    assert detect_root(out).root_name == "A"


def test_full_mix_input_and_percussion_preserved():
    full = tonal_piece(7, "major")
    state = run_generation(GenerationRequest(mode="midi", input_role="full_mix", generated_roles=["full_composition"],
                                             target_key="Bb", n_variations=1, **NO_CLS),
                           input_bytes=score_to_bytes(full))
    out = load_midi(state.midi_bytes())
    drum_in = _anchor_notes(full.tracks[3])
    assert any(_anchor_notes(t) == drum_in for t in out.tracks if t.is_drum)   # drums never transposed


def test_ambiguous_and_undetermined_inputs():
    with pytest.raises(GenerationError, match="root"):
        run_generation(GenerationRequest(mode="midi", **NO_CLS), input_bytes=score_to_bytes(drums_only()))
    r = run_generation(GenerationRequest(mode="midi", root_override="E minor", n_variations=1,
                                         generated_roles=["chord_bed"], **NO_CLS),
                       input_bytes=score_to_bytes(whole_tone())).view()
    assert r["analysis"]["root"]["root_detection_method"] == "manual_override"
    assert r["plan"]["root_name"] == "E"


def test_corrupted_input_rejected():
    for data in (b"", b"not a midi file", b"MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0MTrk\xff\xff"):
        with pytest.raises(GenerationError):
            run_generation(GenerationRequest(mode="midi", **NO_CLS), input_bytes=data or None)


def test_longer_than_anchor_continues():
    state = run_generation(GenerationRequest(mode="midi", input_role="lead", bars=12, generated_roles=["chord_bed"],
                                             n_variations=1, **NO_CLS), input_bytes=score_to_bytes(lead_only()))
    out = load_midi(state.midi_bytes())
    assert max(n.end for n in out.tracks[1].notes) > 8 * 4      # generated part continues past bar 8
    assert any("anchor length" in w for w in state.warnings)


# --------------------------------------------------------------------------- regeneration

def test_regenerate_track_and_section():
    state = run_generation(GenerationRequest(mode="midi", input_role="lead", n_variations=1, seed=3,
                                             generated_roles=["chord_bed", "bass_line", "drums"], bars=16, **NO_CLS),
                           input_bytes=score_to_bytes(lead_only(bars=16)))
    cand = state.candidates[0]
    bass_id = next(t for t, role in cand.role_of_track.items() if role == "bass_line")
    chord_id = next(t for t, role in cand.role_of_track.items() if role == "chord_bed")
    notes_of = lambda st: {t.track_id: _anchor_notes(t) for t in st.candidates[0].score.tracks}  # noqa: E731
    before = notes_of(state)

    s2 = regenerate_track(state, 0, bass_id, seed=999)
    after = notes_of(s2)
    assert after[bass_id] != before[bass_id]
    assert after[chord_id] == before[chord_id] and after[0] == before[0]
    assert s2.parent_job_id == state.job_id and s2.candidates[0].validation["passed"]
    assert {t.track_id: _anchor_notes(t) for t in load_midi(s2.midi_bytes()).tracks}[1] != []   # MIDI rebuilt
    assert notes_of(state) == before                            # the parent job's result is untouched

    # Lock the chord track, then regenerate bars 5-8: only unlocked generated tracks change, only there.
    set_track_lock(state, 0, chord_id, True)
    s3 = regenerate_section(state, 0, start_bar=4, end_bar=8, seed=1234)
    after3 = notes_of(s3)
    assert after3[chord_id] == before[chord_id] and after3[0] == before[0]
    # Bars 1-4 (beats < 16) keep their notes (ones crossing into the section are only shortened) ...
    before_head = [n for n in before[bass_id] if n[0] + n[1] <= 16]
    assert [n for n in after3[bass_id] if n[0] + n[1] <= 16][:len(before_head)] == before_head
    # ... and bars 9+ (beats >= 32) are untouched.
    assert [n for n in after3[bass_id] if n[0] >= 32] == [n for n in before[bass_id] if n[0] >= 32]
    assert after3 != before

    with pytest.raises(GenerationError):
        regenerate_track(state, 0, 0)                           # anchor track is locked
    with pytest.raises(GenerationError):
        set_track_lock(state, 0, 0, False)
