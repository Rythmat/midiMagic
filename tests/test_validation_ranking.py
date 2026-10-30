"""MIDI validation, repair and candidate ranking."""
from build_fixtures import tonal_piece
from midicomposer.harmony import ChordSpan
from midicomposer.midi_io import Note, Score, Track, score_to_bytes
from midicomposer.planner import plan_from_metadata
from midicomposer.ranking import CandidateInfo, harmonic_compatibility, note_set_signature, score_candidate
from midicomposer.validation import check_note_pairing, repair_score, validate_score


def gen_track(tid, notes, program=0, role="lead", is_drum=False, channel=0):
    t = Track(tid, f"t{tid}", program, 9 if is_drum else channel, is_drum, notes, role=role, source="generated")
    return t


def test_valid_piece_passes():
    s = tonal_piece(0, "major")
    for t in s.tracks:
        t.source = "generated"
    rep = validate_score(s, mode="major", root_pc_for_key=0, expected_meter="4/4")
    assert rep.passed, rep.to_dict()


def test_detects_and_repairs_problems():
    notes = [Note(0.0, 1.0, 60, 90), Note(0.5, 1.0, 64, 90),        # overlap in a monophonic line
             Note(2.0, 0.0, 62, 90),                                  # zero duration
             Note(3.0, 1.0, 65, 90), Note(3.0, 1.0, 65, 80),          # duplicate
             Note(4.013, 1.0, 67, 200)]                               # off-grid onset, bad velocity
    s = Score(tracks=[gen_track(0, notes, program=73, role="lead"),
                      gen_track(1, [Note(0, 1, 20, 80), Note(1, 1, 22, 80)], program=33, role="bass"),
                      gen_track(2, [Note(0, 0.25, 36, 100)], role="drums", is_drum=True, channel=3)])
    s.tracks[2].channel = 3                                          # drum on the wrong channel
    rep = validate_score(s, mode="major", root_pc_for_key=0)
    codes = {i.code for i in rep.issues}
    assert {"monophonic_overlap", "invalid_duration", "duplicate_notes", "invalid_velocity", "drum_channel",
            "off_grid", "instrument_range"} <= codes
    repairs = repair_score(s)
    assert repairs
    rep2 = validate_score(s, mode="major", root_pc_for_key=0)
    assert rep2.passed, rep2.to_dict()
    assert s.tracks[2].channel == 9
    assert [n.pitch for n in s.tracks[1].notes] == [32, 34]          # bass moved up an octave, not clipped


def test_short_but_positive_input_notes_are_valid():
    grace = Track(0, "lead", 73, 0, False, [Note(0.0, 30 / 480, 72, 90), Note(0.0625, 1.0, 74, 90)], role="lead")
    assert validate_score(Score(tracks=[grace]), mode="major").passed


def test_locked_tracks_are_not_repaired():
    anchor = Track(0, "anchor", 0, 0, False, [Note(0.0, 1.0, 60, 90), Note(0.5, 1.0, 61, 90)], role="lead",
                   locked=True, source="input")
    s = Score(tracks=[anchor])
    repair_score(s, locked_ids={0})
    assert [(n.start, n.pitch) for n in s.tracks[0].notes] == [(0.0, 60), (0.5, 61)]


def test_anchor_modification_detected():
    s = tonal_piece(0, "major")
    ref = {0: sorted((round(n.start, 6), round(n.duration, 6), n.pitch, n.velocity) for n in s.tracks[0].notes)}
    assert validate_score(s, mode="major", anchor_reference=ref).passed
    s.tracks[0].notes[3].pitch += 1
    rep = validate_score(s, mode="major", anchor_reference=ref)
    assert "anchor_modified" in {i.code for i in rep.errors}


def test_missing_role_and_meter_mismatch():
    s = tonal_piece(0, "major")
    rep = validate_score(s, mode="major", requested_roles=["walking_bass"], role_of_track={0: "lead_melody"},
                         expected_meter="3/4")
    codes = {i.code for i in rep.errors}
    assert {"missing_role", "meter_mismatch"} <= codes


def test_off_key_warning():
    s = Score(tracks=[gen_track(0, [Note(i, 1, 61 + (i % 2) * 5, 80) for i in range(8)])])  # C#/F# in C major
    rep = validate_score(s, mode="major", root_pc_for_key=0)
    assert "off_key" in {i.code for i in rep.warnings}


def test_note_pairing_check_on_exported_bytes():
    assert check_note_pairing(score_to_bytes(tonal_piece(0, "major"))) == []


# --------------------------------------------------------------------------- ranking

def _plan():
    return plan_from_metadata(genre="pop", emotion="happy", root_pc=0, mode="major", tempo=110, n_bars=4,
                              roles=["counterline"], seed=0)


def test_harmonic_compatibility_prefers_consonance():
    plan = _plan()
    plan.chords = [ChordSpan(b * 4.0, b * 4.0 + 4, 0, "maj", b) for b in range(4)]
    ref = [Note(i * 1.0, 1.0, 72, 90) for i in range(16)]                  # C5 melody
    consonant = Track(1, "c", 0, 1, False, [Note(i * 1.0, 1.0, 64, 80) for i in range(16)])   # E4: tenth
    dissonant = Track(1, "d", 0, 1, False, [Note(i * 1.0, 1.0, 61, 80) for i in range(16)])   # C#4
    assert harmonic_compatibility([consonant], ref, plan) > 0.9 > 0.5 > harmonic_compatibility([dissonant], ref, plan)


def test_ranking_penalizes_invalid_candidates():
    plan = _plan()
    good = tonal_piece(0, "major", bars=4)
    for t in good.tracks:
        t.source = "generated"
    roles = {0: "lead_melody", 1: "chord_bed", 2: "bass_line", 3: "drums"}
    ok_rep = validate_score(good, mode="major", root_pc_for_key=0)
    bad = good.copy()
    bad.tracks[0].notes[2].pitch = 61
    bad_rep = validate_score(bad, mode="major", root_pc_for_key=0)
    bad_rep.add("synthetic_error", "error", "forced failure")
    ids = [0, 1, 2, 3]
    a = score_candidate(CandidateInfo(good, ids, [], roles, ok_rep, note_set_signature(good, ids)), plan, None, [])
    b = score_candidate(CandidateInfo(bad, ids, [], roles, bad_rep, note_set_signature(bad, ids)), plan, None, [])
    assert a["total"] > b["total"] and a["components"]["validity"] == 1.0 > b["components"]["validity"]
    for k in ("quality", "genre", "emotion", "novelty"):
        assert 0.0 <= a[k] <= 1.0
    assert a["genre_detail"]["genre_classifier"] is None      # no classifier given -> heuristic only
