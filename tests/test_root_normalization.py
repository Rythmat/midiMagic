"""Root extraction, root-relative normalization, denormalization, transposition, percussion."""
import pytest

from build_fixtures import drums_only, tonal_piece, whole_tone
from midicomposer.midi_io import Note, Score, Track, KeySignature, load_midi, score_to_bytes
from midicomposer.representation import (denormalize, fit_track_range, normalize_score,
                                         root_reference_midi, transpose_score)
from midicomposer.root import RootUndeterminedError, detect_root

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def single_track(pitches, program=0):
    notes = [Note(i * 1.0, 1.0, p, 90) for i, p in enumerate(pitches)]
    return Score(tracks=[Track(0, "t", program, 0, False, notes)])


# --------------------------------------------------------------------------- root detection

@pytest.mark.parametrize("root_pc", range(12))
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_detects_root_and_mode_for_all_keys(root_pc, mode):
    r = detect_root(tonal_piece(root_pc, mode))
    assert r.root_pitch_class == root_pc
    assert r.root_name == NAMES[root_pc]
    assert r.mode == mode
    assert not r.ambiguous
    assert r.root_confidence >= 0.5
    assert r.root_detection_method.startswith("pitch_profile")
    assert r.root_reference_midi == 60 + root_pc


def test_root_not_taken_from_first_or_lowest_note():
    # G major material whose first and lowest note is a D.
    s = tonal_piece(7, "major")
    s.tracks[0].notes.insert(0, Note(0.0, 0.25, 26, 40))
    r = detect_root(s)
    assert r.root_name == "G"


def test_key_signature_used_when_consistent():
    s = tonal_piece(2, "major")
    s.key_signatures = [KeySignature(0.0, "D")]
    r = detect_root(s)
    assert r.root_name == "D" and r.key_signature_agrees is True
    assert r.root_detection_method.startswith("key_signature")


def test_wrong_key_signature_is_rejected():
    s = tonal_piece(2, "major")
    s.key_signatures = [KeySignature(0.0, "Eb")]
    r = detect_root(s)
    assert r.root_name == "D"
    assert r.key_signature_agrees is False
    assert "rejected" in r.root_detection_method


def test_ambiguous_root_is_flagged_not_silently_c():
    r = detect_root(whole_tone())
    assert r.ambiguous is True
    assert r.root_confidence < 0.4
    assert len(r.top_candidates) >= 3
    # Several distinct roots tie: the result must say so rather than pretend it is C.
    assert len({c["root_pitch_class"] for c in r.top_candidates}) >= 3


def test_drums_only_root_is_undetermined():
    r = detect_root(drums_only())
    assert r.root_pitch_class is None and r.root_name is None
    assert r.requires_override and r.ambiguous
    assert r.root_detection_method == "undetermined"
    with pytest.raises(RootUndeterminedError):
        normalize_score(drums_only())


def test_manual_override():
    r = detect_root(whole_tone(), override="D minor")
    assert (r.root_pitch_class, r.mode, r.root_detection_method) == (2, "minor", "manual_override")
    assert r.evidence["detected_before_override"]["ambiguous"] is True
    piece = normalize_score(whole_tone(), root_override="F#")
    assert piece.root_name == "F#" and piece.root_reference_midi == 66


# --------------------------------------------------------------------------- normalization

def test_reference_is_octave_four():
    assert root_reference_midi(0) == 60
    assert root_reference_midi(2) == 62
    assert root_reference_midi(9) == 69


def test_c_root_examples():
    # C4 -> 0, D4 -> 2, B3 -> -1
    piece = normalize_score(single_track([60, 62, 59]), root_override="C")
    rel = [n.relative_pitch for n in piece.tracks[0].notes]
    assert rel == [0, 2, -1]
    b3 = piece.tracks[0].notes[2]
    assert (b3.relative_pitch_class, b3.relative_octave, b3.original_pitch) == (11, -1, 59)


def test_d_root_examples():
    # In D: D4 -> 0, F#4 -> 4, C4 -> -2
    piece = normalize_score(single_track([62, 66, 60]), root_override="D")
    assert [n.relative_pitch for n in piece.tracks[0].notes] == [0, 4, -2]
    assert piece.root_reference_midi == 62
    assert piece.condition_tokens()[0] == "<ROOT=D>"


def test_negative_relative_pitches_round_trip():
    pitches = [21, 35, 47, 61, 108]
    piece = normalize_score(single_track(pitches), root_override="D")
    assert [n.relative_pitch for n in piece.tracks[0].notes] == [p - 62 for p in pitches]
    out, issues = denormalize(piece)
    assert [n.pitch for n in out.tracks[0].notes] == pitches and not issues


@pytest.mark.parametrize("root_pc", [0, 2, 5, 9, 11])
def test_normalize_denormalize_is_identity(root_pc):
    s = tonal_piece(root_pc, "major")
    piece = normalize_score(s)
    out, issues = denormalize(piece)
    assert not issues
    for t_in, t_out in zip(s.tracks, out.tracks):
        a = [(n.start, n.duration, n.pitch, n.velocity) for n in t_in.sorted_notes()]
        b = [(n.start, n.duration, n.pitch, n.velocity) for n in t_out.sorted_notes()]
        assert a == b


def test_round_trip_through_midi_bytes():
    s = tonal_piece(4, "minor")
    reloaded = load_midi(score_to_bytes(s))
    piece = normalize_score(reloaded)
    out, _ = denormalize(piece)
    assert [n.pitch for n in out.tracks[0].sorted_notes()] == [n.pitch for n in s.tracks[0].sorted_notes()]


def test_transposition_preserves_relative_sequence():
    c = tonal_piece(0, "major")
    d = tonal_piece(2, "major")
    rel_c = [[n.relative_pitch for n in t.notes] for t in normalize_score(c).tracks]
    rel_d = [[n.relative_pitch for n in t.notes] for t in normalize_score(d).tracks]
    assert rel_c == rel_d


@pytest.mark.parametrize("target", ["D", "A", "B", "F#"])
def test_transpose_canonical_keeps_relative_pitches(target):
    s = tonal_piece(0, "major")
    t, info = transpose_score(s, target, register="canonical")
    assert info["octave_offset"] == 0
    a = normalize_score(s)
    b = normalize_score(t, root_override=target)
    assert [[n.relative_pitch for n in tr.notes] for tr in a.tracks] == \
           [[n.relative_pitch for n in tr.notes] for tr in b.tracks]
    # Rhythm and durations untouched.
    assert [(n.start, n.duration) for n in s.tracks[0].notes] == [(n.start, n.duration) for n in t.tracks[0].notes]


def test_transpose_nearest_register_preserves_intervals():
    s = tonal_piece(0, "major")
    t, info = transpose_score(s, "B", register="nearest")
    assert info["semitones"] == -1
    src = [n.pitch for n in s.tracks[0].notes]
    dst = [n.pitch for n in t.tracks[0].notes]
    assert [b - a for a, b in zip(src, dst)] == [-1] * len(src)
    assert detect_root(t).root_name == "B"


@pytest.mark.parametrize("src,dst", [(0, 11), (9, 2), (2, 9), (11, 0), (5, 11), (6, 0)])
def test_nearest_transposition_stays_within_tritone(src, dst):
    s = tonal_piece(src, "major")
    t, info = transpose_score(s, NAMES[dst], register="nearest")
    shift = t.tracks[0].notes[0].pitch - s.tracks[0].notes[0].pitch
    assert -6 <= shift <= 5 and shift == info["semitones"]
    assert (shift - (dst - src)) % 12 == 0
    assert detect_root(t).root_pitch_class == dst


def test_percussion_is_never_root_normalized():
    s = tonal_piece(7, "major")
    drum_before = [(n.start, n.pitch) for n in s.tracks[3].notes]
    piece = normalize_score(s)
    drum_track = piece.tracks[3]
    assert drum_track.is_drum
    assert all(n.is_drum and n.relative_pitch is None for n in drum_track.notes)
    assert [(n.onset, n.original_pitch) for n in drum_track.notes] == drum_before
    t, _ = transpose_score(s, "C#")
    assert [(n.start, n.pitch) for n in t.tracks[3].notes] == drum_before


def test_out_of_midi_range_is_octave_shifted_not_clipped():
    piece = normalize_score(single_track([120, 125]), root_override="C")
    out, issues = denormalize(piece, target_root="B")   # +11 semitones -> 131, 136 (> 127)
    pitches = [n.pitch for n in out.tracks[0].notes]
    assert pitches == [119, 124]
    assert len(issues) == 2 and all("octave" in i for i in issues)


def test_instrument_range_fit_moves_whole_track_by_octaves():
    s = single_track([20, 22, 24, 27], program=33)   # electric bass, range 28-67
    actions = fit_track_range(s.tracks[0])
    assert [n.pitch for n in s.tracks[0].notes] == [32, 34, 36, 39]
    assert actions and "octave" in actions[0]


def test_beat_bar_positions():
    s = tonal_piece(0, "major")
    piece = normalize_score(s)
    n = piece.tracks[0].notes[5]       # second bar, second beat
    assert (n.bar, n.beat, n.position, n.sub_beat) == (1, 1, 12, 0)
