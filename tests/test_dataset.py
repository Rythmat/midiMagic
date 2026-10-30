"""XMIDI filename parsing and dataset manifest creation."""
import json

import numpy as np
import pytest

from midicomposer.dataset.filenames import build_vocab, discover_midi_files, normalize_label, parse_filename
from midicomposer.dataset.prepare import PrepareConfig, assign_split, load_manifest, prepare_dataset
from midicomposer.features import N_FEATURES


@pytest.mark.parametrize("name,emotion,genre,status", [
    ("XMIDI_exciting_rock_OS6BNCSH.midi", "exciting", "rock", "ok"),
    ("XMIDI_Romantic_Country_TY67UI89.MID", "romantic", "country", "ok"),
    ("xmidi_HAPPY_Pop_AB12CD34.mid", "happy", "pop", "ok"),
    ("XMIDI_quiet-calm_new age_TG55YH66.midi", "quiet_calm", "new_age", "ok"),
])
def test_parse_simple_names(name, emotion, genre, status):
    p = parse_filename(name)
    assert (p.emotion, p.genre, p.status) == (emotion, genre, status)
    assert p.id_length_ok


def test_raw_labels_preserved():
    p = parse_filename("XMIDI_Romantic_Country_TY67UI89.MID")
    assert (p.raw_emotion, p.raw_genre, p.file_id) == ("Romantic", "Country", "TY67UI89")


def test_normalize_label():
    assert normalize_label("Hip-Hop") == "hip_hop"
    assert normalize_label("  New   Age ") == "new_age"
    assert normalize_label("Rock_n_Roll") == "rock_n_roll"


def test_multiword_labels_resolved_with_vocabulary():
    names = ["XMIDI_happy_pop_AAAAAAAA.midi", "XMIDI_sad_rock_BBBBBBBB.midi",
             "XMIDI_very_happy_hip_hop_CCCCCCCC.midi", "XMIDI_angry_rock_n_roll_DDDDDDDD.midi"]
    vocab = build_vocab(names)
    a = parse_filename(names[2], vocab)
    assert (a.emotion, a.genre, a.status) == ("very_happy", "hip_hop", "resolved_by_vocabulary")
    b = parse_filename(names[3], vocab)
    assert (b.emotion, b.genre) == ("angry", "rock_n_roll")


def test_unresolvable_split_is_flagged():
    p = parse_filename("XMIDI_foo_bar_baz_EEEEEEEE.midi", build_vocab([]))
    assert p.status == "ambiguous" and p.notes


def test_non_xmidi_name_is_unparsed():
    p = parse_filename("not_an_xmidi_name.mid")
    assert p.status == "unparsed" and p.emotion is None


def test_id_length_check():
    p = parse_filename("XMIDI_happy_pop_ABC123.midi")
    assert p.status == "ok" and not p.id_length_ok


def test_recursive_discovery(fixture_dataset):
    files = discover_midi_files(fixture_dataset)
    assert any("nested" in f.parts for f in files)
    assert any(f.suffix == ".MID" for f in files)


def test_split_is_deterministic():
    assert assign_split("abc", "s", 0.8, 0.1) == assign_split("abc", "s", 0.8, 0.1)
    splits = [assign_split(str(i), "s", 0.8, 0.1) for i in range(2000)]
    assert 0.75 < splits.count("train") / 2000 < 0.85


@pytest.fixture(scope="module")
def prepared(fixture_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("prep")
    summary = prepare_dataset(PrepareConfig(xmidi_root=str(fixture_dataset), out_dir=str(out), workers=1))
    return out, summary, {r["filename"]: r for r in load_manifest(out)}


def test_manifest_fields(prepared):
    out, summary, recs = prepared
    r = recs["XMIDI_happy_pop_AB12CD34.midi"]
    for k in ("filename", "file_id", "raw_emotion", "raw_genre", "emotion", "genre", "duration_seconds",
              "tempo_bpm", "meter", "root_name", "root_pitch_class", "root_reference_midi", "mode",
              "root_confidence", "tracks", "content_sha256", "split"):
        assert k in r, k
    assert (r["root_name"], r["mode"], r["root_reference_midi"]) == ("D", "major", 62)
    assert r["tracks"][3]["is_drum"] and r["tracks"][3]["role"] == "drums"
    assert (out / "manifest.csv").exists() and (out / "dataset_report.md").exists()


def test_corrupted_and_empty_files_recorded(prepared):
    _, _, recs = prepared
    assert recs["XMIDI_sad_jazz_CORRUPT1.midi"]["status"] == "corrupted"
    assert recs["XMIDI_happy_rock_EMPTY001.midi"]["status"] == "empty_file"
    assert not recs["XMIDI_sad_jazz_CORRUPT1.midi"]["usable_for_training"]


def test_duplicates_share_split_and_are_excluded(prepared):
    _, _, recs = prepared
    orig = recs["XMIDI_happy_pop_AB12CD34.midi"]
    exact = recs["XMIDI_happy_pop_DUP00001.midi"]
    trans = recs["XMIDI_happy_pop_TRN00001.midi"]
    assert exact["content_sha256"] == orig["content_sha256"]
    assert exact["is_duplicate"] and exact["duplicate_kind"] == "content_sha256"
    assert trans["is_duplicate"] and trans["duplicate_kind"] == "transposition_hash"
    assert orig["split"] == exact["split"] == trans["split"]
    assert orig["usable_for_training"] and not exact["usable_for_training"]


def test_problem_roots_flagged(prepared):
    _, _, recs = prepared
    drums = recs["XMIDI_exciting_pop_DRUMS001.midi"]
    assert drums["root_pitch_class"] is None and drums["root_requires_override"]
    assert not drums["usable_for_training"]
    wt = recs["XMIDI_fear_jazz_WHOLE001.midi"]
    assert wt["root_ambiguous"] and len(wt["root_top_candidates"]) >= 3


def test_label_report(prepared):
    out, _, _ = prepared
    vocab = json.loads((out / "label_vocab.json").read_text())
    genres = {g["label"] for g in vocab["genres"]}
    assert {"pop", "rock", "jazz", "classical", "hip_hop", "rock_n_roll", "new_age"} <= genres
    emotions = {e["label"]: e for e in vocab["emotions"]}
    assert emotions["romantic"]["raw_spellings"] == {"Romantic": 1}
    rep = json.loads((out / "dataset_report.json").read_text())
    assert rep["summary"]["files_discovered"] == 23


def test_features_saved(prepared):
    out, summary, _ = prepared
    f = np.load(out / "features.npy")
    assert f.shape == (summary["n_usable"], N_FEATURES)


def test_max_files_dev_mode(fixture_dataset, tmp_path):
    s = prepare_dataset(PrepareConfig(xmidi_root=str(fixture_dataset), out_dir=str(tmp_path), max_files=5, workers=1))
    assert s["n_processed"] == 5 and s["n_discovered"] == 23
