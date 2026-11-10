"""API integration tests: analysis, async generation, download validity, regeneration, errors."""
import time

import pytest
from fastapi.testclient import TestClient

from build_fixtures import tonal_piece
from midicomposer.api.app import Settings, create_app
from midicomposer.midi_io import Score, load_midi, score_to_bytes
from midicomposer.root import detect_root


@pytest.fixture(scope="module")
def client():
    st = Settings(classifier_checkpoint=None, model_checkpoint=None, label_vocab=None)
    with TestClient(create_app(st)) as c:
        yield c


def wait(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/generation/{job_id}")
        assert r.status_code == 200
        j = r.json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise AssertionError("job timed out")


def lead_bytes():
    s = tonal_piece(2, "major")
    return score_to_bytes(Score(tracks=[s.tracks[0]], tempos=s.tempos, time_signatures=s.time_signatures))


def test_health_and_options(client):
    assert client.get("/health").json()["status"] == "ok"
    o = client.get("/options").json()
    assert "pop" in o["genres"] and "happy" in o["emotions"] and o["keys"][2] == "D"
    assert "walking_bass" in o["generated_roles"] and "full_mix" in o["input_roles"]
    assert o["engines"]["rule"] is True and o["engines"]["neural"] is False


def test_web_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "MIDI Composer" in r.text
    assert client.get("/ui-state.js").status_code == 200


def test_analyze_midi(client):
    r = client.post("/analyze-midi", files={"file": ("lead.mid", lead_bytes(), "audio/midi")}, data={"input_role": "lead"})
    assert r.status_code == 200, r.text
    a = r.json()["analysis"]
    assert a["root"]["root_name"] == "D" and a["root"]["root_confidence"] > 0.5
    assert a["tempo_bpm"] == 120 and a["meter"] == "4/4" and a["tracks"][0]["assigned_role"] == "lead"


@pytest.mark.parametrize("payload,code", [(b"", 400), (b"hello world", 422),
                                          (b"MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0MTrk\xff\xff\xff\xff", 422)])
def test_analyze_rejects_bad_uploads(client, payload, code):
    r = client.post("/analyze-midi", files={"file": ("x.mid", payload, "audio/midi")})
    assert r.status_code == code
    assert r.json()["detail"]


def test_metadata_generation_and_download(client):
    r = client.post("/generate", json={"mode": "metadata", "genre": "jazz", "emotion": "warm", "key": "F",
                                       "bars": 8, "n_variations": 2, "seed": 3})
    assert r.status_code == 202
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "done", j
    res = j["result"]
    assert len(res["candidates"]) == 2 and res["candidates"][0]["validation"]["passed"]
    d = client.get(f"/generation/{res['job_id']}/download", params={"candidate": 1})
    assert d.status_code == 200 and d.headers["content-type"] == "audio/midi"
    midi = load_midi(d.content)                      # downloaded MIDI parses
    assert sum(len(t.notes) for t in midi.tracks) > 0
    assert detect_root(midi).root_name == "F"
    m = client.get(f"/generation/{res['job_id']}/manifest").json()
    assert m["input_conditions"]["genre"] == "jazz"
    assert 3 in [c["seed"] for c in m["candidates"]]


def test_midi_mode_generation_regeneration_and_locks(client):
    up = client.post("/analyze-midi", files={"file": ("lead.mid", lead_bytes(), "audio/midi")}).json()
    r = client.post("/generate", json={"mode": "midi", "upload_id": up["upload_id"], "input_role": "lead",
                                       "generated_roles": ["chord_bed", "bass_line", "drums"], "n_variations": 1,
                                       "target_key": "E"})
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "done", j
    res = j["result"]
    cand = res["candidates"][0]
    assert cand["tracks"][0]["source"] == "input" and cand["tracks"][0]["locked"]
    out = load_midi(client.get(f"/generation/{res['job_id']}/download").content)
    assert detect_root(out).root_name == "E"
    bass = next(t for t in cand["tracks"] if t["role"] == "bass_line")
    chord = next(t for t in cand["tracks"] if t["role"] == "chord_bed")

    # Input tracks cannot be regenerated or unlocked.
    assert client.post("/regenerate-track", json={"job_id": res["job_id"], "track_id": 0}).status_code == 422
    assert client.post(f"/generation/{res['job_id']}/lock", json={"track_id": 0, "locked": False}).status_code == 422

    rt = client.post("/regenerate-track", json={"job_id": res["job_id"], "track_id": bass["track_id"], "seed": 77})
    assert rt.status_code == 202
    j2 = wait(client, rt.json()["job_id"])
    assert j2["status"] == "done" and j2["result"]["parent_job_id"] == res["job_id"]

    assert client.post(f"/generation/{res['job_id']}/lock", json={"track_id": chord["track_id"], "locked": True}).status_code == 200
    rs = client.post("/regenerate-section", json={"job_id": res["job_id"], "section_index": 0, "seed": 5})
    j3 = wait(client, rs.json()["job_id"])
    assert j3["status"] == "done", j3
    new_chord = next(t for t in j3["result"]["candidates"][0]["tracks"] if t["track_id"] == chord["track_id"])
    assert new_chord["notes"] == chord["notes"] and new_chord["locked"]


def test_generation_errors(client):
    assert client.post("/generate", json={"mode": "midi"}).status_code == 422                   # no upload id
    assert client.post("/generate", json={"mode": "midi", "upload_id": "nope"}).status_code == 404
    assert client.post("/generate", json={"n_variations": 99}).status_code == 422
    r = client.post("/generate", json={"generated_roles": ["kazoo"]})
    assert r.status_code == 422 or wait(client, r.json()["job_id"])["status"] == "error"
    assert client.get("/generation/doesnotexist").status_code == 404
    assert client.get("/generation/doesnotexist/download").status_code == 404


def test_undetermined_root_job_reports_error(client):
    from build_fixtures import drums_only
    up = client.post("/analyze-midi", files={"file": ("d.mid", score_to_bytes(drums_only()), "audio/midi")}).json()
    assert up["analysis"]["root"]["root_pitch_class"] is None
    r = client.post("/generate", json={"mode": "midi", "upload_id": up["upload_id"], "generated_roles": ["chord_bed"]})
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "error" and "root" in j["error"].lower()
    r = client.post("/generate", json={"mode": "midi", "upload_id": up["upload_id"], "generated_roles": ["chord_bed"],
                                       "root_override": "G"})
    assert wait(client, r.json()["job_id"])["status"] == "done"


def test_prepare_dataset_is_off_by_default(client, fixture_dataset):
    r = client.post("/prepare-dataset", json={"xmidi_root": str(fixture_dataset), "max_files": 1})
    assert r.status_code == 403 and "prepare" in r.json()["detail"]


def test_prepare_dataset_only_writes_inside_data(fixture_dataset, tmp_path):
    st = Settings(classifier_checkpoint=None, model_checkpoint=None, label_vocab=None, allow_dataset_preparation=True)
    with TestClient(create_app(st)) as c:
        for out_dir in (str(tmp_path / "elsewhere"), "../outside", "data/../../escape", "data", "web"):
            r = c.post("/prepare-dataset", json={"xmidi_root": str(fixture_dataset), "out_dir": out_dir, "max_files": 1})
            assert r.status_code == 422, out_dir
        assert c.post("/prepare-dataset", json={"xmidi_root": str(tmp_path / "missing")}).status_code == 422


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MIDICOMPOSER_CONFIG", str(tmp_path / "no-config.yaml"))
    monkeypatch.setenv("MIDICOMPOSER_ALLOW_DATASET_PREPARATION", "false")
    monkeypatch.setenv("MIDICOMPOSER_MAX_UPLOAD_MB", "2.5")
    monkeypatch.setenv("MIDICOMPOSER_JOB_WORKERS", "3")
    monkeypatch.setenv("MIDICOMPOSER_MODEL_CHECKPOINT", "")
    s = Settings.load()
    assert s.allow_dataset_preparation is False and s.max_upload_mb == 2.5 and s.job_workers == 3
    assert s.model_checkpoint is None
    monkeypatch.setenv("MIDICOMPOSER_ALLOW_DATASET_PREPARATION", "1")
    assert Settings.load().allow_dataset_preparation is True
