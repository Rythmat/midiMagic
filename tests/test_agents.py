"""Agents: text planner, emotion reader and critic (baselines), and their API endpoints."""
import time

import pytest
from fastapi.testclient import TestClient

from build_fixtures import tonal_piece
from midicomposer.agents import critique, plan_from_text, read_emotion
from midicomposer.api.app import Settings, create_app
from midicomposer.midi_io import score_to_bytes
from midicomposer.root import detect_root

GENRES = ["classical", "country", "jazz", "pop", "rock", "traditional"]
EMOTIONS = ["angry", "exciting", "fear", "funny", "happy", "lazy", "magnificent", "quiet", "romantic", "sad", "warm"]


@pytest.mark.parametrize("prompt, expected", [
    ("A sad, slow piano and cello piece in D minor, 16 bars",
     {"emotion": "sad", "key": "D", "scale_mode": "minor", "bars": 16, "tempo": 76}),
    ("upbeat summer pop song with guitar at 128 bpm", {"genre": "pop", "tempo": 128}),
    ("epic heroic orchestral theme", {"genre": "classical", "emotion": "magnificent", "instrumentation": "orchestral"}),
    ("a lazy jazz trio tune in Bb", {"genre": "jazz", "emotion": "lazy", "key": "A#", "instrumentation": "jazz trio"}),
])
def test_plan_from_text_maps_keywords(prompt, expected):
    p = plan_from_text(prompt, GENRES, EMOTIONS)
    assert p["source"] == "keywords"
    for k, v in expected.items():
        assert p["settings"][k] == v, (k, p["settings"])
    assert p["settings"]["generated_roles"] == ["full_composition"]


def test_plan_from_text_reports_what_it_could_not_find():
    p = plan_from_text("something nice", GENRES, EMOTIONS)
    assert p["settings"]["genre"] is None and set(p["unmatched"]) == {"genre", "emotion"}
    assert "no drums" in plan_from_text("chill tune without drums", GENRES, EMOTIONS)["settings"]["instrumentation"]


def test_read_emotion_heuristic_without_classifier():
    s = tonal_piece(2, "minor")
    out = read_emotion(s, detect_root(s), None, EMOTIONS)
    assert out["source"] == "heuristic" and len(out["emotion"]) == 3
    scores = [e["score"] for e in out["emotion"]]
    assert scores == sorted(scores, reverse=True) and all(0 < x <= 1 for x in scores)
    assert out["emotion"][0]["label"] in {"sad", "fear", "angry", "quiet", "lazy"}   # minor mode -> low valence


def _result(tracks, sections, emotion=0.9, drum_style="pop"):
    return {"meter": "4/4", "plan": {"emotion": "happy", "genre": "pop", "drum_style": drum_style, "sections": sections},
            "candidates": [{"scores": {"emotion": emotion}, "validation": {"issues": []}, "tracks": tracks}]}


def _track(tid, name, role, program, notes, source="generated", is_drum=False, locked=False):
    return {"track_id": tid, "name": name, "role": role, "program": program, "instrument": f"prog{program}",
            "is_drum": is_drum, "source": source, "locked": locked, "n_notes": len(notes), "notes": notes}


def test_critic_flags_duplicate_timbre_missing_drums_and_flat_transitions():
    lead = _track(1, "Lead", "lead_melody", 0, [[b, 1, 72, 90] for b in range(0, 64, 2)])
    chords = _track(2, "Chords", "chord_bed", 0, [[b, 2, 60, 80] for b in range(0, 64, 2)])
    sections = [{"name": "verse", "letter": "A", "start_bar": 0, "end_bar": 8, "energy": 0.5, "progression": ["I", "V"]},
                {"name": "chorus", "letter": "B", "start_bar": 8, "end_bar": 16, "energy": 0.5, "progression": ["I", "V"]}]
    c = critique(_result([lead, chords], sections, emotion=0.3), 0)
    kinds = [(n["kind"], n["severity"]) for n in c["notes"]]
    assert ("instrumentation", "suggestion") in kinds
    msgs = " ".join(n["message"] for n in c["notes"])
    assert "both use" in msgs and "no bass" in msgs and "usually has drums" in msgs and "same chords and energy" in msgs
    actions = {n["kind"]: n["action"] for n in c["notes"] if n["action"]}
    assert actions["transition"] == {"type": "regenerate_section", "section_index": 1,
                                     "label": "Regenerate chorus (bars 9-16)"}
    assert actions["emotion"]["type"] == "regenerate_track" and actions["emotion"]["track_id"] == 1
    assert "instrumentation" not in actions        # regenerating can't change an instrument


def test_critic_never_suggests_regenerating_input_or_locked_tracks():
    a = _track(0, "Input", "lead", 0, [[b, 1, 72, 90] for b in range(16)], source="input")
    b = _track(1, "Locked", "chord_bed", 0, [[b, 1, 60, 80] for b in range(16)], locked=True)
    c = critique(_result([a, b], [], emotion=0.3), 0)
    assert all(n["action"] is None for n in c["notes"])
    assert c["notes"][0]["severity"] == "warning"              # weak emotion fit comes first


@pytest.fixture(scope="module")
def client():
    st = Settings(classifier_checkpoint=None, model_checkpoint=None, label_vocab=None)
    with TestClient(create_app(st)) as c:
        yield c


def test_agent_endpoints(client):
    o = client.get("/options").json()
    assert o["agents"] == {"text_planner": "keywords", "critic": "rules", "emotion_reader": "heuristic"}
    p = client.post("/plan-from-text", json={"prompt": "happy pop in G, 8 bars"}).json()
    assert p["settings"]["key"] == "G" and p["settings"]["bars"] == 8
    assert client.post("/plan-from-text", json={"prompt": "   "}).status_code == 422

    a = client.post("/analyze-midi", files={"file": ("t.mid", score_to_bytes(tonal_piece(0, "major")), "audio/midi")})
    assert a.status_code == 200 and a.json()["analysis"]["style"]["emotion"]

    r = client.post("/generate", json={"mode": "metadata", "genre": "pop", "emotion": "happy", "key": "C", "bars": 16,
                                       "n_variations": 2, "seed": 1})
    job = r.json()["job_id"]
    for _ in range(300):
        if client.get(f"/generation/{job}").json()["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    cr = client.get(f"/generation/{job}/critique", params={"candidate": 1})
    assert cr.status_code == 200
    body = cr.json()
    assert body["source"] == "rules" and body["summary"] and isinstance(body["notes"], list)
    assert client.get(f"/generation/{job}/critique", params={"candidate": 5}).status_code == 404
