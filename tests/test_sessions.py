"""Transient storage: uploads and jobs live in memory, end with their session and never touch the disk."""
import os
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient
from starlette.formparsers import MultiPartParser

from build_fixtures import tonal_piece
from midicomposer.api.app import Settings, create_app
from midicomposer.api.jobs import Job, JobCancelled, JobManager, new_id
from midicomposer.api.sessions import SessionClosed, SessionStore
from midicomposer.generation.pipeline import GenerationRequest, run_generation
from midicomposer.midi_io import Score, load_midi, score_to_bytes

BIG_UPLOAD = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0" + b"\x00" * (2 * 1024 * 1024)   # past the parser's 1 MB spool

# An audit hook sees every file opened and temp file created in this process, in any thread.
_disk_events: list = []
_watching = False


def _audit(event, args):
    if not _watching:
        return
    if event in ("tempfile.mkstemp", "tempfile.mkdtemp"):
        _disk_events.append((event, str(args[0])))
    elif event == "open":
        path, mode, flags = args
        writes = (any(c in mode for c in "wax+") if isinstance(mode, str)
                  else isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
        # Python caching its own bytecode on first import is not application data.
        if writes and isinstance(path, (str, bytes, os.PathLike)) and "__pycache__" not in os.fsdecode(path):
            _disk_events.append((event, os.fsdecode(path)))


sys.addaudithook(_audit)


@pytest.fixture
def disk_watch():
    global _watching
    _disk_events.clear()
    _watching = True
    yield _disk_events
    _watching = False


def make_client(**overrides):
    return TestClient(create_app(Settings(classifier_checkpoint=None, model_checkpoint=None, label_vocab=None, **overrides)))


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def wait(client, job_id, headers=None, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/generation/{job_id}", headers=headers).json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job timed out")


def lead_bytes():
    s = tonal_piece(2, "major")
    return score_to_bytes(Score(tracks=[s.tracks[0]], tempos=s.tempos, time_signatures=s.time_signatures))


def open_session(client):
    r = client.post("/session")
    assert r.status_code == 201
    return r.json()["session_id"], {"X-Session-Id": r.json()["session_id"]}


def upload(client, headers):
    r = client.post("/analyze-midi", files={"file": ("lead.mid", lead_bytes(), "audio/midi")}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["upload_id"]


def finished_job(client, headers, upload_id=None):
    body = {"mode": "midi", "upload_id": upload_id, "input_role": "lead", "generated_roles": ["chord_bed", "bass_line"],
            "n_variations": 1} if upload_id else {"bars": 4, "n_variations": 1, "generated_roles": ["chord_bed"]}
    r = client.post("/generate", json=body, headers=headers)
    assert r.status_code == 202, r.text
    j = wait(client, r.json()["job_id"], headers)
    assert j["status"] == "done", j
    return j["job_id"]


def counts(client):
    return client.get("/health").json()["in_memory"]


# --------------------------------------------------------------------------- nothing on disk

def test_the_watch_really_catches_writes_and_parser_spills(client, disk_watch, tmp_path, monkeypatch):
    """Positive control: without this, the 'no disk writes' tests below could pass by seeing nothing."""
    (tmp_path / "x.txt").write_text("x")
    assert any(e == "open" and p.endswith("x.txt") for e, p in disk_watch)
    disk_watch.clear()
    monkeypatch.setattr(MultiPartParser, "spool_max_size", 1024 * 1024)    # the parser's default
    client.post("/analyze-midi", files={"file": ("big.mid", BIG_UPLOAD, "audio/midi")})
    assert any(e == "tempfile.mkstemp" for e, _ in disk_watch)
    disk_watch.clear()


def test_full_session_never_touches_the_disk(client, disk_watch):
    sid, h = open_session(client)
    uid = upload(client, h)
    job = finished_job(client, h, uid)
    assert load_midi(client.get(f"/generation/{job}/download", headers=h).content).tracks
    assert client.get(f"/generation/{job}/manifest", headers=h).json()["job_id"] == job
    tracks = client.get(f"/generation/{job}", headers=h).json()["result"]["candidates"][0]["tracks"]
    generated = [t["track_id"] for t in tracks if t["source"] == "generated"]
    assert client.post(f"/generation/{job}/lock", json={"track_id": generated[0], "locked": True}, headers=h).status_code == 200
    r = client.post("/regenerate-track", json={"job_id": job, "track_id": generated[1], "seed": 5}, headers=h)
    assert wait(client, r.json()["job_id"], h)["status"] == "done"
    r = client.post("/regenerate-section", json={"job_id": job, "start_bar": 0, "end_bar": 2, "seed": 6}, headers=h)
    assert wait(client, r.json()["job_id"], h)["status"] == "done"
    assert client.post(f"/session/{sid}/close").status_code == 204
    assert disk_watch == []


def test_generation_itself_writes_nothing(disk_watch):
    state = run_generation(GenerationRequest(mode="midi", input_role="lead", n_variations=2, classifier_path=None,
                                             generated_roles=["chord_bed"]), input_bytes=lead_bytes())
    assert state.midi_bytes(1) and state.manifest and disk_watch == []


def test_uploads_over_the_parsers_spool_size_stay_in_memory(client, disk_watch):
    r = client.post("/analyze-midi", files={"file": ("big.mid", BIG_UPLOAD, "audio/midi")})
    assert r.status_code == 422
    assert disk_watch == []


def test_oversized_upload_is_refused_before_parsing(disk_watch):
    with make_client(max_upload_mb=0.01) as c:
        r = c.post("/analyze-midi", files={"file": ("big.mid", b"x" * 200_000, "audio/midi")})
    assert r.status_code == 413 and "MB" in r.json()["detail"]
    assert disk_watch == []


# --------------------------------------------------------------------------- closing a session

def test_closing_a_session_disposes_of_everything(client):
    sid, h = open_session(client)
    uid = upload(client, h)
    job = finished_job(client, h, uid)
    assert counts(client) == {"sessions": 1, "uploads": 1, "jobs": 1}
    assert client.get(f"/generation/{job}/download", headers=h).status_code == 200

    assert client.post(f"/session/{sid}/close").status_code == 204
    assert counts(client) == {"sessions": 0, "uploads": 0, "jobs": 0}
    assert client.get(f"/generation/{job}/download").status_code == 404
    assert client.get(f"/generation/{job}").status_code == 404
    assert client.get(f"/generation/{job}/manifest").status_code == 404
    r = client.post("/generate", json={"mode": "midi", "upload_id": uid, "generated_roles": ["pad"]})
    assert r.status_code == 404
    assert client.post(f"/session/{sid}/close").status_code == 204        # closing twice is harmless


def test_requests_from_an_ended_session_are_told_so(client):
    sid, h = open_session(client)
    client.post(f"/session/{sid}/close")
    assert client.post(f"/session/{sid}/heartbeat").status_code == 410
    assert client.post("/generate", json={"bars": 4}, headers=h).status_code == 410
    assert client.get("/generation/whatever", headers=h).status_code == 410
    assert client.post("/analyze-midi", files={"file": ("a.mid", lead_bytes(), "audio/midi")}, headers=h).status_code == 410
    assert counts(client) == {"sessions": 0, "uploads": 0, "jobs": 0}     # and nothing was created for it


def test_a_refresh_is_a_new_session_and_tabs_are_independent(client):
    sid_a, a = open_session(client)
    sid_b, b = open_session(client)
    assert sid_a != sid_b
    job_a, job_b = finished_job(client, a), finished_job(client, b)
    client.post(f"/session/{sid_a}/close")
    assert client.get(f"/generation/{job_a}/download").status_code == 404
    assert client.get(f"/generation/{job_b}/download", headers=b).status_code == 200


def test_server_shutdown_clears_memory():
    c = make_client()
    with c:
        _, h = open_session(c)
        finished_job(c, h)
        store = c.app.state.store
        assert store.counts()["jobs"] == 1
    assert store.counts() == {"sessions": 0, "uploads": 0, "jobs": 0}


# --------------------------------------------------------------------------- quiet sessions

def test_quiet_session_is_swept_but_heartbeats_keep_it_alive(client):
    store = client.app.state.store
    sid, h = open_session(client)
    job = finished_job(client, h)
    ttl = store.session_ttl
    store.sessions[sid].last_seen = time.time() - ttl + 60               # nearly expired
    assert store.sweep() == []
    assert client.post(f"/session/{sid}/heartbeat").status_code == 200
    assert store.sweep(now=time.time() + ttl - 1) == []                   # the heartbeat reset the clock
    assert store.sweep(now=time.time() + ttl + 1) == [sid]
    assert client.get(f"/generation/{job}").status_code == 404
    assert client.post(f"/session/{sid}/heartbeat").status_code == 410


def test_clients_without_sessions_work_and_expire_when_unused(client):
    store = client.app.state.store
    uid = upload(client, None)
    job = finished_job(client, None, uid)
    assert client.get(f"/generation/{job}/download").status_code == 200
    assert store.sweep(now=time.time() + store.session_ttl + 1) == []     # survive a page's shorter limit
    assert len(store.sweep(now=time.time() + store.anonymous_ttl + 1)) == 2
    assert client.get(f"/generation/{job}").status_code == 404
    assert client.post("/generate", json={"mode": "midi", "upload_id": uid}).status_code == 404


def test_anonymous_running_job_is_not_expired():
    now = [1000.0]
    store = SessionStore(session_ttl=10, anonymous_ttl=100, clock=lambda: now[0])
    s = store.open(anonymous=True)
    job = Job(new_id(), "prepare-dataset", status="running")
    store.add_job(s, job)
    now[0] += 1000
    assert store.sweep() == []
    job.status, job.finished_at = "done", now[0]
    now[0] += 50
    assert store.sweep() == []                                            # timed from when it finished
    now[0] += 51
    assert store.sweep() == [s.session_id]


# --------------------------------------------------------------------------- limits

def test_old_uploads_and_finished_jobs_are_dropped_past_the_limits():
    store = SessionStore(max_uploads=2, max_jobs=2)
    s = store.open()
    u1, u2, u3 = (store.add_upload(s, b"x", None) for _ in range(3))
    assert store.find_upload(u1.upload_id) is None and store.find_upload(u3.upload_id) is u3
    running = Job(new_id(), "generation", status="running")
    done1, done2 = Job(new_id(), "generation", status="done"), Job(new_id(), "generation", status="done")
    for j in (running, done1, done2):
        store.add_job(s, j)
    assert store.find_job(running.job_id) is running                      # active jobs are never dropped
    assert store.find_job(done1.job_id) is None and store.find_job(done2.job_id) is done2
    store.close(s.session_id)
    with pytest.raises(SessionClosed):
        store.add_upload(s, b"x", None)
    with pytest.raises(SessionClosed):
        store.add_job(s, Job(new_id(), "generation"))


# --------------------------------------------------------------------------- cancellation

def test_running_job_stops_and_drops_its_result_when_its_session_closes():
    store, jobs = SessionStore(), JobManager(1)
    s = store.open()
    started, release = threading.Event(), threading.Event()

    def work(job):
        started.set()
        release.wait(5)
        job.report("next step")                     # the cancellation point
        return {"never": "kept"}

    job = Job(new_id(), "generation")
    store.add_job(s, job)
    jobs.submit(job, work)
    assert started.wait(5)
    store.close(s.session_id)
    release.set()
    for _ in range(200):
        if not job.active:
            break
        time.sleep(0.01)
    assert job.status == "cancelled" and job.result is None
    jobs.shutdown()


def test_job_that_finishes_after_cancellation_keeps_nothing():
    jobs = JobManager(1)
    job = Job(new_id(), "generation")
    job.cancel()
    jobs.submit(job, lambda j: {"result": "ignored"})
    jobs.shutdown()
    for _ in range(200):
        if not job.active:
            break
        time.sleep(0.01)
    assert job.status == "cancelled" and job.result is None


def test_generation_stops_at_the_next_step_when_progress_raises():
    calls = []

    def progress(message):
        calls.append(message)
        if len(calls) == 3:
            raise JobCancelled()

    with pytest.raises(JobCancelled):
        run_generation(GenerationRequest(genre="pop", bars=8, n_variations=3, classifier_path=None,
                                         generated_roles=["chord_bed", "bass_line", "drums"]), progress=progress)
    assert len(calls) == 3
