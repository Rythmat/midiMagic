"""FastAPI service + static web UI.

    python -m midicomposer.cli serve            # http://127.0.0.1:8000

Uploads and generation results are held in process memory only (see sessions.py). The server never
writes them to disk; a page's data is disposed of when that page closes or refreshes.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.formparsers import MultiPartParser

from .. import __version__
from ..analyzer import analyze_score
from ..classifiers import load_scorer
from ..generation.neural import load_neural_generator
from ..generation.pipeline import (GenerationError, GenerationRequest, GenerationState, regenerate_section,
                                   regenerate_track, run_generation, set_track_lock)
from ..midi_io import MidiLoadError, load_midi
from ..roles import GENERATED_ROLES, INPUT_ROLES, normalize_input_role
from ..styles import PRESETS
from ..theory import MODES, PITCH_NAMES
from .jobs import Job, JobManager, new_id
from .sessions import Session, SessionClosed, SessionStore

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
XMIDI_DEFAULT_GENRES = ["classical", "country", "jazz", "pop", "rock", "traditional"]
XMIDI_DEFAULT_EMOTIONS = ["angry", "exciting", "fear", "funny", "happy", "lazy", "magnificent", "quiet", "romantic",
                          "sad", "warm"]
SWEEP_SECONDS = 15
FORM_OVERHEAD_BYTES = 64 * 1024     # multipart boundaries and text fields around the uploaded file
SESSION_GONE = "This session has ended; its uploads and results were discarded."


@dataclass
class Settings:
    classifier_checkpoint: str | None = "checkpoints/classifiers.pt"
    model_checkpoint: str | None = "checkpoints/composer_dev.pt"
    label_vocab: str | None = "data/xmidi/label_vocab.json"
    max_upload_mb: float = 10
    job_workers: int = 2
    session_ttl_seconds: float = 180        # a page that stops sending heartbeats is dropped after this
    heartbeat_seconds: float = 30
    anonymous_ttl_seconds: float = 600      # curl/script uploads and jobs expire when unused for this long
    max_uploads_per_session: int = 8
    max_jobs_per_session: int = 20

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        s = cls()
        p = Path(path or os.environ.get("MIDICOMPOSER_CONFIG", ROOT / "configs" / "default.yaml"))
        if p.exists():
            for k, v in (yaml.safe_load(p.read_text()) or {}).items():
                if hasattr(s, k):
                    setattr(s, k, v)
        for k in asdict(s):
            env = os.environ.get(f"MIDICOMPOSER_{k.upper()}")
            if env is not None:
                current = getattr(s, k)
                setattr(s, k, type(current)(env) if isinstance(current, (int, float)) else (env or None))
        for k in ("classifier_checkpoint", "model_checkpoint", "label_vocab"):
            v = getattr(s, k)
            if v and not Path(v).is_absolute():
                setattr(s, k, str(ROOT / v))
        return s


class GenerateBody(BaseModel):
    mode: str = Field("metadata", description="'metadata' (mode 2) or 'midi' (mode 1)")
    upload_id: str | None = None
    genre: str | None = "pop"
    emotion: str | None = "happy"
    key: str | None = "C"
    scale_mode: str | None = None
    tempo: float | None = None
    meter: str | None = "4/4"
    bars: int | None = None
    density: float = 0.5
    instrumentation: str | None = None
    form: str | None = None
    input_role: str | None = None
    generated_roles: list[str] = ["full_composition"]
    target_key: str | None = None
    root_override: str | None = None
    n_variations: int = 3
    seed: int = 0
    engine: str = "rule"
    track_roles: dict[int, str] | None = None


class RegenerateTrackBody(BaseModel):
    job_id: str
    track_id: int
    candidate: int = 0
    seed: int | None = None


class RegenerateSectionBody(BaseModel):
    job_id: str
    candidate: int = 0
    section_index: int | None = None
    start_bar: int | None = None
    end_bar: int | None = None
    track_ids: list[int] | None = None
    seed: int | None = None


class LockBody(BaseModel):
    track_id: int
    locked: bool
    candidate: int = 0


class PrepareBody(BaseModel):
    xmidi_root: str | None = None
    out_dir: str = "data/xmidi_api"
    max_files: int | None = 200
    workers: int = 4


def create_app(settings: Settings | None = None) -> FastAPI:
    st = settings or Settings.load()
    jobs = JobManager(st.job_workers)
    store = SessionStore(st.session_ttl_seconds, st.anonymous_ttl_seconds, st.max_uploads_per_session,
                         st.max_jobs_per_session)
    max_upload_bytes = int(st.max_upload_mb * 1024 * 1024)
    # The form parser spills anything over its spool size to a temp file. Keep every upload we accept
    # in memory; bigger bodies are refused below, before parsing starts.
    MultiPartParser.spool_max_size = max_upload_bytes + FORM_OVERHEAD_BYTES

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def sweeper():
            while True:
                await asyncio.sleep(SWEEP_SECONDS)
                store.sweep()
        task = asyncio.create_task(sweeper())
        try:
            yield
        finally:
            task.cancel()
            store.clear()
            jobs.shutdown()

    app = FastAPI(title="MIDI Composer", version=__version__, lifespan=lifespan,
                  description="Conditional symbolic-music composition with root-relative MIDI normalization.")

    @app.middleware("http")
    async def refuse_oversized_uploads(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/analyze-midi":
            length = request.headers.get("content-length", "")
            if not length.isdigit():
                return JSONResponse({"detail": "Content-Length is required for uploads."}, status_code=411)
            if int(length) > max_upload_bytes + FORM_OVERHEAD_BYTES:
                return JSONResponse({"detail": f"File exceeds {st.max_upload_mb} MB."}, status_code=413)
        return await call_next(request)

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.settings, app.state.jobs, app.state.store = st, jobs, store

    def live_session(x_session_id: str | None = Header(default=None)) -> Session | None:
        """The caller's session, or None for clients that don't use sessions. A stale id is an error."""
        if x_session_id is None:
            return None
        session = store.get(x_session_id)
        if session is None:
            raise HTTPException(410, SESSION_GONE)
        return session

    def _start_job(session: Session, kind: str, fn, parent_job_id: str | None = None, user_errors=()) -> Job:
        job = Job(new_id(), kind, parent_job_id=parent_job_id)
        try:
            store.add_job(session, job)
        except SessionClosed:
            raise HTTPException(410, SESSION_GONE)
        return jobs.submit(job, fn, user_errors=user_errors)

    def _job_or_404(job_id: str) -> Job:
        job = store.find_job(job_id)
        if job is None:
            raise HTTPException(404, f"Unknown job {job_id}")
        return job

    def _finished_state(job_id: str) -> GenerationState:
        job = _job_or_404(job_id)
        if job.status != "done":
            raise HTTPException(409, f"Job is {job.status}; no result yet.")
        if not isinstance(job.result, GenerationState):
            raise HTTPException(404, f"Job {job_id} is not a generation job.")
        return job.result

    def _request(body: GenerateBody) -> GenerationRequest:
        d = body.model_dump()
        d.pop("upload_id")
        return GenerationRequest(**d, classifier_path=st.classifier_checkpoint, model_path=st.model_checkpoint)

    # ------------------------------------------------------------------ meta
    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "in_memory": store.counts()}

    # ------------------------------------------------------------------ sessions
    @app.post("/session", status_code=201)
    def open_session():
        session = store.open()
        return {"session_id": session.session_id, "heartbeat_seconds": st.heartbeat_seconds,
                "ttl_seconds": st.session_ttl_seconds}

    @app.post("/session/{session_id}/heartbeat")
    def session_heartbeat(session_id: str):
        if store.get(session_id) is None:
            raise HTTPException(410, SESSION_GONE)
        return {"status": "alive"}

    @app.post("/session/{session_id}/close", status_code=204)
    def close_session(session_id: str):
        store.close(session_id)

    @app.get("/options")
    def options():
        genres, emotions, source = XMIDI_DEFAULT_GENRES, XMIDI_DEFAULT_EMOTIONS, "built-in XMIDI defaults"
        if st.label_vocab and Path(st.label_vocab).exists():
            lv = json.loads(Path(st.label_vocab).read_text())
            genres = [g["label"] for g in lv["genres"]]
            emotions = [e["label"] for e in lv["emotions"]]
            source = st.label_vocab
        neural = load_neural_generator(st.model_checkpoint)
        clf = load_scorer(st.classifier_checkpoint)
        return {
            "genres": genres, "emotions": emotions, "label_source": source, "keys": PITCH_NAMES,
            "modes": list(MODES), "input_roles": INPUT_ROLES, "generated_roles": GENERATED_ROLES,
            "meters": ["4/4", "3/4", "6/8", "2/4", "12/8", "5/4"],
            "forms": ["auto", "verse-chorus", "AABA", "ABA", "AABB", "rondo", "strophic", "through-composed",
                      "12-bar-blues", "head-solo-head", "build-drop"],
            "instrumentation_presets": sorted(PRESETS),
            "engines": {"rule": True, "neural": neural is not None, "hybrid": neural is not None},
            "neural_model": neural.describe() if neural else None,
            "classifier": {"test_metrics": clf.metrics.get("test")} if clf else None,
        }

    # ------------------------------------------------------------------ analysis
    @app.post("/analyze-midi")
    async def analyze_midi(file: UploadFile = File(...), root_override: str | None = Form(None),
                           input_role: str | None = Form(None), caller: Session | None = Depends(live_session)):
        data = await file.read()
        if not data:
            raise HTTPException(400, "Uploaded file is empty.")
        if len(data) > max_upload_bytes:
            raise HTTPException(413, f"File exceeds {st.max_upload_mb} MB.")
        try:
            score = load_midi(data)
        except MidiLoadError as e:
            raise HTTPException(422, str(e))
        if not any(t.notes for t in score.tracks):
            raise HTTPException(422, "The MIDI file contains no notes.")
        try:
            role = normalize_input_role(input_role) if input_role else None
            analysis = analyze_score(score, root_override=root_override or None, input_role=role)
        except ValueError as e:
            raise HTTPException(422, str(e))
        try:
            upload = store.add_upload(caller or store.open(anonymous=True), data, file.filename)
        except SessionClosed:
            raise HTTPException(410, SESSION_GONE)
        return {"upload_id": upload.upload_id, "filename": file.filename, "analysis": analysis}

    # ------------------------------------------------------------------ generation
    @app.post("/generate", status_code=202)
    def generate(body: GenerateBody, caller: Session | None = Depends(live_session)):
        try:
            req = _request(body)
            req.validate()
        except (GenerationError, TypeError) as e:
            raise HTTPException(422, str(e))
        data = name = None
        if req.mode == "midi":
            if not body.upload_id:
                raise HTTPException(422, "MIDI mode needs an upload_id from /analyze-midi.")
            upload = store.find_upload(body.upload_id)
            if upload is None:
                raise HTTPException(404, f"Unknown upload_id {body.upload_id}")
            data, name = upload.data, upload.filename

        def work(job):
            return run_generation(req, input_bytes=data, input_name=name, job_id=job.job_id, progress=job.report)
        job = _start_job(caller or store.open(anonymous=True), "generation", work, user_errors=(GenerationError,))
        return {"job_id": job.job_id, "status": job.status}

    @app.get("/generation/{job_id}")
    def generation_status(job_id: str, _caller: Session | None = Depends(live_session)):
        return _job_or_404(job_id).view()

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, _caller: Session | None = Depends(live_session)):
        return _job_or_404(job_id).view()

    @app.get("/generation/{job_id}/download")
    def download(job_id: str, candidate: int = Query(0, ge=0), _caller: Session | None = Depends(live_session)):
        state = _finished_state(job_id)
        if candidate >= len(state.candidates):
            raise HTTPException(404, "candidate index out of range")
        name = f"{job_id}_candidate_{candidate + 1}_seed{state.candidates[candidate].seed}.mid"
        return Response(state.midi_bytes(candidate), media_type="audio/midi",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/generation/{job_id}/manifest")
    def manifest(job_id: str, _caller: Session | None = Depends(live_session)):
        return _finished_state(job_id).manifest

    @app.post("/regenerate-track", status_code=202)
    def regen_track(body: RegenerateTrackBody, _caller: Session | None = Depends(live_session)):
        parent = _finished_state(body.job_id)
        if not 0 <= body.candidate < len(parent.candidates):
            raise HTTPException(422, "candidate index out of range")
        if body.track_id not in parent.candidates[body.candidate].generated_ids:
            raise HTTPException(422, "Only generated tracks can be regenerated; input tracks are locked.")
        job = _start_job(store.owner_of_job(body.job_id), "regenerate-track", lambda j: regenerate_track(
            parent, body.candidate, body.track_id, body.seed, job_id=j.job_id, progress=j.report),
            parent_job_id=body.job_id, user_errors=(GenerationError,))
        return {"job_id": job.job_id, "status": job.status, "parent_job_id": body.job_id}

    @app.post("/regenerate-section", status_code=202)
    def regen_section(body: RegenerateSectionBody, _caller: Session | None = Depends(live_session)):
        parent = _finished_state(body.job_id)
        if not 0 <= body.candidate < len(parent.candidates):
            raise HTTPException(422, "candidate index out of range")
        if body.section_index is None and (body.start_bar is None or body.end_bar is None):
            raise HTTPException(422, "give section_index or start_bar and end_bar")
        job = _start_job(store.owner_of_job(body.job_id), "regenerate-section", lambda j: regenerate_section(
            parent, body.candidate, body.section_index, body.start_bar, body.end_bar, body.track_ids, body.seed,
            job_id=j.job_id, progress=j.report), parent_job_id=body.job_id, user_errors=(GenerationError,))
        return {"job_id": job.job_id, "status": job.status, "parent_job_id": body.job_id}

    @app.post("/generation/{job_id}/lock")
    def lock_track(job_id: str, body: LockBody, _caller: Session | None = Depends(live_session)):
        state = _finished_state(job_id)
        try:
            set_track_lock(state, body.candidate, body.track_id, body.locked)
        except (GenerationError, IndexError) as e:
            raise HTTPException(422, str(e))
        return {"job_id": job_id, "track_id": body.track_id, "locked": body.locked}

    # ------------------------------------------------------------------ dataset
    @app.post("/prepare-dataset", status_code=202)
    def prepare(body: PrepareBody):
        from ..dataset.prepare import ENV_ROOT, PrepareConfig, prepare_dataset
        root = body.xmidi_root or os.environ.get(ENV_ROOT)
        if not root or not Path(root).is_dir():
            raise HTTPException(422, f"Provide xmidi_root or set {ENV_ROOT} to an existing directory.")
        out = body.out_dir if Path(body.out_dir).is_absolute() else str(ROOT / body.out_dir)
        cfg = PrepareConfig(xmidi_root=root, out_dir=out, max_files=body.max_files, workers=body.workers)
        job = _start_job(store.open(anonymous=True), "prepare-dataset", lambda j: prepare_dataset(
            cfg, progress=lambda i, n: j.report(f"{i}/{n}")))
        return {"job_id": job.job_id, "status": job.status}

    # ------------------------------------------------------------------ web UI
    # Mounted last so API routes take precedence; relative asset paths in index.html then work
    # both here and when web/ is served by any static server (with ?api=http://host:port).
    if WEB_DIR.exists():
        app.mount("/", _RevalidatingStatic(directory=WEB_DIR, html=True), name="web")

    return app


class _RevalidatingStatic(StaticFiles):
    """Static files that browsers must revalidate (ETag), so UI updates are never served stale."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app = create_app()
