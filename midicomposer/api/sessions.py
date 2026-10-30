"""Process-memory storage for client sessions. Uploads and jobs exist only here, never on disk.

A browser page opens one session per page load and closes it when the page closes or refreshes.
Since browsers can't guarantee that last message (crash, sleep, lost network), the page also sends
heartbeats and a sweep drops sessions that have gone quiet. Clients that don't use sessions (curl,
scripts) get a short-lived anonymous session per item, which expires when left unused.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .jobs import Job, new_id


class SessionClosed(KeyError):
    """The session ended while a request for it was still being handled."""


@dataclass
class Upload:
    upload_id: str
    data: bytes
    filename: str | None


@dataclass
class Session:
    session_id: str
    ttl: float
    anonymous: bool
    last_seen: float
    uploads: dict[str, Upload] = field(default_factory=dict)
    jobs: dict[str, Job] = field(default_factory=dict)


class SessionStore:
    def __init__(self, session_ttl: float = 180, anonymous_ttl: float = 600, max_uploads: int = 8,
                 max_jobs: int = 20, clock: Callable[[], float] = time.time):
        self.session_ttl, self.anonymous_ttl = session_ttl, anonymous_ttl
        self.max_uploads, self.max_jobs = max_uploads, max_jobs
        self.clock = clock
        self.lock = threading.RLock()
        self.sessions: dict[str, Session] = {}
        self._upload_owner: dict[str, Session] = {}
        self._job_owner: dict[str, Session] = {}

    # ------------------------------------------------------------------ sessions
    def open(self, anonymous: bool = False) -> Session:
        ttl = self.anonymous_ttl if anonymous else self.session_ttl
        s = Session(uuid.uuid4().hex, ttl, anonymous, self.clock())
        with self.lock:
            self.sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        """Look up a session and mark it as still in use; None once it has ended."""
        with self.lock:
            s = self.sessions.get(session_id)
            if s is not None:
                s.last_seen = self.clock()
            return s

    def close(self, session_id: str) -> bool:
        """Dispose of a session: running jobs are told to stop and all its data is released."""
        with self.lock:
            s = self.sessions.pop(session_id, None)
            if s is None:
                return False
            for job_id, job in s.jobs.items():
                job.cancel()
                self._job_owner.pop(job_id, None)
            for upload_id in s.uploads:
                self._upload_owner.pop(upload_id, None)
            s.jobs.clear()
            s.uploads.clear()
            return True

    def sweep(self, now: float | None = None) -> list[str]:
        """Close every session that has gone quiet. Returns the ids that were closed."""
        now = self.clock() if now is None else now
        with self.lock:
            expired = [sid for sid, s in self.sessions.items() if self._expired(s, now)]
            for sid in expired:
                self.close(sid)
        return expired

    @staticmethod
    def _expired(s: Session, now: float) -> bool:
        last = s.last_seen
        if s.anonymous:
            # Nobody heartbeats for these: wait for running work, then time from when it finished.
            if any(j.active for j in s.jobs.values()):
                return False
            last = max([last] + [j.finished_at or 0.0 for j in s.jobs.values()])
        return now - last > s.ttl

    def clear(self) -> None:
        with self.lock:
            for sid in list(self.sessions):
                self.close(sid)

    def counts(self) -> dict:
        with self.lock:
            return {"sessions": len(self.sessions), "uploads": len(self._upload_owner), "jobs": len(self._job_owner)}

    # ------------------------------------------------------------------ uploads
    def add_upload(self, session: Session, data: bytes, filename: str | None) -> Upload:
        upload = Upload(new_id(), data, filename)
        with self.lock:
            if session.session_id not in self.sessions:
                raise SessionClosed(session.session_id)
            session.uploads[upload.upload_id] = upload
            self._upload_owner[upload.upload_id] = session
            while len(session.uploads) > self.max_uploads:
                oldest = next(iter(session.uploads))
                del session.uploads[oldest]
                self._upload_owner.pop(oldest, None)
        return upload

    def find_upload(self, upload_id: str) -> Upload | None:
        with self.lock:
            owner = self._upload_owner.get(upload_id)
            if owner is None:
                return None
            owner.last_seen = self.clock()
            return owner.uploads[upload_id]

    # ------------------------------------------------------------------ jobs
    def add_job(self, session: Session, job: Job) -> None:
        with self.lock:
            if session.session_id not in self.sessions:
                raise SessionClosed(session.session_id)
            session.jobs[job.job_id] = job
            self._job_owner[job.job_id] = session
            # Each regenerate creates a job, so long sessions drop their oldest finished results.
            excess = len(session.jobs) - self.max_jobs
            for old in [j for j in session.jobs.values() if not j.active][:max(0, excess)]:
                del session.jobs[old.job_id]
                self._job_owner.pop(old.job_id, None)

    def find_job(self, job_id: str) -> Job | None:
        with self.lock:
            owner = self._job_owner.get(job_id)
            if owner is None:
                return None
            owner.last_seen = self.clock()
            return owner.jobs[job_id]

    def owner_of_job(self, job_id: str) -> Session | None:
        with self.lock:
            return self._job_owner.get(job_id)
