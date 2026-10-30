"""Minimal in-process asynchronous jobs (thread pool). Jobs and their results live only in memory."""
from __future__ import annotations

import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


class JobCancelled(Exception):
    """Raised inside a running job once its client has gone away."""


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "queued"          # queued | running | done | error | cancelled
    progress: list[str] = field(default_factory=list)
    result: object = None           # GenerationState, or a plain dict for dataset preparation
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    parent_job_id: str | None = None
    cancelled: bool = False

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def cancel(self) -> None:
        self.cancelled = True

    def report(self, message: str) -> None:
        """Progress callback. Doubles as the cancellation point between generation steps."""
        if self.cancelled:
            raise JobCancelled()
        self.progress.append(message)

    def view(self) -> dict:
        d = {"job_id": self.job_id, "kind": self.kind, "status": self.status, "progress": self.progress[-5:],
             "error": self.error, "parent_job_id": self.parent_job_id,
             "elapsed_seconds": round((self.finished_at or time.time()) - self.created_at, 2)}
        if self.result is not None:
            d["result"] = self.result.view() if hasattr(self.result, "view") else self.result
        return d


class JobManager:
    def __init__(self, workers: int = 2):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")

    def submit(self, job: Job, fn: Callable[[Job], object], user_errors: tuple[type[Exception], ...] = ()) -> Job:
        def run():
            job.status = "running"
            try:
                job.report(f"started {job.kind}")
                result = fn(job)
                if job.cancelled:
                    raise JobCancelled()
                job.result = result
                job.status = "done"
            except JobCancelled:
                job.status = "cancelled"
            except user_errors as e:
                job.status, job.error = "error", str(e)
            except Exception as e:  # unexpected: keep a short message, log the traceback
                job.status, job.error = "error", f"Internal error: {type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                job.finished_at = time.time()

        self.pool.submit(run)
        return job

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
