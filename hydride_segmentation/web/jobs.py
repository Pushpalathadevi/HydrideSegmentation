"""In-memory background jobs and user-visible progress for the web app."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobEvent:
    """One timestamped progress or log event."""

    sequence: int
    timestamp_utc: str
    stage: str
    percent: int
    message: str
    level: str = "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "timestamp_utc": self.timestamp_utc,
            "stage": self.stage,
            "percent": int(self.percent),
            "message": self.message,
            "level": self.level,
        }


@dataclass
class WebJob:
    """Thread-safe state for one non-persistent segmentation job."""

    job_id: str
    state: str = "queued"
    stage: str = "queued"
    percent: int = 0
    message: str = "Waiting for an available worker."
    created_utc: str = field(default_factory=_utc_now)
    started_utc: str = ""
    finished_utc: str = ""
    created_monotonic: float = field(default_factory=time.monotonic)
    finished_monotonic: float = 0.0
    events: list[JobEvent] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_event(
        self,
        stage: str,
        percent: int,
        message: str,
        *,
        level: str = "info",
    ) -> None:
        """Update headline progress and append a timestamped event."""

        clean_stage = str(stage or "working").strip().lower() or "working"
        clean_message = str(message or "Working.").strip() or "Working."
        bounded = max(0, min(100, int(percent)))
        with self._lock:
            if self.state in {"completed", "failed"}:
                return
            self.stage = clean_stage
            self.percent = max(self.percent, bounded)
            self.message = clean_message
            event = JobEvent(
                sequence=len(self.events) + 1,
                timestamp_utc=_utc_now(),
                stage=clean_stage,
                percent=self.percent,
                message=clean_message,
                level=str(level or "info"),
            )
            self.events.append(event)
        _LOGGER.info(
            "WEB_JOB | id=%s state=%s stage=%s percent=%d message=%s",
            self.job_id,
            self.state,
            clean_stage,
            self.percent,
            clean_message,
        )

    def mark_running(self) -> None:
        with self._lock:
            self.state = "running"
            self.started_utc = _utc_now()
        self.add_event("starting", 4, "A worker accepted the job.")

    def mark_completed(self, result: dict[str, Any]) -> None:
        with self._lock:
            if self.state == "failed":
                return
            self.state = "completed"
            self.stage = "complete"
            self.percent = 100
            self.message = "Segmentation and analysis completed."
            self.result = result
            self.finished_utc = _utc_now()
            self.finished_monotonic = time.monotonic()
            self.events.append(
                JobEvent(
                    sequence=len(self.events) + 1,
                    timestamp_utc=self.finished_utc,
                    stage="complete",
                    percent=100,
                    message=self.message,
                )
            )

    def mark_failed(self, code: str, detail: str) -> None:
        clean_detail = str(detail or "The job failed.").strip()
        with self._lock:
            self.state = "failed"
            self.stage = "failed"
            self.message = clean_detail
            self.error = {"code": str(code), "detail": clean_detail}
            self.finished_utc = _utc_now()
            self.finished_monotonic = time.monotonic()
            self.events.append(
                JobEvent(
                    sequence=len(self.events) + 1,
                    timestamp_utc=self.finished_utc,
                    stage="failed",
                    percent=self.percent,
                    message=clean_detail,
                    level="error",
                )
            )

    def to_dict(self, *, after_sequence: int = 0, include_result: bool = True) -> dict[str, Any]:
        """Return a JSON-safe snapshot, optionally with only new events."""

        with self._lock:
            payload: dict[str, Any] = {
                "ok": self.state != "failed",
                "job_id": self.job_id,
                "state": self.state,
                "stage": self.stage,
                "percent": int(self.percent),
                "message": self.message,
                "created_utc": self.created_utc,
                "started_utc": self.started_utc,
                "finished_utc": self.finished_utc,
                "terminal": self.state in {"completed", "failed"},
                "events": [
                    event.to_dict()
                    for event in self.events
                    if int(event.sequence) > int(after_sequence)
                ],
                "last_event_sequence": len(self.events),
            }
            if self.error is not None:
                payload["error"] = dict(self.error)
            if include_result and self.state == "completed":
                payload["result"] = self.result
            return payload


class WebJobManager:
    """Bounded, in-memory job manager with no upload or result persistence."""

    def __init__(
        self,
        *,
        max_concurrent_jobs: int = 2,
        max_retained_jobs: int = 32,
        retention_seconds: int = 1800,
    ) -> None:
        self._slots = threading.BoundedSemaphore(max(1, int(max_concurrent_jobs)))
        self._max_concurrent_jobs = max(1, int(max_concurrent_jobs))
        self._max_retained_jobs = max(self._max_concurrent_jobs, int(max_retained_jobs))
        self._retention_seconds = max(60, int(retention_seconds))
        self._jobs: dict[str, WebJob] = {}
        self._lock = threading.Lock()

    def _cleanup(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if job.finished_monotonic
                and now - job.finished_monotonic > self._retention_seconds
            ]
            for job_id in expired:
                self._jobs.pop(job_id, None)

            if len(self._jobs) <= self._max_retained_jobs:
                return
            completed = sorted(
                (
                    job
                    for job in self._jobs.values()
                    if job.state in {"completed", "failed"}
                ),
                key=lambda item: item.finished_monotonic,
            )
            for job in completed:
                if len(self._jobs) <= self._max_retained_jobs:
                    break
                self._jobs.pop(job.job_id, None)

    @property
    def active_jobs(self) -> int:
        self._cleanup()
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.state == "running")

    @property
    def queued_jobs(self) -> int:
        self._cleanup()
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.state == "queued")

    def get(self, job_id: str) -> WebJob | None:
        self._cleanup()
        with self._lock:
            return self._jobs.get(str(job_id))

    def submit(
        self,
        runner: Callable[[Callable[[str, int, str], None]], dict[str, Any]],
    ) -> WebJob | None:
        """Submit work or return ``None`` when the bounded queue is full."""

        self._cleanup()
        with self._lock:
            pending = sum(
                1 for job in self._jobs.values() if job.state in {"queued", "running"}
            )
            if pending >= self._max_retained_jobs:
                return None
            job = WebJob(job_id=uuid.uuid4().hex)
            self._jobs[job.job_id] = job
        job.add_event("queued", 1, "Validated request queued in memory.")

        def work() -> None:
            acquired = self._slots.acquire(timeout=300)
            if not acquired:
                job.mark_failed(
                    "SERVER_BUSY",
                    "No worker became available before the queue timeout. Please try again.",
                )
                return
            try:
                job.mark_running()
                result = runner(job.add_event)
                job.mark_completed(result)
            except MemoryError:
                job.mark_failed(
                    "OUT_OF_MEMORY",
                    "The server ran out of memory. Try a smaller image or disable optional analysis.",
                )
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                _LOGGER.exception("Background web job %s failed", job.job_id)
                job.mark_failed("SEGMENTATION_FAILED", f"Segmentation failed: {exc}")
            finally:
                self._slots.release()

        thread = threading.Thread(
            target=work,
            name=f"microseg-web-job-{job.job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return job
