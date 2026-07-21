"""In-memory job registry for the adesc web backend.

Deliberately web-agnostic: no FastAPI/starlette imports live here so the store
stays unit-testable and the web layer (``server.py``) owns all HTTP/WS types.

State is kept in a single process (see ``JobStore``). Every mutation also
mirrors a small ``status.json`` into the job's temp dir, so a restart can
recover terminal jobs and flag in-flight ones as ``interrupted`` rather than
404ing. This design assumes a *single* uvicorn worker — an in-memory store
with ``--workers > 1`` would split jobs across processes.
"""

import contextlib
import json
import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from timeline import Timeline

logger = logging.getLogger(__name__)

# Single owned root so startup recovery / TTL sweeps can glob only our dirs.
JOBS_ROOT = Path(tempfile.gettempdir()) / "adesc-jobs"

# Max jobs that may be queued or processing at once; uploads past this are
# rejected (server returns 503) rather than exhausting disk/CPU.
MAX_ACTIVE_JOBS = 4

# How long a terminal job's temp dir (keyframes, audio, status) is kept around
# so Q&A can still read the frames, before the sweep deletes it.
JOB_TTL_SEC = 60 * 60

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_INTERRUPTED = "interrupted"

ACTIVE_STATUSES = {STATUS_QUEUED, STATUS_PROCESSING}
TERMINAL_STATUSES = {STATUS_DONE, STATUS_ERROR, STATUS_INTERRUPTED}


@dataclass
class Job:
    id: str
    dir: Path
    status: str = STATUS_QUEUED
    timeline: Timeline | None = None
    error: str | None = None
    # Append-only log of every event published for this job, replayed to any
    # websocket that connects (or reconnects) after events have already fired.
    events: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def source_video(self) -> Path | None:
        matches = sorted(self.dir.glob("source.*"))
        return matches[0] if matches else None


class JobStore:
    """Registry of jobs plus per-job websocket subscriber fan-out.

    All methods are synchronous and free of ``await``, so on a single event
    loop they run atomically with respect to each other and to the sweep task
    — no explicit lock is required. Subscriber queues are pushed via
    ``put_nowait``, which is safe from the loop thread. Callers must invoke
    ``publish`` from the event loop (never a worker thread), since
    ``asyncio.Queue`` is not thread-safe.
    """

    def __init__(self, root: Path = JOBS_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        # job_id -> list of subscriber queues (asyncio.Queue, typed loosely to
        # avoid importing asyncio types into this web-agnostic module).
        self._subscribers: dict[str, list] = {}

    # -- lifecycle -----------------------------------------------------------

    def create(self) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, dir=job_dir)
        self._jobs[job_id] = job
        self._persist(job)
        logger.info("job %s created at %s", job_id, job_dir)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ACTIVE_STATUSES)

    def at_capacity(self) -> bool:
        return self.active_count() >= MAX_ACTIVE_JOBS

    # -- mutation + streaming ------------------------------------------------

    def set_status(self, job_id: str, status: str, error: str | None = None) -> None:
        job = self._jobs[job_id]
        job.status = status
        if error is not None:
            job.error = error
        if status == STATUS_ERROR:
            logger.error("job %s -> %s: %s", job_id, status, error)
        else:
            logger.info("job %s -> %s", job_id, status)
        self.publish(job_id, {"type": "status", "status": status, "error": error})

    def set_timeline(self, job_id: str, timeline: Timeline) -> None:
        self._jobs[job_id].timeline = timeline

    def publish(self, job_id: str, event: dict) -> None:
        job = self._jobs[job_id]
        job.events.append(event)
        job.updated_at = time.time()
        self._persist(job)
        for queue in self._subscribers.get(job_id, []):
            queue.put_nowait(event)

    def subscribe(self, job_id: str, queue) -> None:
        """Register ``queue`` for a job and seed it with the replay log.

        The queue receives every already-published event first (history
        replay), then all future events. If the job is already terminal, the
        replay includes its terminal ``status`` event, so the consumer will
        see it and can close.
        """
        job = self._jobs[job_id]
        for event in job.events:
            queue.put_nowait(event)
        self._subscribers.setdefault(job_id, []).append(queue)

    def unsubscribe(self, job_id: str, queue) -> None:
        subs = self._subscribers.get(job_id)
        if not subs:
            return
        with contextlib.suppress(ValueError):
            subs.remove(queue)
        if not subs:
            self._subscribers.pop(job_id, None)

    # -- cleanup + recovery --------------------------------------------------

    def sweep(
        self, ttl_sec: float = JOB_TTL_SEC, now: float | None = None
    ) -> list[str]:
        """Delete temp dirs of terminal jobs older than ``ttl_sec``.

        Never touches queued/processing jobs. Returns the swept job ids.
        """
        now = now if now is not None else time.time()
        swept = []
        for job_id, job in list(self._jobs.items()):
            if job.status not in TERMINAL_STATUSES:
                continue
            if now - job.updated_at < ttl_sec:
                continue
            shutil.rmtree(job.dir, ignore_errors=True)
            self._jobs.pop(job_id, None)
            self._subscribers.pop(job_id, None)
            swept.append(job_id)
            logger.debug("swept expired job %s (%s)", job_id, job.dir)
        return swept

    def recover(self) -> None:
        """Rebuild in-memory jobs from status.json on disk after a restart.

        Terminal jobs are restored as-is; jobs that were mid-flight are marked
        ``interrupted`` so status/QA endpoints report cleanly instead of 404ing.
        """
        recovered = 0
        for status_path in self.root.glob("*/status.json"):
            try:
                data = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                logger.warning("recover: skipping unreadable %s", status_path)
                continue
            job_id = data.get("id")
            if not job_id:
                logger.warning("recover: skipping %s with no job id", status_path)
                continue
            job_dir = status_path.parent
            timeline = None
            if data.get("timeline") is not None:
                try:
                    timeline = Timeline.model_validate(data["timeline"])
                except Exception:
                    logger.warning(
                        "recover: job %s had an invalid timeline, dropping it", job_id
                    )
                    timeline = None
            status = data.get("status", STATUS_INTERRUPTED)
            if status in ACTIVE_STATUSES:
                logger.info(
                    "recover: job %s was mid-flight, marking interrupted", job_id
                )
                status = STATUS_INTERRUPTED
            job = Job(
                id=job_id,
                dir=job_dir,
                status=status,
                timeline=timeline,
                error=data.get("error"),
                events=data.get("events", []),
                created_at=data.get("created_at", time.time()),
                updated_at=data.get("updated_at", time.time()),
            )
            self._jobs[job_id] = job
            recovered += 1
        if recovered:
            logger.info("recover: restored %d job(s) from disk", recovered)

    # -- persistence ---------------------------------------------------------

    def _persist(self, job: Job) -> None:
        snapshot = {
            "id": job.id,
            "status": job.status,
            "error": job.error,
            "events": job.events,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "timeline": (
                json.loads(job.timeline.model_dump_json())
                if job.timeline is not None
                else None
            ),
        }
        tmp = job.dir / "status.json.tmp"
        tmp.write_text(json.dumps(snapshot))
        tmp.replace(job.dir / "status.json")
