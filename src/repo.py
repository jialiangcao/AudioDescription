"""Job state in Postgres.

Replaces ``jobs.JobStore``: an in-memory dict mirrored to a per-job status.json,
which only worked in one process and rewrote the entire snapshot — timeline
included — on every published event.

Everything here is scoped by ``owner_id`` at the call site even though the
backend connects as the service role, so a missing RLS policy can never become
a cross-tenant read. Queries are plain SQL; there is no ORM.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

import events
from db import get_pool
from timeline import Timeline

logger = logging.getLogger(__name__)

STATUS_CREATED = "created"
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_INTERRUPTED = "interrupted"

ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_PROCESSING)
TERMINAL_STATUSES = (STATUS_DONE, STATUS_ERROR, STATUS_INTERRUPTED)

# Concurrent jobs allowed per user. Was a single global cap (MAX_ACTIVE_JOBS)
# back when there was one process and one anonymous user.
MAX_ACTIVE_JOBS_PER_USER = 3

# A processing job whose worker hasn't checked in for this long is presumed
# dead and gets requeued. Must comfortably exceed the slowest single stage.
HEARTBEAT_TIMEOUT_SEC = 600


@dataclass
class Job:
    id: str
    owner_id: str
    status: str
    stage: str | None
    error: str | None
    filename: str | None
    source_key: str | None
    duration_sec: float | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row) -> "Job":
        return cls(
            id=str(row["id"]),
            owner_id=str(row["owner_id"]),
            status=row["status"],
            stage=row["stage"],
            error=row["error"],
            filename=row["filename"],
            source_key=row["source_key"],
            duration_sec=row["duration_sec"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


_JOB_COLUMNS = (
    "id, owner_id, status, stage, error, filename, source_key, duration_sec, "
    "created_at, updated_at"
)


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


async def create_job(owner_id: str, filename: str | None, source_key: str) -> Job:
    pool = await get_pool()
    row = await pool.fetchrow(
        f"""
        insert into jobs (owner_id, status, filename, source_key)
        values ($1, '{STATUS_CREATED}', $2, $3)
        returning {_JOB_COLUMNS}
        """,
        uuid.UUID(owner_id),
        filename,
        source_key,
    )
    logger.info("job %s created for owner %s", row["id"], owner_id)
    return Job.from_row(row)


async def get_job(job_id: str, owner_id: str | None = None) -> Job | None:
    """One job. Pass ``owner_id`` on any path reachable from a request."""
    pool = await get_pool()
    if owner_id is None:
        row = await pool.fetchrow(
            f"select {_JOB_COLUMNS} from jobs where id = $1", uuid.UUID(job_id)
        )
    else:
        row = await pool.fetchrow(
            f"select {_JOB_COLUMNS} from jobs where id = $1 and owner_id = $2",
            uuid.UUID(job_id),
            uuid.UUID(owner_id),
        )
    return Job.from_row(row) if row is not None else None


async def list_jobs(owner_id: str, limit: int = 50) -> list[Job]:
    pool = await get_pool()
    rows = await pool.fetch(
        f"""
        select {_JOB_COLUMNS} from jobs
        where owner_id = $1
        order by created_at desc
        limit $2
        """,
        uuid.UUID(owner_id),
        limit,
    )
    return [Job.from_row(row) for row in rows]


async def active_count(owner_id: str) -> int:
    pool = await get_pool()
    return await pool.fetchval(
        "select count(*) from jobs where owner_id = $1 and status = any($2::text[])",
        uuid.UUID(owner_id),
        list(ACTIVE_STATUSES),
    )


async def at_capacity(owner_id: str) -> bool:
    return await active_count(owner_id) >= MAX_ACTIVE_JOBS_PER_USER


async def set_status(job_id: str, status: str, error: str | None = None) -> None:
    """Move a job to ``status`` and publish the change as an event."""
    pool = await get_pool()
    await pool.execute(
        """
        update jobs
        set status = $2,
            error = coalesce($3, error),
            heartbeat_at = case when $2 = 'processing' then now() else null end
        where id = $1
        """,
        uuid.UUID(job_id),
        status,
        error,
    )
    if status == STATUS_ERROR:
        logger.error("job %s -> %s: %s", job_id, status, error)
    else:
        logger.info("job %s -> %s", job_id, status)
    await append_event(job_id, {"type": "status", "status": status, "error": error})


async def set_stage(job_id: str, stage: str) -> None:
    pool = await get_pool()
    await pool.execute(
        "update jobs set stage = $2, heartbeat_at = now() where id = $1",
        uuid.UUID(job_id),
        stage,
    )


async def set_duration(job_id: str, duration_sec: float) -> None:
    pool = await get_pool()
    await pool.execute(
        "update jobs set duration_sec = $2 where id = $1",
        uuid.UUID(job_id),
        duration_sec,
    )


async def heartbeat(job_id: str) -> None:
    """Renew a worker's lease on a job. See ``reap_stale_jobs``."""
    pool = await get_pool()
    await pool.execute(
        "update jobs set heartbeat_at = now() where id = $1", uuid.UUID(job_id)
    )


async def reap_stale_jobs(timeout_sec: int = HEARTBEAT_TIMEOUT_SEC) -> list[str]:
    """Fail jobs whose worker stopped checking in, and return their ids.

    The old startup ``recover()`` could only find these on restart, and only
    for the one process that owned them; a crashed worker's job stayed
    "processing" forever. Runs periodically from Celery beat instead.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        update jobs
        set status = $2, error = 'worker stopped responding'
        where status = 'processing'
          and heartbeat_at is not null
          and heartbeat_at < now() - make_interval(secs => $1)
        returning id
        """,
        float(timeout_sec),
        STATUS_INTERRUPTED,
    )
    job_ids = [str(row["id"]) for row in rows]
    for job_id in job_ids:
        logger.warning("job %s: lease expired, marked interrupted", job_id)
        await append_event(
            job_id,
            {
                "type": "status",
                "status": STATUS_INTERRUPTED,
                "error": "worker stopped responding",
            },
        )
    return job_ids


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #


async def append_event(job_id: str, event: dict) -> int:
    """Durably record an event, then fan it out live. Returns its seq.

    The sequence is allocated under a per-job advisory lock rather than by
    retrying on the (job_id, seq) primary key: the vision stage publishes one
    event per frame from several workers at once, so contention is the normal
    case, not the exception.
    """
    pool = await get_pool()
    job_uuid = uuid.UUID(job_id)
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "select pg_advisory_xact_lock(hashtextextended($1::text, 0))", job_id
        )
        seq = await conn.fetchval(
            """
            insert into job_events (job_id, seq, type, payload)
            select $1, coalesce(max(seq), 0) + 1, $2, $3::jsonb
            from job_events where job_id = $1
            returning seq
            """,
            job_uuid,
            event.get("type", "unknown"),
            json.dumps(event),
        )
    await events.publish(job_id, {**event, "seq": seq})
    return seq


async def events_since(job_id: str, since_seq: int = 0) -> list[dict]:
    """Every event after ``since_seq``, oldest first — the reconnect replay.

    The old store could only replay from zero, so a client that dropped near
    the end of a long job re-received the whole log.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        select seq, payload from job_events
        where job_id = $1 and seq > $2
        order by seq
        """,
        uuid.UUID(job_id),
        since_seq,
    )
    return [{**json.loads(row["payload"]), "seq": row["seq"]} for row in rows]


# --------------------------------------------------------------------------- #
# timelines
# --------------------------------------------------------------------------- #


async def save_timeline(job_id: str, timeline: Timeline) -> None:
    pool = await get_pool()
    await pool.execute(
        """
        insert into timelines (job_id, doc) values ($1, $2::jsonb)
        on conflict (job_id) do update set doc = excluded.doc, updated_at = now()
        """,
        uuid.UUID(job_id),
        timeline.model_dump_json(),
    )


async def get_timeline(job_id: str) -> Timeline | None:
    pool = await get_pool()
    doc = await pool.fetchval(
        "select doc from timelines where job_id = $1", uuid.UUID(job_id)
    )
    if doc is None:
        return None
    return Timeline.model_validate_json(doc)


# --------------------------------------------------------------------------- #
# qa_runs
# --------------------------------------------------------------------------- #


async def create_qa_run(job_id: str, owner_id: str, question: str) -> str:
    pool = await get_pool()
    run_id = await pool.fetchval(
        """
        insert into qa_runs (job_id, owner_id, question)
        values ($1, $2, $3)
        returning id
        """,
        uuid.UUID(job_id),
        uuid.UUID(owner_id),
        question,
    )
    return str(run_id)


async def get_qa_run(run_id: str, owner_id: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        select id, job_id, question, status, answer, reason, cycles, history, error
        from qa_runs where id = $1 and owner_id = $2
        """,
        uuid.UUID(run_id),
        uuid.UUID(owner_id),
    )
    if row is None:
        return None
    result = dict(row)
    result["id"] = str(result["id"])
    result["job_id"] = str(result["job_id"])
    if result["history"] is not None:
        result["history"] = json.loads(result["history"])
    return result


async def get_qa_run_unscoped(run_id: str) -> dict | None:
    """A run without an owner check — for the worker that was handed its id.

    The API only ever reaches runs through ``get_qa_run``, which is scoped; a
    Celery task has already been authorized by whoever enqueued it and has no
    user context of its own.
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        "select id, job_id, owner_id, question, status from qa_runs where id = $1",
        uuid.UUID(run_id),
    )
    if row is None:
        return None
    result = dict(row)
    result["id"] = str(result["id"])
    result["job_id"] = str(result["job_id"])
    result["owner_id"] = str(result["owner_id"])
    return result


async def set_qa_run_status(run_id: str, status: str) -> None:
    pool = await get_pool()
    await pool.execute(
        "update qa_runs set status = $2 where id = $1", uuid.UUID(run_id), status
    )


async def finish_qa_run(
    run_id: str,
    status: str,
    answer: str | None = None,
    reason: str | None = None,
    cycles: int | None = None,
    history: list[dict] | None = None,
    error: str | None = None,
) -> None:
    pool = await get_pool()
    await pool.execute(
        """
        update qa_runs
        set status = $2, answer = $3, reason = $4, cycles = $5,
            history = $6::jsonb, error = $7
        where id = $1
        """,
        uuid.UUID(run_id),
        status,
        answer,
        reason,
        cycles,
        json.dumps(history) if history is not None else None,
        error,
    )
