"""FastAPI backend for the adesc web app.

Stateless: every request resolves a Supabase JWT to a user id, reads and writes
job state in Postgres, and hands media to the browser as presigned R2 URLs. No
job state, no worker pool and no uploaded bytes live in this process, so it can
run as many replicas as it likes behind a load balancer — unlike the previous
version, which drained an in-process asyncio.Queue and served files off its own
disk.

Run with (preserving the flat-import convention that puts ``src/`` on sys.path):

    uv run uvicorn server:app --app-dir src --reload --reload-dir src
"""

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db
import events
import repo
import tickets
from auth import current_user, warn_if_insecure
from blobs import JobBlobs
from log_config import configure_logging
from tasks import answer_question as answer_question_task
from tasks import enqueue_job
from timeline import Timeline

load_dotenv()
configure_logging()

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

# Refused before a presigned URL is even issued, and re-checked against the
# object's real size before the job is enqueued — the browser uploads straight
# to R2, so the API never sees the bytes and cannot enforce this mid-transfer.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

# Long enough for a big upload over a slow connection.
UPLOAD_URL_TTL_SEC = 3600

# Media URLs handed to the browser, re-issued on every status read.
MEDIA_URL_TTL_SEC = 3600

# Cap on one presign batch, so a single request cannot be used to generate an
# unbounded number of signed URLs.
MAX_PRESIGN_BATCH = 500

# How often an idle websocket pings, so a dead peer is noticed and the
# connection does not sit open through an idle-timeout proxy.
WS_PING_INTERVAL_SEC = 25


def _suffix_of(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    warn_if_insecure()
    logger.info("adesc api up")
    try:
        yield
    finally:
        await db.close_pool()
        await events.close_redis()


app = FastAPI(title="adesc", lifespan=lifespan)


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    # Vercel gives every preview deploy a generated hostname, so they cannot be
    # enumerated in ALLOWED_ORIGINS. Set ALLOWED_ORIGIN_REGEX to match them —
    # e.g. ^https://adesc-[a-z0-9-]+\.vercel\.app$ — and leave it unset in
    # production, where the origin list is known and should stay exact.
    allow_origin_regex=os.environ.get("ALLOWED_ORIGIN_REGEX") or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# request/response models
# --------------------------------------------------------------------------- #


class CreateJobRequest(BaseModel):
    filename: str
    content_type: str | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    upload_url: str
    key: str


class MediaUrlsRequest(BaseModel):
    keys: list[str]


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    """A question is answered asynchronously; poll or watch the event stream."""

    run_id: str
    status: str


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


async def _require_job(job_id: str, user_id: str) -> repo.Job:
    try:
        job = await repo.get_job(job_id, owner_id=user_id)
    except ValueError as exc:  # a malformed id is simply not a job we have
        raise HTTPException(status_code=404, detail="job not found") from exc
    if job is None:
        # Deliberately the same response as someone else's job, so job ids
        # cannot be probed for existence.
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _presign_timeline(timeline: Timeline, job_id: str) -> dict:
    """Serialize a timeline with every blob key replaced by a fetchable URL.

    The browser talks to R2 directly, so media bytes never pass through the
    API — and the stored keys stay opaque rather than being exposed as paths.
    """
    blobs = JobBlobs.remote(job_id)
    doc = timeline.model_dump()

    def _url(key: str | None) -> str | None:
        if not key:
            return None
        try:
            return blobs.presign_get(key, ttl_sec=MEDIA_URL_TTL_SEC)
        except RuntimeError:
            # No bucket configured (local runs); leave the key for the client.
            return key

    for segment in doc["segments"]:
        for frame in segment["frames"]:
            frame["url"] = _url(frame["key"])
        segment["ad_narration_url"] = _url(segment.get("ad_narration_key"))
    doc["ad_track_url"] = _url(doc.get("ad_track_key"))
    doc["described_url"] = _url(doc.get("described_key"))
    return doc


def _is_terminal(event: dict) -> bool:
    return (
        event.get("type") == "status" and event.get("status") in repo.TERMINAL_STATUSES
    )


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness: the process is up. Deliberately touches nothing else."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict:
    """Readiness: this replica can actually serve traffic."""
    checks = {}
    try:
        pool = await db.get_pool()
        await pool.fetchval("select 1")
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        checks["postgres"] = f"error: {exc}"
    try:
        await events.get_redis().ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"

    if any(value != "ok" for value in checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ok", **checks}


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


@app.post("/api/jobs", status_code=201)
async def create_job(
    body: CreateJobRequest, user_id: str = Depends(current_user)
) -> CreateJobResponse:
    """Reserve a job and hand back a presigned URL to PUT the video to.

    The browser uploads straight to R2: the API never touches the bytes, so a
    large upload neither occupies a request nor lands on this machine's disk.
    """
    if await repo.at_capacity(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"you already have {repo.MAX_ACTIVE_JOBS_PER_USER} jobs in flight",
        )

    suffix = _suffix_of(body.filename)
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400, detail=f"unsupported video type: {suffix or 'unknown'}"
        )

    source_key = f"source{suffix}"
    job = await repo.create_job(user_id, body.filename, source_key)
    blobs = JobBlobs.remote(job.id)
    try:
        upload_url = blobs.presign_put(
            source_key, ttl_sec=UPLOAD_URL_TTL_SEC, content_type=body.content_type
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="object storage unavailable"
        ) from exc

    logger.info("job %s: reserved for %s (%s)", job.id, user_id, body.filename)
    return CreateJobResponse(job_id=job.id, upload_url=upload_url, key=source_key)


@app.post("/api/jobs/{job_id}/start", status_code=202)
async def start_job(job_id: str, user_id: str = Depends(current_user)) -> dict:
    """Enqueue a job once its source video has landed in the bucket."""
    job = await _require_job(job_id, user_id)
    if job.status != repo.STATUS_CREATED:
        raise HTTPException(status_code=409, detail=f"job is already {job.status}")

    blobs = JobBlobs.remote(job_id)
    size = blobs.size(job.source_key or "")
    if size is None:
        raise HTTPException(status_code=400, detail="no video was uploaded")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"video is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )

    await repo.set_status(job_id, repo.STATUS_QUEUED)
    enqueue_job(job_id)
    logger.info("job %s: queued (%d bytes)", job_id, size)
    return {"job_id": job_id, "status": repo.STATUS_QUEUED}


@app.get("/api/jobs")
async def list_jobs(user_id: str = Depends(current_user)) -> dict:
    jobs = await repo.list_jobs(user_id)
    return {
        "jobs": [
            {
                "id": job.id,
                "status": job.status,
                "stage": job.stage,
                "filename": job.filename,
                "duration_sec": job.duration_sec,
                "created_at": job.created_at.isoformat(),
            }
            for job in jobs
        ]
    }


@app.post("/api/jobs/{job_id}/media-urls")
async def media_urls(
    job_id: str, body: MediaUrlsRequest, user_id: str = Depends(current_user)
) -> dict:
    """Presign a batch of this job's blob keys.

    Progress events carry keys, not URLs — a presigned URL expires, and the
    event log is durable and replayed on reconnect, so baking one in would mean
    replaying dead links. The client asks for URLs when it needs them instead.

    ``<img>``/``<audio>``/``<video>`` cannot send an Authorization header, which
    is why these have to be self-authenticating URLs rather than an API route
    that streams or redirects.
    """
    await _require_job(job_id, user_id)
    if len(body.keys) > MAX_PRESIGN_BATCH:
        raise HTTPException(
            status_code=400, detail=f"at most {MAX_PRESIGN_BATCH} keys per request"
        )

    blobs = JobBlobs.remote(job_id)
    urls: dict[str, str] = {}
    for key in body.keys:
        # Keys come from the client, so refuse anything that could address
        # another job's objects.
        if key.startswith("/") or ".." in key:
            continue
        with contextlib.suppress(RuntimeError):
            urls[key] = blobs.presign_get(key, ttl_sec=MEDIA_URL_TTL_SEC)
    return {"urls": urls}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, user_id: str = Depends(current_user)) -> dict:
    job = await _require_job(job_id, user_id)
    timeline = await repo.get_timeline(job_id)
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "error": job.error,
        "timeline": _presign_timeline(timeline, job_id) if timeline else None,
    }


# --------------------------------------------------------------------------- #
# live progress
# --------------------------------------------------------------------------- #


@app.post("/api/jobs/{job_id}/ws-ticket")
async def create_ws_ticket(job_id: str, user_id: str = Depends(current_user)) -> dict:
    """A short-lived, single-use credential for the event websocket."""
    await _require_job(job_id, user_id)
    return {"ticket": await tickets.issue(user_id, job_id)}


@app.websocket("/api/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    """Replay everything missed, then stream live until the job is terminal.

    ``?since=`` is the last sequence the client already has, so a reconnect
    resumes instead of replaying the whole log — the previous implementation
    could only replay from zero. Live events arrive over Redis pub/sub, so it
    does not matter which replica holds the socket or which worker produced
    the event.
    """
    redeemed = await tickets.redeem(websocket.query_params.get("ticket", ""))
    if redeemed is None or redeemed[1] != job_id:
        await websocket.close(code=4401)
        return
    user_id, _ = redeemed

    try:
        job = await repo.get_job(job_id, owner_id=user_id)
    except ValueError:
        job = None
    if job is None:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    try:
        since = int(websocket.query_params.get("since", "0"))
    except ValueError:
        since = 0

    pubsub = events.get_redis().pubsub()
    # Subscribe *before* replaying, so an event published during the replay is
    # buffered rather than lost in the gap between the two.
    await pubsub.subscribe(events.channel(job_id))
    try:
        seen = since
        for event in await repo.events_since(job_id, since):
            await websocket.send_json(event)
            seen = event.get("seq", seen)
            if _is_terminal(event):
                return

        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=WS_PING_INTERVAL_SEC
            )
            if message is None:
                # Also how a disconnected client is noticed: send_json raises.
                await websocket.send_json({"type": "ping"})
                continue
            event = json.loads(message["data"])
            # Replay and live stream overlap by design; drop the duplicates.
            if event.get("seq", 0) <= seen:
                continue
            seen = event.get("seq", seen)
            await websocket.send_json(event)
            if _is_terminal(event):
                return
    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.debug("job %s: websocket closed by client", job_id)
    except Exception:
        logger.exception("job %s: websocket stream failed", job_id)
    finally:
        with contextlib.suppress(Exception):
            await pubsub.unsubscribe(events.channel(job_id))
            await pubsub.aclose()
        with contextlib.suppress(Exception):
            await websocket.close()


# --------------------------------------------------------------------------- #
# Q&A
# --------------------------------------------------------------------------- #


@app.post("/api/jobs/{job_id}/ask", status_code=202)
async def ask(
    job_id: str, body: AskRequest, user_id: str = Depends(current_user)
) -> AskResponse:
    """Queue a question. A run takes minutes, so it is not answered inline."""
    job = await _require_job(job_id, user_id)
    if job.status != repo.STATUS_DONE:
        raise HTTPException(status_code=409, detail="job not finished")
    if await repo.get_timeline(job_id) is None:
        raise HTTPException(status_code=410, detail="job data expired")

    run_id = await repo.create_qa_run(job_id, user_id, body.question)
    answer_question_task.apply_async(args=[run_id, job_id])
    logger.info("job %s: queued Q&A run %s", job_id, run_id)
    return AskResponse(run_id=run_id, status="queued")


@app.get("/api/qa/{run_id}")
async def get_qa_run(run_id: str, user_id: str = Depends(current_user)) -> dict:
    try:
        run = await repo.get_qa_run(run_id, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run
