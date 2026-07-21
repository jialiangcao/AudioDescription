"""FastAPI backend for the adesc web app.

Serves the drag-and-drop upload → live audio-description → Q&A flow. Jobs run
in-process: uploads are queued onto an ``asyncio.Queue`` drained by a small
fixed pool of worker tasks, progress is streamed over a websocket, and per-job
temp dirs hold the keyframes that Q&A reads afterward.

Run with (preserving the flat-import convention that puts ``src/`` on sys.path):

    uv run uvicorn server:app --app-dir src --reload --reload-dir src

Single worker process only — the in-memory JobStore is not shared across
workers, so never pass ``--workers > 1``.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

import qa
from jobs import (
    JOB_TTL_SEC,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PROCESSING,
    TERMINAL_STATUSES,
    Job,
    JobStore,
)
from pipeline import run_pipeline
from timeline import Timeline

load_dotenv()

# Number of jobs processed concurrently. Kept small on purpose: the CPU-bound
# stages (VAD/Whisper/OpenCV) contend for cores, and torch intra-op threads are
# pinned to 1 (below) so this is the only real parallelism knob.
NUM_WORKERS = 2

# Per-Gemini-request timeout (ms) so a single hung call fails the job instead
# of pinning a worker slot forever.
GEMINI_REQUEST_TIMEOUT_MS = 120_000

# How often the background sweep runs to reclaim expired job dirs.
SWEEP_INTERVAL_SEC = 300

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

UPLOAD_CHUNK = 1024 * 1024


def _get_pipeline_client(app: FastAPI) -> genai.Client:
    """Lazily build the shared Gemini client used by the pipeline.

    Built on first job (not at startup) so the server can boot and serve
    status/frames endpoints even without a configured GEMINI_API_KEY.
    """
    client = getattr(app.state, "gemini_client", None)
    if client is None:
        client = genai.Client(
            http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS)
        )
        app.state.gemini_client = client
    return client


async def _process_job(app: FastAPI, job_id: str) -> None:
    store: JobStore = app.state.store
    job = store.get(job_id)
    if job is None:
        return
    video = job.source_video
    if video is None:
        store.set_status(job_id, STATUS_ERROR, error="uploaded video missing")
        return

    store.set_status(job_id, STATUS_PROCESSING)

    async def on_event(event: dict) -> None:
        if event.get("type") == "timeline":
            with contextlib.suppress(Exception):
                store.set_timeline(job_id, Timeline.model_validate(event["timeline"]))
        store.publish(job_id, event)

    try:
        client = _get_pipeline_client(app)
        timeline = await run_pipeline(video, job.dir, on_event, client=client)
        store.set_timeline(job_id, timeline)
        store.set_status(job_id, STATUS_DONE)
    except Exception as exc:  # noqa: BLE001 - surface any stage failure to the client
        store.set_status(job_id, STATUS_ERROR, error=f"{type(exc).__name__}: {exc}")


async def _worker(app: FastAPI) -> None:
    queue: asyncio.Queue = app.state.job_queue
    while True:
        job_id = await queue.get()
        try:
            await _process_job(app, job_id)
        finally:
            queue.task_done()


async def _sweeper(app: FastAPI) -> None:
    store: JobStore = app.state.store
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SEC)
        with contextlib.suppress(Exception):
            store.sweep(ttl_sec=JOB_TTL_SEC)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Pin torch intra-op threads: Silero VAD and faster-whisper each parallelize
    # internally, so without this concurrent jobs oversubscribe cores.
    torch.set_num_threads(1)

    store = JobStore()
    store.recover()
    app.state.store = store
    app.state.job_queue = asyncio.Queue()
    app.state.gemini_client = None

    tasks = [asyncio.create_task(_worker(app)) for _ in range(NUM_WORKERS)]
    tasks.append(asyncio.create_task(_sweeper(app)))
    app.state.background_tasks = tasks
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="adesc", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_job(app: FastAPI, job_id: str) -> Job:
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile) -> dict:
    store: JobStore = app.state.store
    if store.at_capacity():
        raise HTTPException(status_code=503, detail="server busy, too many active jobs")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400, detail=f"unsupported video type: {suffix or 'unknown'}"
        )

    job = store.create()
    dest = job.dir / f"source{suffix}"
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                out.write(chunk)
    finally:
        await file.close()

    await app.state.job_queue.put(job.id)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = _require_job(app, job_id)
    return {
        "id": job.id,
        "status": job.status,
        "error": job.error,
        "timeline": job.timeline.model_dump() if job.timeline is not None else None,
    }


@app.get("/api/jobs/{job_id}/frames/{filename}")
async def get_frame(job_id: str, filename: str) -> FileResponse:
    job = _require_job(app, job_id)
    if not filename.endswith(".jpg"):
        raise HTTPException(status_code=404, detail="not found")

    frames_dir = (job.dir / "frames").resolve()
    target = (frames_dir / filename).resolve()
    if not target.is_relative_to(frames_dir) or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/narration/{filename}")
async def get_narration(job_id: str, filename: str) -> FileResponse:
    job = _require_job(app, job_id)
    if not filename.endswith(".wav"):
        raise HTTPException(status_code=404, detail="not found")

    narration_dir = (job.dir / "narration").resolve()
    target = (narration_dir / filename).resolve()
    if not target.is_relative_to(narration_dir) or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target, media_type="audio/wav")


@app.post("/api/jobs/{job_id}/ask")
async def ask(job_id: str, body: AskRequest) -> AskResponse:
    job = _require_job(app, job_id)
    if not job.dir.exists():
        raise HTTPException(status_code=410, detail="job data expired")
    if job.status != STATUS_DONE or job.timeline is None:
        raise HTTPException(status_code=409, detail="job not finished")

    try:
        answer = await asyncio.to_thread(
            qa.answer_question, job.timeline, body.question
        )
    except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the client
        raise HTTPException(
            status_code=500, detail=f"could not answer question: {exc}"
        ) from exc
    return AskResponse(answer=answer)


@app.websocket("/api/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    store: JobStore = websocket.app.state.store
    if store.get(job_id) is None:
        await websocket.close(code=4004)
        return

    queue: asyncio.Queue = asyncio.Queue()
    store.subscribe(job_id, queue)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if (
                event.get("type") == "status"
                and event.get("status") in TERMINAL_STATUSES
            ):
                break
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe(job_id, queue)
        with contextlib.suppress(RuntimeError):
            await websocket.close()
