"""The pipeline as a Celery canvas.

``pipeline.run_pipeline`` remains the reference implementation and the local /
test path; these tasks run the same stage functions, but each one is a separate
unit of work that can land on a different machine. Every task takes only a
``job_id``, loads what it needs from Postgres and R2, and writes its output back
— so a task can be retried or redelivered without depending on whatever ran
before it still being in memory somewhere.

The shape:

    segment (media)
      └─ replaced by chord(
             group(vision × one per shot (gemini), audio (media)),
             chain(build_timeline, narrate, tts, mux)
         )

The fan-out width is only known once segmentation has found the shots, which is
why ``segment`` rewrites itself with ``self.replace`` instead of the whole
canvas being built up front.

Two things this buys over the single-process pipeline: per-frame vision spreads
across machines rather than four coroutines in one event loop, and the
network-bound vision stage overlaps the CPU-bound audio stages instead of
waiting for them.
"""

import asyncio
import logging
import threading

from celery import chain, chord, group

import job_state
import repo
from audio_extract import AUDIO_KEY, NoAudioStreamError, extract_audio
from blobs import JobBlobs
from celery_app import app
from mux import DESCRIBED_KEY, mux_described_video
from pipeline import frame_event, shots_event
from segmentation import probe_video, segment_video
from timeline import build_timeline
from transcription import transcribe
from tts import synthesize_narration
from vision_analysis import (
    analyze_shots,
    fill_narration_gaps,
    retry_optimize_narration,
)
from voice_activity import detect_speech_regions

logger = logging.getLogger(__name__)

# Stages that talk to the network or to a subprocess get a few attempts; a
# stage that fails on the *content* of the video will fail identically every
# time, so this is deliberately not a catch-all.
RETRY_EXCEPTIONS = (OSError, ConnectionError, TimeoutError)
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 10


def _blobs(job_id: str) -> JobBlobs:
    return JobBlobs.remote(job_id)


async def _emit(job_id: str, event: dict) -> None:
    await repo.append_event(job_id, event)


_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _worker_loop() -> asyncio.AbstractEventLoop:
    """This process's long-lived event loop for async stage bodies.

    Deliberately not ``asyncio.run`` per task: that would build and tear down a
    loop — and with it the asyncpg pool and the Redis client, which are bound to
    the loop that made them — on every task the worker picks up. One loop per
    process keeps those connections alive across tasks.
    """
    global _loop
    if _loop is not None:
        return _loop
    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="adesc-worker-loop", daemon=True
            )
            thread.start()
            _loop = loop
            logger.debug("worker: async loop started")
    return _loop


def _run(coro):
    """Run a stage's async body from a synchronous Celery task."""
    return asyncio.run_coroutine_threadsafe(coro, _worker_loop()).result()


# Per-Gemini-request timeout (ms), so one hung call fails a stage rather than
# holding a worker slot open forever.
GEMINI_REQUEST_TIMEOUT_MS = 120_000

_CLIENT = None


def _gemini_client():
    """The worker process's shared Gemini client, built on first use."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        from google.genai import types

        _CLIENT = genai.Client(
            http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS)
        )
    return _CLIENT


def _run_stage(task, job_id: str, body):
    """Run a stage's async body, reporting a terminal failure on the job.

    Failure reporting is explicit here rather than a canvas errback or a
    Task.on_failure hook. A chord's error callback does not reliably reach its
    *header*, and vision and audio run in the header — so those failures would
    leave the job in "processing" until the reaper timed it out, a silent stall
    instead of a reported error.

    Only the *last* attempt marks the job: autoretry wraps the whole task, so
    an exception here may still be retried and recover.
    """
    try:
        return _run(body())
    except Exception as exc:
        attempts_left = (task.max_retries or 0) - (task.request.retries or 0)
        if attempts_left > 0 and isinstance(exc, RETRY_EXCEPTIONS):
            logger.warning(
                "job %s: stage %s failed, %d attempt(s) left: %r",
                job_id,
                task.name,
                attempts_left,
                exc,
            )
            raise
        logger.error("job %s: stage %s failed: %r", job_id, task.name, exc)
        _run(
            repo.set_status(
                job_id, repo.STATUS_ERROR, error=f"{type(exc).__name__}: {exc}"
            )
        )
        raise


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def enqueue_job(job_id: str) -> None:
    """Kick off the canvas for a job whose source video is already uploaded."""
    segment.apply_async(args=[job_id])


# --------------------------------------------------------------------------- #
# stage 1: segmentation
# --------------------------------------------------------------------------- #


@app.task(
    name="tasks.segment",
    bind=True,
    autoretry_for=RETRY_EXCEPTIONS,
    max_retries=MAX_RETRIES,
    retry_backoff=RETRY_BACKOFF_SEC,
    retry_jitter=True,
)
def segment(self, job_id: str):
    """Detect shots, sample frames, and fan out the rest of the pipeline."""
    blobs = _blobs(job_id)

    async def _body():
        job = await repo.get_job(job_id)
        if job is None or job.source_key is None:
            raise RuntimeError(f"job {job_id} has no source video")

        await repo.set_status(job_id, repo.STATUS_PROCESSING)
        await repo.set_stage(job_id, "segmentation")
        await _emit(
            job_id, {"type": "stage", "stage": "segmentation", "status": "start"}
        )

        source = blobs.fetch(job.source_key)
        meta = probe_video(str(source))
        shots = segment_video(str(source), blobs)
        blobs.put_many([frame["key"] for shot in shots for frame in shot["frames"]])

        job_state.save_shots(blobs, shots, meta)
        await repo.set_duration(job_id, meta["duration_sec"])

        frame_count = sum(len(shot["frames"]) for shot in shots)
        await _emit(job_id, shots_event(shots))
        await _emit(
            job_id,
            {
                "type": "stage",
                "stage": "segmentation",
                "status": "done",
                "count": len(shots),
                "frame_count": frame_count,
            },
        )
        logger.info(
            "job %s: segmentation done, %d shot(s), %d frame(s)",
            job_id,
            len(shots),
            frame_count,
        )
        return shots

    shots = _run_stage(self, job_id, _body)

    # Vision fans out one task per shot — the largest unit a single task can own
    # end to end, so no two tasks ever write the same vision blob. Audio joins
    # the same group so the CPU stages overlap the network ones.
    header = group(
        [vision.si(job_id, shot["id"]) for shot in shots] + [audio.si(job_id)]
    )
    body = chain(
        build_timeline_task.si(job_id),
        narrate.si(job_id),
        tts.si(job_id),
        mux.si(job_id),
    )
    return self.replace(chord(header, body))


# --------------------------------------------------------------------------- #
# stage 2: per-frame vision (Gemini)
# --------------------------------------------------------------------------- #


@app.task(
    name="tasks.vision",
    bind=True,
    autoretry_for=RETRY_EXCEPTIONS,
    max_retries=MAX_RETRIES,
    retry_backoff=RETRY_BACKOFF_SEC,
    retry_jitter=True,
)
def vision(self, job_id: str, shot_id: int):
    """Describe every sampled frame of one shot."""
    blobs = _blobs(job_id)

    async def _body():
        shots, _meta = job_state.load_shots(blobs)
        shot = next((s for s in shots if s["id"] == shot_id), None)
        if shot is None:
            logger.warning("job %s: shot %s vanished before vision", job_id, shot_id)
            return

        async def _on_frame(shot, frame):
            await _emit(job_id, frame_event(shot, frame))

        await analyze_shots([shot], blobs, on_frame=_on_frame, client=_gemini_client())
        job_state.save_shot_vision(blobs, shot)
        await repo.heartbeat(job_id)

    _run_stage(self, job_id, _body)


# --------------------------------------------------------------------------- #
# stages 3-5: audio extraction, VAD, transcription
# --------------------------------------------------------------------------- #


@app.task(
    name="tasks.audio",
    bind=True,
    autoretry_for=RETRY_EXCEPTIONS,
    max_retries=MAX_RETRIES,
    retry_backoff=RETRY_BACKOFF_SEC,
    retry_jitter=True,
)
def audio(self, job_id: str):
    """Pull the soundtrack, find speech, transcribe it.

    A source with no audio track is not an error: the job continues with empty
    speech/transcript, and ad_eligible falls back to shot duration alone.
    """
    blobs = _blobs(job_id)

    async def _body():
        job = await repo.get_job(job_id)
        if job is None or job.source_key is None:
            raise RuntimeError(f"job {job_id} has no source video")

        await repo.set_stage(job_id, "audio")
        await _emit(job_id, {"type": "stage", "stage": "audio", "status": "start"})

        source = blobs.fetch(job.source_key)
        try:
            extract_audio(str(source), out_path=str(blobs.path(AUDIO_KEY)))
        except NoAudioStreamError:
            logger.warning(
                "job %s: no audio stream, skipping VAD + transcription", job_id
            )
            job_state.save_audio(
                blobs, has_audio=False, speech_regions=[], transcript_segments=[]
            )
            await _emit(
                job_id,
                {
                    "type": "stage",
                    "stage": "audio",
                    "status": "done",
                    "has_audio": False,
                },
            )
            return

        blobs.put(AUDIO_KEY)
        speech_regions = detect_speech_regions(str(blobs.path(AUDIO_KEY)))
        transcript_segments = transcribe(str(blobs.path(AUDIO_KEY)), speech_regions)
        job_state.save_audio(blobs, True, speech_regions, transcript_segments)

        await _emit(
            job_id,
            {
                "type": "stage",
                "stage": "audio",
                "status": "done",
                "has_audio": True,
                "speech_regions": len(speech_regions),
                "dialogue_segments": len(transcript_segments),
            },
        )
        await repo.heartbeat(job_id)

    _run_stage(self, job_id, _body)


# --------------------------------------------------------------------------- #
# stage 6a: timeline assembly (the chord callback)
# --------------------------------------------------------------------------- #


@app.task(name="tasks.build_timeline", bind=True)
def build_timeline_task(self, job_id: str):
    """Merge the fanned-out vision results with the audio analysis."""
    blobs = _blobs(job_id)

    async def _body():
        await repo.set_stage(job_id, "timeline")
        await _emit(job_id, {"type": "stage", "stage": "timeline", "status": "start"})

        shots, meta = job_state.load_shots(blobs)
        shots = job_state.load_vision_into(blobs, shots)
        _has_audio, speech_regions, transcript_segments = job_state.load_audio(blobs)

        timeline = build_timeline(
            job_id, meta["duration_sec"], shots, speech_regions, transcript_segments
        )
        await repo.save_timeline(job_id, timeline)

        eligible = sum(1 for s in timeline.segments if s.ad_eligible)
        await _emit(job_id, {"type": "timeline", "timeline": timeline.model_dump()})
        await _emit(job_id, {"type": "stage", "stage": "timeline", "status": "done"})
        logger.info(
            "job %s: timeline done, %d segment(s), %d AD-eligible",
            job_id,
            len(timeline.segments),
            eligible,
        )

    _run_stage(self, job_id, _body)


# --------------------------------------------------------------------------- #
# stage 6b: narration (Gemini, sequential by design)
# --------------------------------------------------------------------------- #


@app.task(
    name="tasks.narrate",
    bind=True,
    autoretry_for=RETRY_EXCEPTIONS,
    max_retries=MAX_RETRIES,
    retry_backoff=RETRY_BACKOFF_SEC,
    retry_jitter=True,
)
def narrate(self, job_id: str):
    """Write an AD line for every eligible segment.

    Stays one task rather than fanning out: each line is written with the
    preceding shots' descriptions, dialogue and narration as context, so the
    calls are inherently ordered.
    """
    blobs = _blobs(job_id)

    async def _body():
        timeline = await repo.get_timeline(job_id)
        if timeline is None:
            raise RuntimeError(f"job {job_id} has no timeline to narrate")

        await repo.set_stage(job_id, "narration")
        await _emit(job_id, {"type": "stage", "stage": "narration", "status": "start"})

        async def _on_segment(segment):
            await repo.heartbeat(job_id)
            await _emit(
                job_id,
                {
                    "type": "narration",
                    "segment_id": segment.id,
                    "text": segment.ad_narration,
                },
            )

        timeline = await fill_narration_gaps(
            timeline, blobs, on_segment=_on_segment, client=_gemini_client()
        )
        await repo.save_timeline(job_id, timeline)
        await _emit(job_id, {"type": "stage", "stage": "narration", "status": "done"})

    _run_stage(self, job_id, _body)


# --------------------------------------------------------------------------- #
# stage 7: TTS
# --------------------------------------------------------------------------- #


@app.task(
    name="tasks.tts",
    bind=True,
    autoretry_for=RETRY_EXCEPTIONS,
    max_retries=MAX_RETRIES,
    retry_backoff=RETRY_BACKOFF_SEC,
    retry_jitter=True,
)
def tts(self, job_id: str):
    """Synthesize each narration line to a WAV clip."""
    blobs = _blobs(job_id)

    async def _body():
        timeline = await repo.get_timeline(job_id)
        if timeline is None:
            raise RuntimeError(f"job {job_id} has no timeline to synthesize")

        await repo.set_stage(job_id, "tts")
        await _emit(job_id, {"type": "stage", "stage": "tts", "status": "start"})

        async def _on_segment(segment):
            blobs.put(segment.ad_narration_key)
            await repo.heartbeat(job_id)
            await _emit(
                job_id,
                {
                    "type": "narration_audio",
                    "segment_id": segment.id,
                    "audio": segment.ad_narration_key,
                    "duration_sec": segment.ad_narration_duration_sec,
                    "overflow": segment.ad_narration_overflow,
                },
            )

        client = _gemini_client()

        async def _retry_optimize(segment, tts_duration, gap_sec):
            return await retry_optimize_narration(
                client, segment.ad_narration, tts_duration, gap_sec
            )

        timeline = await synthesize_narration(
            timeline,
            blobs,
            on_segment=_on_segment,
            retry_optimize=_retry_optimize,
        )
        await repo.save_timeline(job_id, timeline)
        await _emit(job_id, {"type": "stage", "stage": "tts", "status": "done"})

    _run_stage(self, job_id, _body)


# --------------------------------------------------------------------------- #
# stage 8: AD track + mux
# --------------------------------------------------------------------------- #


@app.task(
    name="tasks.mux",
    bind=True,
    autoretry_for=RETRY_EXCEPTIONS,
    max_retries=MAX_RETRIES,
    retry_backoff=RETRY_BACKOFF_SEC,
    retry_jitter=True,
)
def mux(self, job_id: str):
    """Lay the clips into one AD track and mix it into the video."""
    from ad_track import build_ad_track

    blobs = _blobs(job_id)

    async def _body():
        job = await repo.get_job(job_id)
        timeline = await repo.get_timeline(job_id)
        if job is None or job.source_key is None or timeline is None:
            raise RuntimeError(f"job {job_id} is not ready to mux")

        await repo.set_stage(job_id, "mux")
        await _emit(job_id, {"type": "stage", "stage": "mux", "status": "start"})

        result = build_ad_track(timeline, blobs)
        if result is None:
            logger.info("job %s: no narration to mux, skipping described video", job_id)
            await _emit(
                job_id,
                {"type": "stage", "stage": "mux", "status": "done", "described": False},
            )
            await repo.set_status(job_id, repo.STATUS_DONE)
            return

        ad_track_key, ad_track_duration = result
        blobs.put(ad_track_key)
        timeline.ad_track_key = ad_track_key
        timeline.ad_track_duration_sec = ad_track_duration
        await _emit(
            job_id,
            {
                "type": "ad_track",
                "audio": ad_track_key,
                "duration_sec": ad_track_duration,
            },
        )

        source = blobs.fetch(job.source_key)
        mux_described_video(
            source, blobs.fetch(ad_track_key), blobs.path(DESCRIBED_KEY)
        )
        blobs.put(DESCRIBED_KEY)
        timeline.described_key = DESCRIBED_KEY

        await repo.save_timeline(job_id, timeline)
        await _emit(job_id, {"type": "described_video", "video": DESCRIBED_KEY})
        await _emit(
            job_id,
            {"type": "stage", "stage": "mux", "status": "done", "described": True},
        )
        await repo.set_status(job_id, repo.STATUS_DONE)
        logger.info("job %s: pipeline complete", job_id)

    _run_stage(self, job_id, _body)


# --------------------------------------------------------------------------- #
# failure handling + housekeeping
# --------------------------------------------------------------------------- #


@app.task(name="tasks.reap_stale_jobs")
def reap_stale_jobs():
    """Beat task: fail jobs whose worker stopped renewing its lease."""
    reaped = _run(repo.reap_stale_jobs())
    if reaped:
        logger.warning("reaper: marked %d job(s) interrupted: %s", len(reaped), reaped)
    return reaped


# --------------------------------------------------------------------------- #
# Q&A
# --------------------------------------------------------------------------- #


@app.task(name="tasks.answer_question", bind=True)
def answer_question(self, run_id: str, job_id: str):
    """Run the multi-agent Q&A for one question.

    Asynchronous because a run can take minutes — far longer than an HTTP
    request should be held open through a load balancer. The trace streams to
    the browser over the job's existing event channel.
    """
    import qa

    blobs = _blobs(job_id)

    async def _body():
        run = await repo.get_qa_run_unscoped(run_id)
        timeline = await repo.get_timeline(job_id)
        if run is None or timeline is None:
            await repo.finish_qa_run(run_id, "failed", error="job has no timeline")
            return

        await repo.set_qa_run_status(run_id, "running")
        await _emit(
            job_id, {"type": "qa_status", "run_id": run_id, "status": "running"}
        )

        async def _on_step(node: str, records: list[dict]) -> None:
            """Stream the reasoning trace as the graph produces it.

            A run is up to 17 planner cycles, so this is the difference between
            a trace panel that fills in live and one that appears all at once
            several minutes later.
            """
            if records:
                await _emit(
                    job_id,
                    {
                        "type": "qa_trace",
                        "run_id": run_id,
                        "node": node,
                        "records": records,
                    },
                )

        try:
            result = await qa.answer_question(
                timeline, run["question"], _gemini_client(), blobs, on_step=_on_step
            )
        except Exception as exc:  # noqa: BLE001 - recorded on the run, not raised
            logger.exception("qa run %s failed", run_id)
            await repo.finish_qa_run(
                run_id, "failed", error=f"{type(exc).__name__}: {exc}"
            )
            await _emit(
                job_id, {"type": "qa_status", "run_id": run_id, "status": "failed"}
            )
            return

        await repo.finish_qa_run(
            run_id,
            "completed" if result.status == "completed" else "failed",
            answer=result.answer,
            reason=result.reason,
            cycles=result.cycles,
            history=result.history,
        )
        await _emit(
            job_id,
            {
                "type": "qa_result",
                "run_id": run_id,
                "status": result.status,
                "answer": result.answer,
                "cycles": result.cycles,
            },
        )

    # Not _run_stage: a question that fails is recorded on its qa_run, and must
    # not mark the *job* errored — the described video is still perfectly good.
    _run(_body())
