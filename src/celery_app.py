"""Celery application: queues, routing, retry defaults, and beat schedule.

Three queues, because the stages are not alike:

``media``   ffmpeg, PySceneDetect, OpenCV, Silero, faster-whisper, Kokoro.
            CPU-bound and memory-hungry, needs the fat image and a scratch
            volume. Runs at concurrency 1 — the Kokoro pipeline is a single
            shared, non-reentrant model instance.
``gemini``  Per-frame vision and narration. Pure network I/O, so it runs at
            high concurrency on a slim image with no torch in it at all.
``qa``      The multi-agent Q&A run. Gemini-bound like ``gemini``, but its
            CLIP retriever needs torch, so it is served by the fat image.
            (Precomputing frame embeddings into pgvector would free it to
            join the slim pool; see the plan's Phase 2 note.)

Splitting them is what lets the expensive pool stay small while the cheap,
latency-bound one scales out.
"""

import logging
import os

from celery import Celery
from celery.signals import worker_process_init

import events

logger = logging.getLogger(__name__)

QUEUE_MEDIA = "media"
QUEUE_GEMINI = "gemini"
QUEUE_QA = "qa"

# How often the reaper looks for jobs whose worker stopped checking in.
REAP_INTERVAL_SEC = 120

app = Celery(
    "adesc",
    broker=events.redis_url(),
    backend=events.redis_url(),
    include=["tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Deliver one task at a time: these run for minutes, so prefetching would
    # leave work queued behind a busy worker while another sits idle.
    worker_prefetch_multiplier=1,
    # Ack after the task finishes, not on receipt, so a machine that dies
    # mid-stage puts its task back on the queue instead of dropping it.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Results are only used to sequence the canvas; the real output is in
    # Postgres and R2, so they need not outlive a job.
    result_expires=3600,
    # Without this, a chord's error callback covers only its body — a failing
    # vision or audio task would never reach tasks.job_failed, and the job
    # would sit in "processing" until the reaper eventually timed it out.
    task_allow_error_cb_on_chord_header=True,
    broker_transport_options={
        # Must exceed the slowest single stage, or Redis will redeliver a task
        # that is still running and the job will be processed twice.
        "visibility_timeout": 3600,
    },
    task_routes={
        "tasks.fetch_source": {"queue": QUEUE_MEDIA},
        "tasks.segment": {"queue": QUEUE_MEDIA},
        "tasks.audio": {"queue": QUEUE_MEDIA},
        "tasks.build_timeline": {"queue": QUEUE_MEDIA},
        "tasks.tts": {"queue": QUEUE_MEDIA},
        "tasks.mux": {"queue": QUEUE_MEDIA},
        "tasks.vision": {"queue": QUEUE_GEMINI},
        "tasks.narrate": {"queue": QUEUE_GEMINI},
        "tasks.answer_question": {"queue": QUEUE_QA},
        "tasks.reap_stale_jobs": {"queue": QUEUE_MEDIA},
    },
    beat_schedule={
        "reap-stale-jobs": {
            "task": "tasks.reap_stale_jobs",
            "schedule": float(REAP_INTERVAL_SEC),
        },
    },
)


@worker_process_init.connect
def _configure_worker(**_kwargs) -> None:
    """Per-worker-process setup.

    Pinning torch's intra-op threads matters on the media queue: Silero and
    faster-whisper each parallelize internally, so without this several
    processes on one machine oversubscribe its cores and all of them slow
    down. Guarded because the slim image has no torch at all.
    """
    from log_config import configure_logging

    configure_logging()
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        logger.debug("worker init: no torch in this image, skipping thread pinning")

    logger.info(
        "celery worker process ready (queues=%s)", os.environ.get("QUEUES", "-")
    )
