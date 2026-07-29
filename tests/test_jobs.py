import asyncio
import json

from jobs import (
    MAX_ACTIVE_JOBS,
    STATUS_DONE,
    STATUS_INTERRUPTED,
    STATUS_PROCESSING,
    Job,
    JobStore,
)
from timeline import AudioAnalysis, Frame, FrameAnalysis, Segment, Timeline


def _store(tmp_path) -> JobStore:
    return JobStore(root=tmp_path / "adesc-jobs")


def _timeline() -> Timeline:
    return Timeline(
        video_id="v.mp4",
        duration_sec=2.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=2.0,
                frames=[
                    Frame(
                        index=0,
                        time=0.0,
                        path="shot_0000_00.jpg",
                        visual=FrameAnalysis(
                            description="d",
                            entities=[],
                            actions=[],
                            setting="s",
                            on_screen_text=None,
                        ),
                    )
                ],
                audio=AudioAnalysis(
                    has_speech=False, transcript=None, silence_ratio=1.0
                ),
                ad_eligible=True,
                narratable_gap_sec=2.0,
            )
        ],
    )


def test_create_makes_dir_and_persists_status(tmp_path):
    store = _store(tmp_path)
    job = store.create()

    assert job.dir.is_dir()
    status = json.loads((job.dir / "status.json").read_text())
    assert status["id"] == job.id
    assert status["status"] == "queued"


def test_admission_cap(tmp_path):
    store = _store(tmp_path)
    for _ in range(MAX_ACTIVE_JOBS):
        assert not store.at_capacity()
        store.create()
    assert store.at_capacity()


def test_at_capacity_ignores_terminal_jobs(tmp_path):
    store = _store(tmp_path)
    for _ in range(MAX_ACTIVE_JOBS):
        job = store.create()
        store.set_status(job.id, STATUS_DONE)
    # all done -> no longer active, so capacity is free again
    assert not store.at_capacity()


async def test_subscribe_replays_then_streams_live(tmp_path):
    store = _store(tmp_path)
    job = store.create()
    store.publish(job.id, {"type": "shot", "shot": {"id": 0}})

    queue: asyncio.Queue = asyncio.Queue()
    store.subscribe(job.id, queue)

    # replayed history first
    replayed = queue.get_nowait()
    assert replayed == {"type": "shot", "shot": {"id": 0}}

    # then live events
    store.publish(job.id, {"type": "shot", "shot": {"id": 1}})
    live = queue.get_nowait()
    assert live == {"type": "shot", "shot": {"id": 1}}


async def test_multiple_subscribers_each_get_events(tmp_path):
    store = _store(tmp_path)
    job = store.create()

    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()
    store.subscribe(job.id, q1)
    store.subscribe(job.id, q2)

    store.publish(job.id, {"type": "status", "status": "done", "error": None})

    assert q1.get_nowait()["status"] == "done"
    assert q2.get_nowait()["status"] == "done"


def test_sweep_removes_expired_terminal_jobs_only(tmp_path):
    store = _store(tmp_path)
    done = store.create()
    store.set_status(done.id, STATUS_DONE)
    active = store.create()
    store.set_status(active.id, STATUS_PROCESSING)

    # nothing expired yet
    assert store.sweep(ttl_sec=1000, now=done.updated_at + 1) == []

    swept = store.sweep(ttl_sec=10, now=done.updated_at + 100)
    assert swept == [done.id]
    assert not done.dir.exists()
    # active job untouched even though it is old
    assert store.get(active.id) is not None
    assert active.dir.exists()


def test_recover_marks_inflight_as_interrupted(tmp_path):
    root = tmp_path / "adesc-jobs"
    store = _store(tmp_path)
    processing = store.create()
    store.set_status(processing.id, STATUS_PROCESSING)
    finished = store.create()
    store.set_timeline(finished.id, _timeline())
    store.set_status(finished.id, STATUS_DONE)

    # simulate restart: a brand-new store reading the same dir
    recovered = JobStore(root=root)
    recovered.recover()

    processing_job = recovered.get(processing.id)
    assert processing_job is not None
    assert processing_job.status == STATUS_INTERRUPTED
    done_job = recovered.get(finished.id)
    assert done_job is not None
    assert done_job.status == STATUS_DONE
    assert done_job.timeline is not None
    assert done_job.timeline.segments[0].id == 0


def test_source_video_property(tmp_path):
    store = _store(tmp_path)
    job = store.create()
    (job.dir / "source.mp4").write_bytes(b"data")
    assert job.source_video == job.dir / "source.mp4"


def test_job_source_video_none_when_missing(tmp_path):
    job = Job(id="x", dir=tmp_path)
    assert job.source_video is None
