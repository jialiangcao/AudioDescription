"""The Celery canvas, run eagerly against a real Postgres.

Only the external ML/API boundaries are stubbed (Gemini, VAD, Whisper, Kokoro),
exactly as tests/test_e2e.py does for the single-process pipeline — so these
exercise the real stage handoff: what each task writes to blobs and Postgres,
and what the next one reads back.
"""

import uuid
from pathlib import Path

import pytest

import job_state
import repo
import tasks
from celery_app import app as celery_app
from timeline import MIN_NARRATABLE_GAP_SEC


@pytest.fixture
def eager(monkeypatch):
    """Run tasks inline, so a canvas executes end to end in-process.

    An in-memory result backend replaces Redis: chords consult the backend to
    know when their header is done, and these tests are about the canvas's
    shape and stage handoff, not about the broker.
    """
    monkeypatch.setattr(celery_app.conf, "task_always_eager", True)
    monkeypatch.setattr(celery_app.conf, "task_eager_propagates", True)
    monkeypatch.setattr(celery_app.conf, "result_backend", "cache+memory://")
    # app.backend caches the instance built from the conf above, in one of two
    # places depending on whether that backend is thread safe.
    monkeypatch.setattr(celery_app, "_backend_cache", None, raising=False)
    monkeypatch.delattr(celery_app._local, "backend", raising=False)
    return celery_app


@pytest.fixture
def fast_retries(monkeypatch):
    """Collapse retry backoff so a deliberately failing stage doesn't sleep."""
    for task in (
        tasks.segment,
        tasks.vision,
        tasks.audio,
        tasks.narrate,
        tasks.tts,
        tasks.mux,
    ):
        monkeypatch.setattr(task, "max_retries", 0)


@pytest.fixture
def stub_stages(monkeypatch, fake_gemini_client, fake_kokoro, job_blobs):
    """Point every task at one local scratch dir and stub the heavy models."""
    monkeypatch.setattr(tasks, "_blobs", lambda job_id: job_blobs)
    monkeypatch.setattr(tasks, "_gemini_client", lambda: fake_gemini_client)
    monkeypatch.setattr(tasks, "detect_speech_regions", lambda audio_path: [])
    monkeypatch.setattr(tasks, "transcribe", lambda audio_path, regions: [])
    return job_blobs


async def _uploaded_job(db, owner, blobs, synthetic_video):
    """A job whose source video is already in the blob store."""
    job = await repo.create_job(owner, "clip.mp4", "source.mp4")
    blobs.path("source.mp4").write_bytes(Path(synthetic_video).read_bytes())
    return job


# --------------------------------------------------------------------------- #
# the full canvas
# --------------------------------------------------------------------------- #


async def test_canvas_runs_every_stage_and_finishes_the_job(
    db, owner, eager, stub_stages, synthetic_video
):
    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)

    tasks.segment.apply(args=[job.id]).get()

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_DONE
    assert fetched.duration_sec == pytest.approx(4.0, abs=0.2)

    timeline = await repo.get_timeline(job.id)
    assert timeline is not None
    assert timeline.job_id == job.id
    assert len(timeline.segments) == 2
    assert timeline.described_key == "described.mp4"
    assert stub_stages.path("described.mp4").exists()

    for segment in timeline.segments:
        assert segment.frames
        for frame in segment.frames:
            # The vision results survived the fan-out and the reload.
            assert frame.visual is not None
            assert frame.visual.description == "a test scene"
        assert segment.ad_eligible == (
            (segment.narratable_gap_sec or 0) >= MIN_NARRATABLE_GAP_SEC
        )
        if segment.ad_eligible:
            assert segment.ad_narration
            assert segment.ad_narration_key is not None


async def test_canvas_streams_the_same_events_as_the_single_process_pipeline(
    db, owner, eager, stub_stages, synthetic_video
):
    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)

    tasks.segment.apply(args=[job.id]).get()

    events = await repo.events_since(job.id)
    stages = {e["stage"] for e in events if e.get("type") == "stage"}
    assert {"segmentation", "audio", "timeline", "narration", "tts", "mux"} <= stages

    # Sequences are dense and ordered, which is what makes resume-from-seq work.
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))

    frame_events = [e for e in events if e.get("type") == "frame"]
    timeline = await repo.get_timeline(job.id)
    assert timeline is not None
    assert len(frame_events) == sum(len(s.frames) for s in timeline.segments)

    assert any(e.get("type") == "described_video" for e in events)
    assert events[-1]["type"] == "status"
    assert events[-1]["status"] == repo.STATUS_DONE


async def test_canvas_handles_a_video_with_no_audio_track(
    db, owner, eager, stub_stages, monkeypatch, synthetic_video
):
    from audio_extract import NoAudioStreamError

    def _no_audio(video_path, out_path):
        raise NoAudioStreamError("no audio")

    monkeypatch.setattr(tasks, "extract_audio", _no_audio)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("audio stages must be skipped when there is no audio")

    monkeypatch.setattr(tasks, "detect_speech_regions", _should_not_run)
    monkeypatch.setattr(tasks, "transcribe", _should_not_run)

    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)
    tasks.segment.apply(args=[job.id]).get()

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_DONE

    has_audio, regions, transcript = job_state.load_audio(stub_stages)
    assert has_audio is False
    assert regions == [] and transcript == []


async def test_canvas_skips_the_mux_when_nothing_was_narrated(
    db, owner, eager, stub_stages, monkeypatch, synthetic_video
):
    import ad_track

    monkeypatch.setattr(ad_track, "build_ad_track", lambda timeline, blobs: None)

    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)
    tasks.segment.apply(args=[job.id]).get()

    timeline = await repo.get_timeline(job.id)
    assert timeline is not None
    assert timeline.described_key is None
    assert not stub_stages.path("described.mp4").exists()

    # The job still finishes — there was simply nothing to mix.
    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_DONE


# --------------------------------------------------------------------------- #
# individual stages
# --------------------------------------------------------------------------- #


async def test_segment_records_duration_and_the_shot_skeleton(
    db, owner, eager, stub_stages, monkeypatch, synthetic_video
):
    """Segmentation is the only stage that reads the source video's metadata."""
    replaced = {}
    monkeypatch.setattr(
        tasks.segment, "replace", lambda sig: replaced.setdefault("sig", sig)
    )

    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)
    tasks.segment.apply(args=[job.id]).get()

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_PROCESSING
    assert fetched.duration_sec == pytest.approx(4.0, abs=0.2)

    shots, meta = job_state.load_shots(stub_stages)
    assert len(shots) == 2
    assert meta["duration_sec"] == pytest.approx(4.0, abs=0.2)
    for shot in shots:
        for frame in shot["frames"]:
            assert stub_stages.path(frame["key"]).exists()


async def test_segment_fans_out_one_vision_task_per_shot_plus_audio(
    db, owner, eager, stub_stages, monkeypatch, synthetic_video
):
    """The fan-out width isn't known until the shots are found, hence replace()."""
    captured = {}
    monkeypatch.setattr(
        tasks.segment, "replace", lambda sig: captured.setdefault("sig", sig)
    )

    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)
    tasks.segment.apply(args=[job.id]).get()

    header_tasks = [t["task"] for t in captured["sig"].tasks]
    assert header_tasks.count("tasks.vision") == 2  # one per shot
    assert header_tasks.count("tasks.audio") == 1
    # Vision tasks are addressed by shot id, so each owns a distinct blob.
    vision_shot_ids = [
        t.args[1] for t in captured["sig"].tasks if t["task"] == "tasks.vision"
    ]
    assert sorted(vision_shot_ids) == [0, 1]

    body_tasks = [t["task"] for t in captured["sig"].body.tasks]
    assert body_tasks == [
        "tasks.build_timeline",
        "tasks.narrate",
        "tasks.tts",
        "tasks.mux",
    ]


async def test_vision_writes_only_its_own_shots_results(
    db, owner, eager, stub_stages, monkeypatch, synthetic_video
):
    monkeypatch.setattr(tasks.segment, "replace", lambda sig: None)
    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)
    tasks.segment.apply(args=[job.id]).get()

    tasks.vision.apply(args=[job.id, 0]).get()

    assert stub_stages.path("state/vision/shot_0000.json").exists()
    assert not stub_stages.path("state/vision/shot_0001.json").exists()


async def test_vision_tolerates_a_shot_that_is_no_longer_there(
    db, owner, eager, stub_stages, monkeypatch, synthetic_video
):
    """A redelivered task must not crash on state that has moved on."""
    monkeypatch.setattr(tasks.segment, "replace", lambda sig: None)
    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)
    tasks.segment.apply(args=[job.id]).get()

    tasks.vision.apply(args=[job.id, 999]).get()  # must not raise

    assert not stub_stages.path("state/vision/shot_0999.json").exists()


# --------------------------------------------------------------------------- #
# failure handling
# --------------------------------------------------------------------------- #


async def test_a_failing_stage_reports_itself_on_the_job(
    db, owner, eager, fast_retries, stub_stages, monkeypatch, synthetic_video
):
    """JobTask.on_failure is what surfaces a stage failure to the client."""

    def _boom(*args, **kwargs):
        raise RuntimeError("kokoro exploded")

    monkeypatch.setattr(tasks, "synthesize_narration", _boom)
    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)

    with pytest.raises(RuntimeError):
        tasks.segment.apply(args=[job.id]).get()

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_ERROR
    assert fetched.error == "RuntimeError: kokoro exploded"


async def test_a_failing_stage_marks_the_job_errored(
    db, owner, eager, fast_retries, stub_stages, monkeypatch, synthetic_video
):
    monkeypatch.setattr(
        tasks, "probe_video", lambda path: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)

    with pytest.raises(RuntimeError):
        tasks.segment.apply(args=[job.id]).get()

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_ERROR
    assert "boom" in (fetched.error or "")


async def test_a_job_with_no_source_video_fails_loudly(
    db, owner, eager, fast_retries, stub_stages
):
    """A missing source blob is retryable (R2 can lag), then gives up."""
    job = await repo.create_job(owner, "clip.mp4", "source.mp4")

    with pytest.raises(FileNotFoundError):
        tasks.segment.apply(args=[job.id]).get()


async def test_a_failing_fan_out_task_still_reaches_the_error_callback(
    db, owner, eager, fast_retries, stub_stages, monkeypatch, synthetic_video
):
    """Vision and audio run in the chord *header*.

    A chord's error callback does not reliably reach header tasks, so failure
    reporting lives on JobTask.on_failure instead; without it these failures
    would leave the job in "processing" until the reaper timed it out.
    """

    def _boom(*args, **kwargs):
        raise RuntimeError("vision exploded")

    monkeypatch.setattr(tasks, "analyze_shots", _boom)

    job = await _uploaded_job(db, owner, stub_stages, synthetic_video)
    with pytest.raises(RuntimeError):
        tasks.segment.apply(args=[job.id]).get()

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_ERROR
    assert "vision exploded" in (fetched.error or "")


async def test_reap_task_marks_abandoned_jobs(db, owner, eager):
    job = await repo.create_job(owner, "clip.mp4", "source.mp4")
    await repo.set_status(job.id, repo.STATUS_PROCESSING)
    await db.execute(
        "update jobs set heartbeat_at = now() - interval '1 hour' where id = $1",
        uuid.UUID(job.id),
    )

    assert tasks.reap_stale_jobs.apply().get() == [job.id]

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_INTERRUPTED
