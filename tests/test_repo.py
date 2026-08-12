"""repo.py against a real Postgres (skipped when one isn't configured)."""

import asyncio
import uuid

import pytest

import repo
from timeline import Segment, Timeline


def _timeline(job_id: str) -> Timeline:
    return Timeline(
        job_id=job_id,
        duration_sec=4.0,
        segments=[Segment(id=0, start=0.0, end=4.0)],
    )


async def _job(owner: str) -> repo.Job:
    return await repo.create_job(owner, "clip.mp4", "source.mp4")


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


async def test_create_and_get_job(db, owner):
    job = await _job(owner)

    assert job.status == repo.STATUS_CREATED
    assert job.source_key == "source.mp4"

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.id == job.id


async def test_get_job_scoped_to_owner_hides_other_users_jobs(db, owner):
    """Ownership is enforced in the query, not only by RLS."""
    other = str(uuid.uuid4())
    await db.execute("insert into auth.users (id) values ($1)", uuid.UUID(other))
    job = await _job(owner)

    assert await repo.get_job(job.id, owner_id=other) is None
    assert await repo.get_job(job.id, owner_id=owner) is not None


async def test_active_count_and_capacity_are_per_user(db, owner):
    other = str(uuid.uuid4())
    await db.execute("insert into auth.users (id) values ($1)", uuid.UUID(other))

    for _ in range(repo.MAX_ACTIVE_JOBS_PER_USER):
        job = await _job(owner)
        await repo.set_status(job.id, repo.STATUS_QUEUED)

    assert await repo.at_capacity(owner) is True
    # Another user is unaffected by this one's queue.
    assert await repo.at_capacity(other) is False


async def test_terminal_jobs_do_not_count_against_capacity(db, owner):
    job = await _job(owner)
    await repo.set_status(job.id, repo.STATUS_QUEUED)
    assert await repo.active_count(owner) == 1

    await repo.set_status(job.id, repo.STATUS_DONE)
    assert await repo.active_count(owner) == 0


async def test_set_status_records_the_error_and_publishes_an_event(db, owner):
    job = await _job(owner)
    await repo.set_status(job.id, repo.STATUS_ERROR, error="ffmpeg exploded")

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_ERROR
    assert fetched.error == "ffmpeg exploded"

    (event,) = await repo.events_since(job.id)
    assert event["type"] == "status"
    assert event["status"] == repo.STATUS_ERROR
    assert event["error"] == "ffmpeg exploded"


# --------------------------------------------------------------------------- #
# heartbeat / reaping
# --------------------------------------------------------------------------- #


async def test_reap_marks_jobs_whose_worker_stopped_checking_in(db, owner):
    job = await _job(owner)
    await repo.set_status(job.id, repo.STATUS_PROCESSING)
    await db.execute(
        "update jobs set heartbeat_at = now() - interval '1 hour' where id = $1",
        uuid.UUID(job.id),
    )

    assert await repo.reap_stale_jobs(timeout_sec=600) == [job.id]

    fetched = await repo.get_job(job.id, owner_id=owner)
    assert fetched is not None
    assert fetched.status == repo.STATUS_INTERRUPTED
    assert fetched.error == "worker stopped responding"


async def test_reap_leaves_a_live_worker_alone(db, owner):
    job = await _job(owner)
    await repo.set_status(job.id, repo.STATUS_PROCESSING)
    await repo.heartbeat(job.id)

    assert await repo.reap_stale_jobs(timeout_sec=600) == []


async def test_reap_ignores_jobs_that_are_not_processing(db, owner):
    """A queued job has no worker yet, so a stale heartbeat means nothing."""
    job = await _job(owner)
    await repo.set_status(job.id, repo.STATUS_QUEUED)
    await db.execute(
        "update jobs set heartbeat_at = now() - interval '1 hour' where id = $1",
        uuid.UUID(job.id),
    )

    assert await repo.reap_stale_jobs(timeout_sec=600) == []


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #


async def test_event_seq_is_monotonic_per_job(db, owner):
    job_a = await _job(owner)
    job_b = await _job(owner)

    assert await repo.append_event(job_a.id, {"type": "stage", "stage": "one"}) == 1
    assert await repo.append_event(job_a.id, {"type": "stage", "stage": "two"}) == 2
    # Sequences are per job, so a second job starts at 1 again.
    assert await repo.append_event(job_b.id, {"type": "stage", "stage": "one"}) == 1


async def test_concurrent_appends_get_distinct_sequences(db, owner):
    """The vision stage publishes one event per frame from several workers."""
    job = await _job(owner)

    seqs = await asyncio.gather(
        *(repo.append_event(job.id, {"type": "frame", "index": i}) for i in range(25))
    )

    assert sorted(seqs) == list(range(1, 26))


async def test_events_since_replays_only_what_the_client_missed(db, owner):
    job = await _job(owner)
    for i in range(5):
        await repo.append_event(job.id, {"type": "frame", "index": i})

    tail = await repo.events_since(job.id, since_seq=3)

    assert [e["seq"] for e in tail] == [4, 5]
    assert [e["index"] for e in tail] == [3, 4]


async def test_events_since_zero_replays_everything(db, owner):
    job = await _job(owner)
    await repo.append_event(job.id, {"type": "stage", "stage": "segmentation"})

    assert len(await repo.events_since(job.id, since_seq=0)) == 1


async def test_deleting_a_job_takes_its_events_with_it(db, owner):
    job = await _job(owner)
    await repo.append_event(job.id, {"type": "stage", "stage": "one"})

    await db.execute("delete from jobs where id = $1", uuid.UUID(job.id))

    assert await repo.events_since(job.id) == []


# --------------------------------------------------------------------------- #
# timelines
# --------------------------------------------------------------------------- #


async def test_timeline_roundtrips_through_jsonb(db, owner):
    job = await _job(owner)
    await repo.save_timeline(job.id, _timeline(job.id))

    loaded = await repo.get_timeline(job.id)

    assert loaded is not None
    assert loaded.job_id == job.id
    assert loaded.duration_sec == 4.0
    assert len(loaded.segments) == 1


async def test_saving_a_timeline_twice_replaces_it(db, owner):
    job = await _job(owner)
    await repo.save_timeline(job.id, _timeline(job.id))

    updated = _timeline(job.id)
    updated.described_key = "described.mp4"
    await repo.save_timeline(job.id, updated)

    loaded = await repo.get_timeline(job.id)
    assert loaded is not None
    assert loaded.described_key == "described.mp4"


async def test_get_timeline_is_none_before_the_pipeline_writes_one(db, owner):
    job = await _job(owner)
    assert await repo.get_timeline(job.id) is None


# --------------------------------------------------------------------------- #
# qa_runs
# --------------------------------------------------------------------------- #


async def test_qa_run_lifecycle(db, owner):
    job = await _job(owner)
    run_id = await repo.create_qa_run(job.id, owner, "what colour is the car?")

    run = await repo.get_qa_run(run_id, owner)
    assert run is not None
    assert run["status"] == "queued"
    assert run["answer"] is None

    await repo.finish_qa_run(
        run_id,
        "completed",
        answer="blue",
        reason="Task finished successfully.",
        cycles=3,
        history=[{"action": "finish", "answer": "blue"}],
    )

    run = await repo.get_qa_run(run_id, owner)
    assert run is not None
    assert run["status"] == "completed"
    assert run["answer"] == "blue"
    assert run["cycles"] == 3
    assert run["history"] == [{"action": "finish", "answer": "blue"}]


async def test_qa_run_is_scoped_to_its_owner(db, owner):
    other = str(uuid.uuid4())
    await db.execute("insert into auth.users (id) values ($1)", uuid.UUID(other))
    job = await _job(owner)
    run_id = await repo.create_qa_run(job.id, owner, "q?")

    assert await repo.get_qa_run(run_id, other) is None


async def test_failed_qa_run_records_its_error(db, owner):
    job = await _job(owner)
    run_id = await repo.create_qa_run(job.id, owner, "q?")

    await repo.finish_qa_run(run_id, "failed", error="Gemini timed out")

    run = await repo.get_qa_run(run_id, owner)
    assert run is not None
    assert run["status"] == "failed"
    assert run["error"] == "Gemini timed out"


@pytest.mark.parametrize("bad_id", ["not-a-uuid", ""])
async def test_get_job_rejects_a_malformed_id(db, owner, bad_id):
    """Route params reach these helpers directly, so bad input must not 500."""
    with pytest.raises(ValueError):
        await repo.get_job(bad_id, owner_id=owner)
