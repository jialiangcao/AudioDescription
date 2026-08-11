"""The HTTP/WS API: auth, presigned upload, presigned media, event streaming."""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

import repo
import server
from timeline import (
    AudioAnalysis,
    Frame,
    FrameAnalysis,
    Segment,
    Timeline,
)


def _timeline(job_id: str) -> Timeline:
    return Timeline(
        job_id=job_id,
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
                        key="frames/shot_0000_00.jpg",
                        visual=FrameAnalysis(
                            description="a test scene",
                            entities=[],
                            actions=["an object moves"],
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
                ad_narration="canned narration",
                ad_narration_key="narration/shot_0000.wav",
            )
        ],
        ad_track_key="narration/ad_track.wav",
        described_key="described.mp4",
    )


class FakeBlobs:
    """A JobBlobs whose presigning and size checks are deterministic."""

    uploaded_size: int | None = 1024

    def __init__(self, job_id):
        self.job_id = job_id

    def presign_put(self, key, ttl_sec=None, content_type=None):
        return f"https://bucket.example/jobs/{self.job_id}/{key}?put"

    def presign_get(self, key, ttl_sec=None):
        return f"https://bucket.example/jobs/{self.job_id}/{key}?get"

    def size(self, key):
        return type(self).uploaded_size


@pytest.fixture
def enqueued(monkeypatch):
    """Capture what the API hands to Celery instead of running it."""
    jobs, questions = [], []
    monkeypatch.setattr(server, "enqueue_job", jobs.append)
    monkeypatch.setattr(
        server.answer_question_task,
        "apply_async",
        lambda args=None, **kw: questions.append(tuple(args or ())),
    )
    return jobs, questions


@pytest.fixture
def client(db, owner, monkeypatch, enqueued):
    """A TestClient authenticated as `owner`, with storage and Celery faked."""
    monkeypatch.setenv("ADESC_DEV_USER_ID", owner)
    monkeypatch.setattr(
        server.JobBlobs, "remote", classmethod(lambda cls, job_id: FakeBlobs(job_id))
    )
    FakeBlobs.uploaded_size = 1024
    with TestClient(server.app) as test_client:
        yield test_client


async def _job(owner, status=repo.STATUS_CREATED):
    job = await repo.create_job(owner, "clip.mp4", "source.mp4")
    if status != repo.STATUS_CREATED:
        await repo.set_status(job.id, status)
    return job


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #


def test_healthz_needs_no_dependencies(client):
    """Liveness must not fail just because a dependency is briefly down."""
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_reports_a_broken_dependency(client, monkeypatch):
    async def _broken():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(server.events, "get_redis", lambda: _Unreachable())

    response = client.get("/readyz")
    assert response.status_code == 503
    assert "redis" in response.json()["detail"]


class _Unreachable:
    async def ping(self):
        raise ConnectionError("redis is down")


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #


def test_every_api_route_requires_a_token(db, monkeypatch):
    monkeypatch.delenv("ADESC_DEV_USER_ID", raising=False)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "not-the-signing-key")

    with TestClient(server.app) as anon:
        assert anon.get("/api/jobs").status_code == 401
        assert anon.post("/api/jobs", json={"filename": "a.mp4"}).status_code == 401


async def test_a_job_belonging_to_someone_else_is_indistinguishable_from_a_missing_one(
    client, db
):
    """Job ids must not be probeable for existence.

    Answering 403 for a real job and 404 for an unknown one would let anyone
    enumerate which ids exist.
    """
    other = str(uuid.uuid4())
    await db.execute("insert into auth.users (id) values ($1)", uuid.UUID(other))
    theirs = await repo.create_job(other, "theirs.mp4", "source.mp4")

    theirs_response = client.get(f"/api/jobs/{theirs.id}")
    missing_response = client.get(f"/api/jobs/{uuid.uuid4()}")

    assert theirs_response.status_code == missing_response.status_code == 404
    assert theirs_response.json() == missing_response.json()


async def test_another_users_job_cannot_be_started_or_asked(client, db):
    other = str(uuid.uuid4())
    await db.execute("insert into auth.users (id) values ($1)", uuid.UUID(other))
    theirs = await repo.create_job(other, "theirs.mp4", "source.mp4")

    assert client.post(f"/api/jobs/{theirs.id}/start").status_code == 404
    assert (
        client.post(f"/api/jobs/{theirs.id}/ask", json={"question": "q?"}).status_code
        == 404
    )
    assert client.post(f"/api/jobs/{theirs.id}/ws-ticket").status_code == 404


def test_a_malformed_job_id_is_a_404_not_a_500(client):
    assert client.get("/api/jobs/not-a-uuid").status_code == 404


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #


def test_create_job_returns_a_presigned_put(client):
    response = client.post("/api/jobs", json={"filename": "clip.mp4"})

    assert response.status_code == 201
    body = response.json()
    assert body["key"] == "source.mp4"
    assert body["upload_url"].endswith("source.mp4?put")
    assert uuid.UUID(body["job_id"])


def test_create_job_rejects_an_unsupported_type(client):
    response = client.post("/api/jobs", json={"filename": "notes.pdf"})
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"]


def test_create_job_enforces_the_per_user_quota(client):
    for _ in range(repo.MAX_ACTIVE_JOBS_PER_USER):
        job_id = client.post("/api/jobs", json={"filename": "clip.mp4"}).json()[
            "job_id"
        ]
        client.post(f"/api/jobs/{job_id}/start")

    response = client.post("/api/jobs", json={"filename": "clip.mp4"})
    assert response.status_code == 429


def test_start_enqueues_the_job(client, enqueued):
    job_id = client.post("/api/jobs", json={"filename": "clip.mp4"}).json()["job_id"]

    response = client.post(f"/api/jobs/{job_id}/start")

    assert response.status_code == 202
    assert response.json()["status"] == repo.STATUS_QUEUED
    queued_jobs, _ = enqueued
    assert queued_jobs == [job_id]


def test_start_refuses_when_nothing_was_uploaded(client, enqueued):
    job_id = client.post("/api/jobs", json={"filename": "clip.mp4"}).json()["job_id"]
    FakeBlobs.uploaded_size = None

    response = client.post(f"/api/jobs/{job_id}/start")

    assert response.status_code == 400
    assert enqueued[0] == []


def test_start_refuses_an_oversized_upload(client, enqueued):
    """Size is checked against the object, since the API never sees the bytes."""
    job_id = client.post("/api/jobs", json={"filename": "clip.mp4"}).json()["job_id"]
    FakeBlobs.uploaded_size = server.MAX_UPLOAD_BYTES + 1

    response = client.post(f"/api/jobs/{job_id}/start")

    assert response.status_code == 413
    assert enqueued[0] == []


def test_start_is_rejected_twice(client, enqueued):
    job_id = client.post("/api/jobs", json={"filename": "clip.mp4"}).json()["job_id"]
    client.post(f"/api/jobs/{job_id}/start")

    response = client.post(f"/api/jobs/{job_id}/start")

    assert response.status_code == 409
    assert len(enqueued[0]) == 1


# --------------------------------------------------------------------------- #
# reading a job
# --------------------------------------------------------------------------- #


async def test_get_job_presigns_every_media_key(client, owner):
    job = await _job(owner, repo.STATUS_DONE)
    await repo.save_timeline(job.id, _timeline(job.id))

    body = client.get(f"/api/jobs/{job.id}").json()

    timeline = body["timeline"]
    assert timeline["described_url"].endswith("described.mp4?get")
    assert timeline["ad_track_url"].endswith("narration/ad_track.wav?get")
    segment = timeline["segments"][0]
    assert segment["ad_narration_url"].endswith("narration/shot_0000.wav?get")
    assert segment["frames"][0]["url"].endswith("frames/shot_0000_00.jpg?get")


async def test_get_job_never_exposes_a_server_path(client, owner):
    """Keys are opaque; the old API returned absolute /tmp paths verbatim."""
    job = await _job(owner, repo.STATUS_DONE)
    await repo.save_timeline(job.id, _timeline(job.id))

    body = client.get(f"/api/jobs/{job.id}").text

    assert "/tmp" not in body
    assert "adesc-jobs" not in body


async def test_list_jobs_returns_only_this_users_jobs(client, db, owner):
    await _job(owner)
    other = str(uuid.uuid4())
    await db.execute("insert into auth.users (id) values ($1)", uuid.UUID(other))
    await repo.create_job(other, "theirs.mp4", "source.mp4")

    jobs = client.get("/api/jobs").json()["jobs"]

    assert len(jobs) == 1
    assert jobs[0]["filename"] == "clip.mp4"


# --------------------------------------------------------------------------- #
# Q&A
# --------------------------------------------------------------------------- #


async def test_ask_queues_a_run_and_returns_its_id(client, owner, enqueued):
    job = await _job(owner, repo.STATUS_DONE)
    await repo.save_timeline(job.id, _timeline(job.id))

    response = client.post(f"/api/jobs/{job.id}/ask", json={"question": "what colour?"})

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert response.json()["status"] == "queued"
    _, questions = enqueued
    assert questions == [(run_id, job.id)]

    run = await repo.get_qa_run(run_id, owner)
    assert run is not None
    assert run["question"] == "what colour?"


async def test_ask_before_the_job_finishes_is_rejected(client, owner, enqueued):
    job = await _job(owner, repo.STATUS_PROCESSING)

    response = client.post(f"/api/jobs/{job.id}/ask", json={"question": "q?"})

    assert response.status_code == 409
    assert enqueued[1] == []


async def test_ask_after_the_timeline_expired_is_gone(client, owner):
    job = await _job(owner, repo.STATUS_DONE)  # done, but no timeline saved

    response = client.post(f"/api/jobs/{job.id}/ask", json={"question": "q?"})

    assert response.status_code == 410


async def test_qa_run_is_readable_by_its_owner_only(client, db, owner):
    job = await _job(owner, repo.STATUS_DONE)
    run_id = await repo.create_qa_run(job.id, owner, "q?")
    await repo.finish_qa_run(
        run_id, "completed", answer="blue", cycles=2, history=[{"action": "finish"}]
    )

    body = client.get(f"/api/qa/{run_id}").json()

    assert body["answer"] == "blue"
    assert body["history"] == [{"action": "finish"}]


def test_unknown_qa_run_is_a_404(client):
    assert client.get(f"/api/qa/{uuid.uuid4()}").status_code == 404


# --------------------------------------------------------------------------- #
# websocket
# --------------------------------------------------------------------------- #


class FakePubSub:
    """A pub/sub that yields a scripted burst of messages, then idles."""

    def __init__(self, messages):
        self._messages = list(messages)

    async def subscribe(self, channel):
        return None

    async def get_message(self, ignore_subscribe_messages=True, timeout=None):
        if self._messages:
            return {"data": json.dumps(self._messages.pop(0))}
        return None

    async def unsubscribe(self, channel):
        return None

    async def aclose(self):
        return None


@pytest.fixture
def fake_pubsub(monkeypatch):
    """Replace Redis with a scriptable pub/sub for the websocket tests."""
    scripted = []

    class _Redis:
        def pubsub(self):
            return FakePubSub(scripted)

    monkeypatch.setattr(server.events, "get_redis", lambda: _Redis())
    return scripted


@pytest.fixture
def fake_tickets(monkeypatch, owner):
    """Ticket redemption without Redis; records what was issued."""
    issued = {}

    async def _issue(user_id, job_id):
        issued["ticket"] = f"t-{job_id}"
        issued["job_id"] = job_id
        return issued["ticket"]

    async def _redeem(ticket):
        if ticket and ticket == issued.get("ticket"):
            return owner, issued["job_id"]
        return None

    monkeypatch.setattr(server.tickets, "issue", _issue)
    monkeypatch.setattr(server.tickets, "redeem", _redeem)
    return issued


async def test_websocket_replays_the_log_then_streams_live(
    client, owner, fake_pubsub, fake_tickets
):
    # Left in "created" so the only events are the ones appended here — a
    # set_status call would publish a status event of its own and shift seqs.
    job = await _job(owner)
    await repo.append_event(job.id, {"type": "stage", "stage": "segmentation"})
    await repo.append_event(job.id, {"type": "stage", "stage": "vision"})
    fake_pubsub.append(
        {"type": "status", "status": repo.STATUS_DONE, "error": None, "seq": 99}
    )

    ticket = client.post(f"/api/jobs/{job.id}/ws-ticket").json()["ticket"]
    received = []
    with client.websocket_connect(
        f"/api/jobs/{job.id}/events?ticket={ticket}"
    ) as socket:
        for _ in range(3):
            received.append(socket.receive_json())

    assert [e["stage"] for e in received[:2]] == ["segmentation", "vision"]
    assert received[-1]["status"] == repo.STATUS_DONE


async def test_websocket_resumes_from_the_clients_last_sequence(
    client, owner, fake_pubsub, fake_tickets
):
    """A reconnect must not re-send the whole log, only what was missed."""
    job = await _job(owner)
    for i in range(4):  # seq 1..4
        await repo.append_event(job.id, {"type": "frame", "index": i})
    fake_pubsub.append(
        {"type": "status", "status": repo.STATUS_DONE, "error": None, "seq": 99}
    )

    ticket = client.post(f"/api/jobs/{job.id}/ws-ticket").json()["ticket"]
    received = []
    with client.websocket_connect(
        f"/api/jobs/{job.id}/events?ticket={ticket}&since=2"
    ) as socket:
        for _ in range(3):
            received.append(socket.receive_json())
    # Frames 0 and 1 (seq 1 and 2) were already delivered before the drop.

    assert [e.get("index") for e in received[:2]] == [2, 3]
    assert received[-1]["status"] == repo.STATUS_DONE


async def test_websocket_drops_live_events_already_covered_by_the_replay(
    client, owner, fake_pubsub, fake_tickets
):
    """Subscribe-then-replay overlaps on purpose; the overlap must not duplicate."""
    job = await _job(owner)
    await repo.append_event(job.id, {"type": "frame", "index": 0})  # seq 1
    fake_pubsub.extend(
        [
            {"type": "frame", "index": 0, "seq": 1},  # already replayed
            {"type": "status", "status": repo.STATUS_DONE, "error": None, "seq": 2},
        ]
    )

    ticket = client.post(f"/api/jobs/{job.id}/ws-ticket").json()["ticket"]
    received = []
    with client.websocket_connect(
        f"/api/jobs/{job.id}/events?ticket={ticket}"
    ) as socket:
        for _ in range(2):
            received.append(socket.receive_json())

    assert [e["type"] for e in received] == ["frame", "status"]


async def test_websocket_without_a_valid_ticket_is_refused(
    client, owner, fake_pubsub, fake_tickets
):
    job = await _job(owner)

    # Starlette closes before accept, which surfaces as an exception here.
    with (
        pytest.raises(Exception),  # noqa: B017
        client.websocket_connect(f"/api/jobs/{job.id}/events?ticket=forged") as socket,
    ):
        socket.receive_json()


async def test_a_ticket_is_bound_to_its_job(client, owner, fake_pubsub, fake_tickets):
    """A ticket for one job must not open another job's stream."""
    job = await _job(owner)
    other = await _job(owner)
    ticket = client.post(f"/api/jobs/{job.id}/ws-ticket").json()["ticket"]

    # Starlette closes before accept, which surfaces as an exception here.
    with (
        pytest.raises(Exception),  # noqa: B017
        client.websocket_connect(
            f"/api/jobs/{other.id}/events?ticket={ticket}"
        ) as socket,
    ):
        socket.receive_json()


# --------------------------------------------------------------------------- #
# media URLs
# --------------------------------------------------------------------------- #


async def test_media_urls_presigns_a_batch_of_keys(client, owner):
    """Progress events carry keys, so the client presigns them as they arrive."""
    job = await _job(owner)

    body = client.post(
        f"/api/jobs/{job.id}/media-urls",
        json={"keys": ["frames/shot_0000_00.jpg", "narration/shot_0000.wav"]},
    ).json()

    assert body["urls"]["frames/shot_0000_00.jpg"].endswith(
        "frames/shot_0000_00.jpg?get"
    )
    assert body["urls"]["narration/shot_0000.wav"].endswith(
        "narration/shot_0000.wav?get"
    )


async def test_media_urls_refuses_keys_that_escape_the_job(client, owner):
    """Keys come from the client, so they must not address another job."""
    job = await _job(owner)

    body = client.post(
        f"/api/jobs/{job.id}/media-urls",
        json={"keys": ["../other-job/described.mp4", "/etc/passwd", "frames/ok.jpg"]},
    ).json()

    assert list(body["urls"]) == ["frames/ok.jpg"]


async def test_media_urls_caps_the_batch_size(client, owner):
    job = await _job(owner)

    response = client.post(
        f"/api/jobs/{job.id}/media-urls",
        json={"keys": [f"frames/{i}.jpg" for i in range(server.MAX_PRESIGN_BATCH + 1)]},
    )

    assert response.status_code == 400


async def test_media_urls_are_scoped_to_the_owner(client, db):
    other = str(uuid.uuid4())
    await db.execute("insert into auth.users (id) values ($1)", uuid.UUID(other))
    theirs = await repo.create_job(other, "theirs.mp4", "source.mp4")

    response = client.post(
        f"/api/jobs/{theirs.id}/media-urls", json={"keys": ["frames/a.jpg"]}
    )

    assert response.status_code == 404
