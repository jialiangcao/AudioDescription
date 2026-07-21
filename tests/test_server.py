import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import qa
import server
from jobs import JobStore
from timeline import AudioAnalysis, Segment, Timeline, VisualAnalysis


def _timeline() -> Timeline:
    return Timeline(
        video_id="v.mp4",
        duration_sec=2.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=2.0,
                keyframe="shot_0000_00.jpg",
                visual=VisualAnalysis(
                    description="a test scene",
                    entities=[],
                    setting="s",
                    on_screen_text=None,
                ),
                audio=AudioAnalysis(
                    has_speech=False, transcript=None, silence_ratio=1.0
                ),
                ad_eligible=True,
                narratable_gap_sec=2.0,
            )
        ],
    )


async def _fake_run_pipeline(video_path, job_dir, on_event, client=None):
    frames = job_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "shot_0000_00.jpg").write_bytes(b"fake-jpeg")

    narration = job_dir / "narration"
    narration.mkdir(parents=True, exist_ok=True)
    (narration / "shot_0000.wav").write_bytes(b"fake-wav")

    await on_event({"type": "stage", "stage": "segmentation", "status": "start"})
    await on_event(
        {
            "type": "shot",
            "shot": {"id": 0, "keyframe": str(frames / "shot_0000_00.jpg")},
        }
    )
    tl = _timeline()
    await on_event({"type": "timeline", "timeline": tl.model_dump()})
    await on_event({"type": "narration", "segment_id": 0, "text": "canned narration"})
    tl.segments[0].ad_narration = "canned narration"
    tl.segments[0].ad_narration_audio = str(narration / "shot_0000.wav")
    tl.segments[0].ad_narration_duration_sec = 1.5
    tl.segments[0].ad_narration_overflow = False
    await on_event(
        {
            "type": "narration_audio",
            "segment_id": 0,
            "audio": "shot_0000.wav",
            "duration_sec": 1.5,
            "overflow": False,
        }
    )
    return tl


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JobStore", lambda: JobStore(root=tmp_path / "jobs"))
    monkeypatch.setattr(server, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(server, "_get_pipeline_client", lambda app: None)
    monkeypatch.setattr(qa, "answer_question", lambda timeline, question: "blue")
    with TestClient(server.app) as c:
        yield c


def _upload(client, name="clip.mp4"):
    return client.post(
        "/api/jobs", files={"file": (name, b"fake-video-bytes", "video/mp4")}
    )


def _wait_status(client, job_id, target, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] == target:
            return data
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {target!r} in {timeout}s")


def test_upload_then_status_then_ask(client):
    resp = _upload(client)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    data = _wait_status(client, job_id, "done")
    assert data["timeline"]["segments"][0]["visual"]["description"] == "a test scene"

    answer = client.post(f"/api/jobs/{job_id}/ask", json={"question": "eye color?"})
    assert answer.status_code == 200
    assert answer.json()["answer"] == "blue"


def test_upload_rejects_unsupported_type(client):
    resp = client.post("/api/jobs", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert resp.status_code == 400


def test_status_404_for_unknown_job(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_ask_before_done_conflicts(client):
    # a job that exists but is not done yet
    store: JobStore = server.app.state.store
    job = store.create()
    resp = client.post(f"/api/jobs/{job.id}/ask", json={"question": "x"})
    assert resp.status_code == 409


def test_websocket_replays_and_streams_to_terminal(client):
    job_id = _upload(client).json()["job_id"]
    _wait_status(client, job_id, "done")

    # connecting after completion still replays the whole log incl. terminal
    with client.websocket_connect(f"/api/jobs/{job_id}/events") as ws:
        types_seen = []
        while True:
            event = ws.receive_json()
            types_seen.append(event.get("type"))
            if event.get("type") == "status" and event.get("status") in {
                "done",
                "error",
                "interrupted",
            }:
                break

    assert "shot" in types_seen
    assert "timeline" in types_seen
    assert types_seen[-1] == "status"


def test_websocket_unknown_job_closed(client):
    with (
        client.websocket_connect("/api/jobs/nope/events") as ws,
        pytest.raises(WebSocketDisconnect),
    ):
        ws.receive_json()


def test_frames_serves_jpg(client):
    job_id = _upload(client).json()["job_id"]
    _wait_status(client, job_id, "done")

    resp = client.get(f"/api/jobs/{job_id}/frames/shot_0000_00.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"fake-jpeg"


@pytest.mark.parametrize(
    "bad_name",
    [
        "../../etc/passwd",
        "..%2f..%2fsecret.jpg",
        "shot_0000_00.png",
        "not-a-frame.txt",
    ],
)
def test_frames_rejects_traversal_and_non_jpg(client, bad_name):
    job_id = _upload(client).json()["job_id"]
    _wait_status(client, job_id, "done")

    resp = client.get(f"/api/jobs/{job_id}/frames/{bad_name}")
    assert resp.status_code == 404


def test_narration_serves_wav(client):
    job_id = _upload(client).json()["job_id"]
    _wait_status(client, job_id, "done")

    resp = client.get(f"/api/jobs/{job_id}/narration/shot_0000.wav")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"fake-wav"


@pytest.mark.parametrize(
    "bad_name",
    [
        "../../etc/passwd",
        "..%2f..%2fsecret.wav",
        "shot_0000.mp3",
        "not-a-clip.txt",
    ],
)
def test_narration_rejects_traversal_and_non_wav(client, bad_name):
    job_id = _upload(client).json()["job_id"]
    _wait_status(client, job_id, "done")

    resp = client.get(f"/api/jobs/{job_id}/narration/{bad_name}")
    assert resp.status_code == 404
