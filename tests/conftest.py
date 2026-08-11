import os
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SQL_DIR = Path(__file__).parent / "sql"
MIGRATIONS_DIR = Path(__file__).parents[1] / "supabase" / "migrations"


@pytest.fixture(scope="session")
def database_url():
    """The Postgres to run repo tests against, or a skip if there isn't one.

    `docker compose up -d postgres` provides it locally and CI runs it as a
    service container; the rest of the suite needs no database, so tests that
    do are skipped rather than failing on a developer's machine.
    """
    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("no TEST_DATABASE_URL/DATABASE_URL; start docker compose postgres")
    return url


@pytest.fixture(scope="session")
def _migrated_db(database_url):
    """Apply the auth shim + every migration once per session."""
    import asyncio

    import asyncpg

    async def _apply():
        conn = await asyncpg.connect(database_url)
        try:
            # A fresh start each session keeps a half-applied schema from an
            # interrupted run out of the way.
            await conn.execute(
                "drop schema if exists public cascade; create schema public;"
            )
            await conn.execute((SQL_DIR / "auth_shim.sql").read_text())
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                await conn.execute(path.read_text())
        finally:
            await conn.close()

    asyncio.run(_apply())
    return database_url


@pytest.fixture
async def db(_migrated_db, no_redis_publish, monkeypatch):
    """A pool bound to the migrated test database, torn down per test."""
    import db as db_module

    monkeypatch.setenv("DATABASE_URL", _migrated_db)
    await db_module.close_pool()
    pool = await db_module.get_pool()
    await pool.execute("truncate jobs, job_events, timelines, qa_runs cascade")
    await pool.execute("truncate auth.users cascade")
    yield pool
    await db_module.close_pool()


@pytest.fixture
async def owner(db):
    """An auth user id that jobs can be created under."""
    owner_id = str(uuid.uuid4())
    await db.execute("insert into auth.users (id) values ($1)", uuid.UUID(owner_id))
    return owner_id


@pytest.fixture
def no_redis_publish(monkeypatch):
    """Keep repo writes from needing a live Redis.

    Fan-out is deliberately best-effort (see events.publish), so tests assert on
    what landed in Postgres; the Redis path has its own tests, which exercise
    the real function — hence this is pulled in by ``db`` rather than autouse.
    """
    import events

    published = []

    async def _capture(job_id, event):
        published.append((job_id, event))

    monkeypatch.setattr(events, "publish", _capture)
    return published


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """A tiny 4s clip: 2s red + 2s blue, with a silent mono audio track.

    Gives PySceneDetect one clear cut (~2s in) and audio_extract a real
    stream to pull, without needing a checked-in media fixture.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    out_path = tmp_path_factory.mktemp("media") / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono:d=4",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            "-r",
            "10",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return str(out_path)


@pytest.fixture(scope="session")
def single_shot_video(tmp_path_factory):
    """A tiny 3s solid-color clip: no camera cuts at all.

    Exercises detect_shots' no-cuts fallback, which reports the whole video as
    one shot from PySceneDetect's `video.duration` rather than a scene list.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    out_path = tmp_path_factory.mktemp("media") / "single_shot.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x240:d=3",
            "-c:v",
            "libx264",
            "-r",
            "10",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return str(out_path)


class FakeGeminiResponse:
    """Minimal stand-in for a genai response: `.text`, `.function_calls`, and
    `.candidates[0].content` (what the QA agents append back into contents)."""

    def __init__(self, text, function_calls=None):
        self.text = text
        self.function_calls = function_calls
        self.candidates = [
            SimpleNamespace(content=SimpleNamespace(role="model", text=text))
        ]


class FakeFunctionCall:
    """Shape-compatible with types.FunctionCall (`.name` / `.args`)."""

    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


def _canned_response(config):
    """Schema-aware canned output, dispatched on the response_schema's name.

    Vision/frame-analysis calls get FrameAnalysis JSON; the QA agents' schemas
    get a minimal valid instance; schemaless calls get a canned sentence."""
    schema = getattr(config, "response_schema", None)
    schema_name = getattr(schema, "__name__", "")
    if schema_name == "CoreDecision":
        return FakeGeminiResponse(
            '{"reason": "enough information", "agent": "finish", '
            '"answer": "a canned answer"}'
        )
    if schema_name == "ReflectionAssessment":
        return FakeGeminiResponse('{"credible": true, "comment": null}')
    if schema_name == "Judgement":
        return FakeGeminiResponse(
            '{"relevance_score": 1, "clip_caption": "an unrelated scene", '
            '"reasoning": null}'
        )
    if schema is not None:
        return FakeGeminiResponse(
            '{"description": "a test scene", "entities": ["object"], '
            '"actions": ["an object moves"], "setting": "a test setting", '
            '"on_screen_text": null}'
        )
    return FakeGeminiResponse("A quiet moment unfolds on screen.")


def _snapshot(contents):
    """Agents mutate their contents list between turns; copy it so each
    recorded call keeps what was actually sent."""
    return list(contents) if isinstance(contents, list) else contents


class _FakeAsyncModels:
    def __init__(self, client):
        self._client = client

    async def generate_content(self, model, contents, config=None):
        self._client.calls.append((model, _snapshot(contents), config))
        if self._client.scripted:
            return self._client.scripted.pop(0)
        return _canned_response(config)


class _FakeAio:
    def __init__(self, client):
        self.models = _FakeAsyncModels(client)


class FakeGeminiClient:
    """Stand-in for google.genai.Client that avoids real network calls.

    Exposes both the sync ``.models.generate_content`` and the async
    ``.aio.models.generate_content`` surfaces. ``queue()`` scripts responses
    (consumed FIFO by either surface) ahead of the canned fallback.
    """

    def __init__(self):
        self.models = self
        self.calls = []
        self.scripted = []
        self.aio = _FakeAio(self)

    def queue(self, *responses):
        self.scripted.extend(responses)
        return self

    def generate_content(self, model, contents, config=None):
        self.calls.append((model, _snapshot(contents), config))
        if self.scripted:
            return self.scripted.pop(0)
        return _canned_response(config)


@pytest.fixture
def fake_gemini_client():
    return FakeGeminiClient()


@pytest.fixture
def job_blobs(tmp_path):
    """A local-only JobBlobs (no bucket) rooted at a fresh scratch dir.

    Every stage writes through this, so a test's artifacts land under tmp_path
    exactly as they would in a worker's scratch, and ``fetch`` is a plain
    lookup rather than a download.
    """
    from blobs import JobBlobs

    return JobBlobs("test-job", root=tmp_path / "job")


@pytest.fixture(scope="session")
def _qa_scratch(tmp_path_factory):
    """Scratch dir shared by ``fake_frame_index`` and ``fake_blobs``."""
    return tmp_path_factory.mktemp("qa-job")


@pytest.fixture(scope="session")
def fake_frame_index(_qa_scratch):
    """A synthetic 100s FrameIndex: one frame every 2s (t=0,2,...,98).

    Each entry is a blob key backed by a tiny fake jpg in the matching scratch
    dir (the vision tools read the bytes through ``fake_blobs``).
    """
    from qa.frame_index import FrameIndex

    entries = []
    for t in range(0, 100, 2):
        key = f"frames/shot_{t:04d}.jpg"
        path = _qa_scratch / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpeg")
        entries.append((float(t), key))
    return FrameIndex(entries=entries, duration_sec=100.0)


@pytest.fixture(scope="session")
def fake_blobs(_qa_scratch):
    """A local-only JobBlobs rooted where ``fake_frame_index`` wrote its jpgs."""
    from blobs import JobBlobs

    return JobBlobs("qa-test-job", root=_qa_scratch)


@pytest.fixture
def stub_retriever(monkeypatch):
    """Replace CLIP retrieval with a deterministic ranking (input order,
    descending fake scores). Returns the list of (keys, cue, top_k) calls."""
    import qa.retriever
    import qa.tools_perception

    calls = []

    async def _fake_retrieve_top_k(frame_keys, cue, top_k, blobs):
        calls.append((list(frame_keys), cue, top_k))
        k = min(top_k, len(frame_keys))
        return [(frame_keys[i], 1.0 - i * 0.01) for i in range(k)]

    monkeypatch.setattr(qa.retriever, "retrieve_top_k", _fake_retrieve_top_k)
    monkeypatch.setattr(
        qa.tools_perception.retriever, "retrieve_top_k", _fake_retrieve_top_k
    )
    return calls


class FakeKokoroPipeline:
    """Stand-in for kokoro.KPipeline that avoids downloading the real model.

    Returns 1s of silence per call — short enough to fit the synthetic clip's
    ~2s gaps without triggering the overflow retry — so tts.synthesize_narration
    runs its real file-writing/streaming glue over stubbed inference.
    """

    def __call__(self, text, voice, speed):
        yield "gs", "ps", np.zeros(24000, dtype=np.float32)


@pytest.fixture
def fake_kokoro(monkeypatch):
    import tts

    monkeypatch.setattr(tts, "_load_pipeline", lambda: FakeKokoroPipeline())
