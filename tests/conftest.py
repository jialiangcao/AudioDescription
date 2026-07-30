import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


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


@pytest.fixture(scope="session")
def fake_frame_index(tmp_path_factory):
    """A synthetic 100s FrameIndex: one frame every 2s (t=0,2,...,98), each
    path backed by a tiny fake jpg (the vision tools read the bytes)."""
    from qa.frame_index import FrameIndex

    frames_dir = tmp_path_factory.mktemp("qa-frames")
    entries = []
    for t in range(0, 100, 2):
        path = frames_dir / f"shot_{t:04d}.jpg"
        path.write_bytes(b"fake-jpeg")
        entries.append((float(t), str(path)))
    return FrameIndex(entries=entries, duration_sec=100.0)


@pytest.fixture
def stub_retriever(monkeypatch):
    """Replace CLIP retrieval with a deterministic ranking (input order,
    descending fake scores). Returns the list of (paths, cue, top_k) calls."""
    import qa.retriever
    import qa.tools_perception

    calls = []

    async def _fake_retrieve_top_k(frame_paths, cue, top_k):
        calls.append((list(frame_paths), cue, top_k))
        k = min(top_k, len(frame_paths))
        return [(frame_paths[i], 1.0 - i * 0.01) for i in range(k)]

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
