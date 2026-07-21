import shutil
import subprocess

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


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class FakeGeminiClient:
    """Stand-in for google.genai.Client that avoids real network calls.

    Shot-analysis calls (config.response_schema set) get back canned JSON
    matching ShotAnalysis; narration calls (no schema) get back a canned
    sentence.
    """

    def __init__(self):
        self.models = self
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append((model, contents, config))
        if getattr(config, "response_schema", None) is not None:
            return FakeGeminiResponse(
                '{"description": "a test scene", "entities": ["object"], '
                '"setting": "a test setting", "on_screen_text": null}'
            )
        return FakeGeminiResponse("A quiet moment unfolds on screen.")


@pytest.fixture
def fake_gemini_client():
    return FakeGeminiClient()
