import subprocess

import pytest

from audio_extract import NoAudioStreamError, extract_audio, has_audio_stream


def _fake_run_returning(stdout):
    def _fake_run(args, capture_output, text, check):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    return _fake_run


def test_has_audio_stream_true_when_ffprobe_reports_a_stream(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", _fake_run_returning('{"streams": [{"index": 0}]}')
    )
    assert has_audio_stream("video.mp4") is True


def test_has_audio_stream_false_when_no_streams(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run_returning('{"streams": []}'))
    assert has_audio_stream("video.mp4") is False


def test_extract_audio_raises_when_no_audio_stream(monkeypatch, tmp_path):
    monkeypatch.setattr("audio_extract.has_audio_stream", lambda video_path: False)
    with pytest.raises(NoAudioStreamError):
        extract_audio("video.mp4", out_path=str(tmp_path / "out.wav"))


def test_extract_audio_invokes_ffmpeg_and_creates_output_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("audio_extract.has_audio_stream", lambda video_path: True)
    captured = {}

    def _fake_run(args, capture_output, text, check):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    out_path = tmp_path / "nested" / "out.wav"
    result = extract_audio("video.mp4", out_path=str(out_path), sample_rate=16000)

    assert result == str(out_path)
    assert out_path.parent.is_dir()

    args = captured["args"]
    assert args[0] == "ffmpeg"
    assert "video.mp4" in args
    assert str(out_path) in args
    assert "16000" in args
