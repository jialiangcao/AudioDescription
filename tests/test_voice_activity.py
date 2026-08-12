import sys

import numpy as np
import pytest

import voice_activity
from voice_activity import longest_speech_free_gap, longest_speech_free_span

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")


def _write_wav(path, rate=16000, seconds=0.5, channels=1):
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    sf.write(str(path), data, rate)
    return tone


def test_read_audio_returns_mono_float32_tensor(tmp_path):
    wav_path = tmp_path / "audio.wav"
    tone = _write_wav(wav_path)

    out = voice_activity._read_audio(str(wav_path), 16000)

    assert out.dtype == torch.float32
    assert out.ndim == 1
    assert len(out) == len(tone)
    assert np.allclose(out.numpy(), tone, atol=1e-4)


def test_read_audio_downmixes_stereo(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    tone = _write_wav(wav_path, channels=2)

    out = voice_activity._read_audio(str(wav_path), 16000)

    assert out.ndim == 1
    assert np.allclose(out.numpy(), tone, atol=1e-4)


def test_read_audio_resamples_mismatched_rate(tmp_path):
    wav_path = tmp_path / "audio8k.wav"
    _write_wav(wav_path, rate=8000, seconds=1.0)

    out = voice_activity._read_audio(str(wav_path), 16000)

    assert len(out) == 16000


def test_detect_speech_regions_does_not_import_torchaudio(tmp_path, monkeypatch):
    # Regression: silero's read_audio pulls in torchaudio -> TorchCodec, which
    # needs FFmpeg's shared libs and blew up at runtime in the media image.
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path)

    class _Poisoned:
        def __getattr__(self, name):
            raise RuntimeError("Could not load libtorchcodec")

    monkeypatch.setitem(sys.modules, "torchaudio", _Poisoned())
    monkeypatch.setattr(voice_activity, "_load_model", lambda: object())
    monkeypatch.setattr(
        "silero_vad.get_speech_timestamps",
        lambda wav, model, **kw: [{"start": 0.0, "end": 0.25}],
    )

    assert voice_activity.detect_speech_regions(str(wav_path)) == [(0.0, 0.25)]


def test_no_speech_returns_full_window():
    assert longest_speech_free_gap(0.0, 10.0, []) == pytest.approx(10.0)


def test_returns_longest_gap_not_total_silence():
    # Speech chops the window into pieces. Total silence is 1+1+1+1+2 = 6s, but
    # the longest *contiguous* silent stretch is only the final 2s (8.0-10.0).
    regions = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    assert longest_speech_free_gap(0.0, 10.0, regions) == pytest.approx(2.0)


def test_merges_overlapping_regions():
    # (1,5) and (3,7) overlap into one 1.0-7.0 span; longest gap is 7.0-10.0.
    assert longest_speech_free_gap(
        0.0, 10.0, [(1.0, 5.0), (3.0, 7.0)]
    ) == pytest.approx(3.0)


def test_clips_regions_extending_outside_window():
    # Regions spill past both edges; only the 3.0-7.0 gap inside the window counts.
    regions = [(0.0, 3.0), (7.0, 10.0)]
    assert longest_speech_free_gap(2.0, 8.0, regions) == pytest.approx(4.0)


def test_unsorted_regions_are_handled():
    assert longest_speech_free_gap(
        0.0, 10.0, [(5.0, 6.0), (1.0, 2.0)]
    ) == pytest.approx(4.0)


def test_empty_window_returns_zero():
    assert longest_speech_free_gap(5.0, 5.0, []) == 0.0
    assert longest_speech_free_gap(5.0, 4.0, []) == 0.0


def test_speech_filling_window_returns_zero():
    assert longest_speech_free_gap(0.0, 3.0, [(0.0, 3.0)]) == pytest.approx(0.0)


def test_span_reports_gap_start_and_length():
    # Longest silent stretch is 8.0-10.0; it begins right after the last burst.
    regions = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    start, length = longest_speech_free_span(0.0, 10.0, regions)
    assert start == pytest.approx(8.0)
    assert length == pytest.approx(2.0)


def test_span_gap_in_the_middle():
    # Silence runs 2.0-7.0 (between the two bursts), longer than the edges.
    start, length = longest_speech_free_span(0.0, 8.0, [(1.0, 2.0), (7.0, 7.5)])
    assert start == pytest.approx(2.0)
    assert length == pytest.approx(5.0)


def test_span_no_speech_starts_at_window_start():
    assert longest_speech_free_span(3.0, 9.0, []) == (
        pytest.approx(3.0),
        pytest.approx(6.0),
    )
