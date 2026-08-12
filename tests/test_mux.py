import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

import mux
from mux import DESCRIBED_KEY, LIMIT, NARRATION_GAIN, mux_described_video

# The AD track is written at Kokoro's rate by ad_track.py; the mux resamples it.
AD_SAMPLE_RATE = 24000

# Distinct test tones so the source and the narration can be told apart in the
# muxed output by measuring each frequency's amplitude.
SOURCE_HZ = 440
NARRATION_HZ = 300

# The AD track is mono and the output is stereo. ffmpeg's upmix holds total power
# constant, so a mono input lands at 1/√2 of its amplitude in each channel. It
# applies equally to the source audio, so the narration/source balance is
# unaffected — but a per-channel amplitude check has to account for it.
MONO_TO_STEREO = 2**-0.5


@pytest.fixture(scope="session")
def tone_video(tmp_path_factory):
    """An 8s clip whose soundtrack is a steady 440Hz tone."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    out_path = tmp_path_factory.mktemp("media") / "tone.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=8",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={SOURCE_HZ}:r=48000:d=8",
            "-c:v",
            "libx264",
            "-r",
            "10",
            "-c:a",
            "aac",
            "-shortest",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return str(out_path)


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory):
    """An 8s clip with no audio stream at all."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on PATH")

    out_path = tmp_path_factory.mktemp("media") / "mute.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x240:d=8",
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


def _write_ad_track(path, total_sec, burst=(1.0, 2.0), amplitude=0.35):
    """An AD track of ``total_sec`` holding one narration burst."""
    n = int(round(total_sec * AD_SAMPLE_RATE))
    track = np.zeros(n, dtype=np.float32)
    start, end = (int(round(t * AD_SAMPLE_RATE)) for t in burst)
    k = np.arange(end - start)
    track[start:end] = amplitude * np.sin(2 * np.pi * NARRATION_HZ * k / AD_SAMPLE_RATE)
    sf.write(str(path), track, AD_SAMPLE_RATE)
    return str(path)


def _decode_audio(video_path, out_path):
    """Decode the muxed soundtrack to float32 samples (left channel)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-c:a", "pcm_f32le", str(out_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    audio, rate = sf.read(str(out_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, rate


def _tone_amplitude(audio, rate, t0, t1, freq):
    """Amplitude of ``freq`` within ``[t0, t1)`` — a single-bin Goertzel-style DFT."""
    window = audio[int(t0 * rate) : int(t1 * rate)]
    k = np.arange(len(window))
    return 2 * abs((window * np.exp(-2j * np.pi * freq * k / rate)).sum()) / len(window)


def _duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def test_missing_ad_track_raises(tmp_path, tone_video):
    with pytest.raises(FileNotFoundError):
        mux_described_video(
            tone_video, str(tmp_path / "nope.wav"), str(tmp_path / DESCRIBED_KEY)
        )


def test_narration_is_mixed_in_and_ducks_the_source(tmp_path, tone_video):
    ad_track = _write_ad_track(tmp_path / "ad_track.wav", 8.0)
    out_path = tmp_path / "sub" / DESCRIBED_KEY

    result = mux_described_video(tone_video, ad_track, str(out_path))

    assert result == str(out_path)
    assert out_path.is_file()

    audio, rate = _decode_audio(str(out_path), tmp_path / "decoded.wav")

    # During the narration burst (1-2s): the narration is audible and the source
    # tone is pulled well down. Outside it, the source plays at full level.
    quiet = _tone_amplitude(audio, rate, 5.0, 6.0, SOURCE_HZ)
    ducked = _tone_amplitude(audio, rate, 1.3, 1.7, SOURCE_HZ)
    narration = _tone_amplitude(audio, rate, 1.3, 1.7, NARRATION_HZ)

    assert quiet > 0.01, "source soundtrack should be present outside narration"
    assert narration > ducked, "narration should sit above the ducked source"
    duck_db = 20 * np.log10(ducked / quiet)
    assert -20 < duck_db < -5, f"expected a moderate duck, got {duck_db:.1f}dB"

    # The soundtrack recovers after the line ends rather than staying ducked.
    after = _tone_amplitude(audio, rate, 2.6, 3.0, SOURCE_HZ)
    assert after > 0.7 * quiet


def test_narration_is_played_at_the_configured_gain(tmp_path, tone_video):
    """The narration is amplified by NARRATION_GAIN on its way into the mix,
    while the duck depth stays keyed off the *ungained* track."""
    amplitude = 0.35
    ad_track = _write_ad_track(tmp_path / "ad_track.wav", 8.0, amplitude=amplitude)
    out_path = tmp_path / DESCRIBED_KEY

    mux_described_video(tone_video, ad_track, str(out_path))
    audio, rate = _decode_audio(str(out_path), tmp_path / "decoded.wav")

    narration = _tone_amplitude(audio, rate, 1.3, 1.7, NARRATION_HZ)
    expected = amplitude * NARRATION_GAIN * MONO_TO_STEREO
    # Codec losses move this a little, hence the tolerance.
    assert narration == pytest.approx(expected, rel=0.1)


def test_hot_narration_is_limited_rather_than_clipped(tmp_path, tone_video):
    """A loud line times NARRATION_GAIN would overshoot full scale; the limiter
    has to hold the mix in range instead of letting it clip."""
    ad_track = _write_ad_track(tmp_path / "ad_track.wav", 8.0, amplitude=0.9)
    out_path = tmp_path / DESCRIBED_KEY

    mux_described_video(tone_video, ad_track, str(out_path))
    audio, _ = _decode_audio(str(out_path), tmp_path / "decoded.wav")

    assert float(np.abs(audio).max()) <= LIMIT + 0.02


def test_short_ad_track_does_not_truncate_the_video(tmp_path, tone_video):
    """The AD track can be shorter than the video (e.g. if the last shot has no
    narration); the mux must still keep the full picture and soundtrack."""
    ad_track = _write_ad_track(tmp_path / "ad_track.wav", 3.0)
    out_path = tmp_path / DESCRIBED_KEY

    mux_described_video(tone_video, ad_track, str(out_path))

    assert _duration(out_path) == pytest.approx(8.0, abs=0.2)
    audio, rate = _decode_audio(str(out_path), tmp_path / "decoded.wav")
    # Source audio still playing past the end of the AD track.
    assert _tone_amplitude(audio, rate, 6.0, 7.0, SOURCE_HZ) > 0.01


def test_source_without_audio_gets_the_narration_as_its_soundtrack(
    tmp_path, silent_video
):
    ad_track = _write_ad_track(tmp_path / "ad_track.wav", 8.0)
    out_path = tmp_path / DESCRIBED_KEY

    mux_described_video(silent_video, ad_track, str(out_path))

    audio, rate = _decode_audio(str(out_path), tmp_path / "decoded.wav")
    assert _tone_amplitude(audio, rate, 1.3, 1.7, NARRATION_HZ) > 0.1
    assert _tone_amplitude(audio, rate, 5.0, 6.0, NARRATION_HZ) < 0.01


def test_falls_back_to_reencoding_when_stream_copy_fails(
    tmp_path, tone_video, monkeypatch
):
    """Not every source codec can be stream-copied into mp4; the second attempt
    re-encodes the video instead of failing the job."""
    ad_track = _write_ad_track(tmp_path / "ad_track.wav", 8.0)
    out_path = tmp_path / DESCRIBED_KEY

    real_run = subprocess.run
    attempts = []

    def _fake_run(args, **kwargs):
        if args[0] != "ffmpeg":  # leave the ffprobe audio-stream check alone
            return real_run(args, **kwargs)
        attempts.append(args)
        if "copy" in args:  # pretend the container rejected the source codec
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="bad codec")
        return real_run(args, **kwargs)

    monkeypatch.setattr(mux.subprocess, "run", _fake_run)

    assert mux_described_video(tone_video, ad_track, str(out_path)) == str(out_path)
    assert out_path.is_file()
    assert len(attempts) == 2
    assert "libx264" in attempts[1]


def test_raises_when_reencode_also_fails(tmp_path, tone_video, monkeypatch):
    ad_track = _write_ad_track(tmp_path / "ad_track.wav", 8.0)

    real_run = subprocess.run

    def _fake_run(args, **kwargs):
        if args[0] != "ffmpeg":  # leave the ffprobe audio-stream check alone
            return real_run(args, **kwargs)
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")

    monkeypatch.setattr(mux.subprocess, "run", _fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        mux_described_video(tone_video, ad_track, str(tmp_path / DESCRIBED_KEY))
