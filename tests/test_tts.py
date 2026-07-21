import numpy as np

import tts
from timeline import AudioAnalysis, Segment, Timeline, VisualAnalysis


def _segment(id_, ad_eligible, ad_narration, narratable_gap_sec):
    return Segment(
        id=id_,
        start=float(id_),
        end=float(id_ + 1),
        keyframe="k.jpg",
        visual=VisualAnalysis(
            description="d", entities=[], setting="s", on_screen_text=None
        ),
        audio=AudioAnalysis(has_speech=False, transcript=None, silence_ratio=1.0),
        ad_eligible=ad_eligible,
        narratable_gap_sec=narratable_gap_sec,
        ad_narration=ad_narration,
    )


class FakePipeline:
    """Stand-in for kokoro.KPipeline that avoids loading the real model.

    `durations_by_call` gives the audio length (in samples) to return on each
    successive call, so tests can control whether a call overflows its gap.
    """

    def __init__(self, durations_by_call):
        self.durations_by_call = list(durations_by_call)
        self.calls = []

    def __call__(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        n_samples = self.durations_by_call.pop(0)
        yield "gs", "ps", np.zeros(n_samples, dtype=np.float32)


def _install_fake_pipeline(monkeypatch, durations_by_call):
    fake = FakePipeline(durations_by_call)
    monkeypatch.setattr(tts, "_load_pipeline", lambda: fake)
    return fake


async def test_synthesize_narration_skips_ineligible_and_unnarrated_segments(
    tmp_path, monkeypatch
):
    fake = _install_fake_pipeline(monkeypatch, [])

    ineligible = _segment(
        0, ad_eligible=False, ad_narration=None, narratable_gap_sec=0.0
    )
    no_narration = _segment(
        1, ad_eligible=True, ad_narration=None, narratable_gap_sec=2.0
    )
    tl = Timeline(video_id="v", duration_sec=2.0, segments=[ineligible, no_narration])

    result = await tts.synthesize_narration(tl, out_dir=str(tmp_path))

    assert fake.calls == []
    assert result.segments[0].ad_narration_audio is None
    assert result.segments[1].ad_narration_audio is None


async def test_synthesize_narration_happy_path_writes_wav_without_retry(
    tmp_path, monkeypatch
):
    # 1.0s of audio at SAMPLE_RATE fits comfortably inside a 2.0s gap.
    fake = _install_fake_pipeline(monkeypatch, [tts.SAMPLE_RATE])

    seg = _segment(
        0, ad_eligible=True, ad_narration="hello there", narratable_gap_sec=2.0
    )
    tl = Timeline(video_id="v", duration_sec=2.0, segments=[seg])

    result = await tts.synthesize_narration(tl, out_dir=str(tmp_path))

    assert len(fake.calls) == 1
    assert fake.calls[0] == ("hello there", tts.VOICE, 1.0)

    out_seg = result.segments[0]
    assert out_seg.ad_narration_audio == str(tmp_path / "shot_0000.wav")
    assert out_seg.ad_narration_duration_sec == 1.0
    assert out_seg.ad_narration_overflow is False
    assert (tmp_path / "shot_0000.wav").exists()


async def test_synthesize_narration_retries_at_higher_speed_when_overflowing(
    tmp_path, monkeypatch
):
    # First call: 3.0s of audio for a 2.0s gap -> overflow, expect a retry at
    # speed = min(MAX_SPEED, 3.0/2.0) = 1.3. Retry call: 1.5s, which fits.
    fake = _install_fake_pipeline(
        monkeypatch, [3 * tts.SAMPLE_RATE, int(1.5 * tts.SAMPLE_RATE)]
    )

    seg = _segment(
        0, ad_eligible=True, ad_narration="a long line", narratable_gap_sec=2.0
    )
    tl = Timeline(video_id="v", duration_sec=2.0, segments=[seg])

    result = await tts.synthesize_narration(tl, out_dir=str(tmp_path))

    assert len(fake.calls) == 2
    assert fake.calls[0] == ("a long line", tts.VOICE, 1.0)
    assert fake.calls[1] == ("a long line", tts.VOICE, tts.MAX_SPEED)

    out_seg = result.segments[0]
    assert out_seg.ad_narration_duration_sec == 1.5
    assert out_seg.ad_narration_overflow is False


async def test_synthesize_narration_flags_persistent_overflow_after_one_retry(
    tmp_path, monkeypatch
):
    # Both calls overflow a 2.0s gap even at MAX_SPEED -> only one retry attempt,
    # and the result is flagged rather than retried again.
    fake = _install_fake_pipeline(
        monkeypatch, [3 * tts.SAMPLE_RATE, int(2.5 * tts.SAMPLE_RATE)]
    )

    seg = _segment(
        0, ad_eligible=True, ad_narration="a very long line", narratable_gap_sec=2.0
    )
    tl = Timeline(video_id="v", duration_sec=2.0, segments=[seg])

    result = await tts.synthesize_narration(tl, out_dir=str(tmp_path))

    assert len(fake.calls) == 2
    out_seg = result.segments[0]
    assert out_seg.ad_narration_duration_sec == 2.5
    assert out_seg.ad_narration_overflow is True


async def test_synthesize_narration_streams_finished_segments(tmp_path, monkeypatch):
    _install_fake_pipeline(monkeypatch, [tts.SAMPLE_RATE, tts.SAMPLE_RATE])

    narrated_a = _segment(
        0, ad_eligible=True, ad_narration="one", narratable_gap_sec=2.0
    )
    skipped = _segment(1, ad_eligible=False, ad_narration=None, narratable_gap_sec=0.0)
    narrated_b = _segment(
        2, ad_eligible=True, ad_narration="two", narratable_gap_sec=2.0
    )
    tl = Timeline(
        video_id="v", duration_sec=3.0, segments=[narrated_a, skipped, narrated_b]
    )

    seen = []

    async def on_segment(segment):
        seen.append((segment.id, segment.ad_narration_audio))

    await tts.synthesize_narration(tl, out_dir=str(tmp_path), on_segment=on_segment)

    # Callback fires only for segments that were actually synthesized, in id order.
    assert [sid for sid, _ in seen] == [0, 2]
    assert all(path is not None for _, path in seen)
