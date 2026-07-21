import pytest

import pipeline
from pipeline import run_pipeline
from timeline import MIN_NARRATABLE_GAP_SEC


async def test_run_pipeline_end_to_end(
    tmp_path, monkeypatch, synthetic_video, fake_gemini_client, fake_kokoro
):
    """Runs the real pipeline glue in pipeline.run_pipeline() over a small
    synthetic clip, stubbing only the external ML/API boundaries (Gemini,
    VAD, Whisper, Kokoro) that would otherwise need network access or slow
    model downloads. Segmentation, audio extraction, timeline assembly, and
    narration synthesis all run for real.
    """
    monkeypatch.setattr(pipeline, "detect_speech_regions", lambda audio_path: [])
    monkeypatch.setattr(pipeline, "transcribe", lambda audio_path, regions: [])

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    events = []

    async def on_event(event):
        events.append(event)

    timeline = await run_pipeline(
        synthetic_video, job_dir, on_event, client=fake_gemini_client
    )

    assert timeline.video_id == synthetic_video
    assert timeline.duration_sec == pytest.approx(4.0, abs=0.2)
    assert len(timeline.segments) == 2
    assert (job_dir / "audio.wav").exists()

    for segment in timeline.segments:
        # Gemini call was stubbed; every shot gets the same canned analysis.
        assert segment.visual.description == "a test scene"
        # VAD was stubbed to find no speech, so each shot is fully silent.
        assert segment.audio is not None
        assert segment.audio.has_speech is False
        assert segment.audio.silence_ratio == 1.0
        # ad_eligible must be consistent with the narratable-gap threshold.
        assert segment.narratable_gap_sec is not None
        assert segment.ad_eligible == (
            segment.narratable_gap_sec >= MIN_NARRATABLE_GAP_SEC
        )
        if segment.ad_eligible:
            assert segment.ad_narration == "A quiet moment unfolds on screen."
            # Narration was synthesized to a real WAV that fits the gap.
            assert segment.ad_narration_audio is not None
            assert (job_dir / "narration" / "shot_0000.wav").exists()
            assert segment.ad_narration_duration_sec == 1.0
            assert segment.ad_narration_overflow is False
        else:
            assert segment.ad_narration is None
            assert segment.ad_narration_audio is None

    assert any(segment.ad_eligible for segment in timeline.segments)

    # Live progress was streamed: stage markers, per-shot vision, a full-timeline
    # snapshot, and per-segment narration audio all showed up as events.
    stages = {e["stage"] for e in events if e.get("type") == "stage"}
    assert {
        "segmentation",
        "vision",
        "audio",
        "timeline",
        "narration",
        "tts",
    } <= stages
    assert sum(1 for e in events if e.get("type") == "shot") == 2
    assert any(e.get("type") == "timeline" for e in events)
    assert any(e.get("type") == "narration_audio" for e in events)


async def test_run_pipeline_handles_video_without_audio(
    tmp_path, monkeypatch, synthetic_video, fake_gemini_client, fake_kokoro
):
    """A source with no audio track skips VAD/transcription and still yields a
    full timeline (ad_eligible computed from shot duration alone)."""
    from audio_extract import NoAudioStreamError

    def _no_audio(video_path, out_path):
        raise NoAudioStreamError("no audio")

    monkeypatch.setattr(pipeline, "extract_audio", _no_audio)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("audio stages must be skipped when there is no audio")

    monkeypatch.setattr(pipeline, "detect_speech_regions", _should_not_run)
    monkeypatch.setattr(pipeline, "transcribe", _should_not_run)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    events = []

    async def on_event(event):
        events.append(event)

    timeline = await run_pipeline(
        synthetic_video, job_dir, on_event, client=fake_gemini_client
    )

    assert len(timeline.segments) == 2
    for segment in timeline.segments:
        assert segment.audio is not None
        assert segment.audio.has_speech is False
        assert segment.audio.silence_ratio == 1.0

    audio_done = [
        e for e in events if e.get("stage") == "audio" and e.get("status") == "done"
    ]
    assert audio_done and audio_done[0]["has_audio"] is False
