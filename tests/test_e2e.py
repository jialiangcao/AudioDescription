import pytest

import pipeline
from pipeline import run_pipeline
from timeline import MIN_NARRATABLE_GAP_SEC


async def test_run_pipeline_end_to_end(
    job_blobs, monkeypatch, synthetic_video, fake_gemini_client, fake_kokoro
):
    """Runs the real pipeline glue in pipeline.run_pipeline() over a small
    synthetic clip, stubbing only the external ML/API boundaries (Gemini,
    VAD, Whisper, Kokoro) that would otherwise need network access or slow
    model downloads. Segmentation, audio extraction, timeline assembly, and
    narration synthesis all run for real.
    """
    monkeypatch.setattr(pipeline, "detect_speech_regions", lambda audio_path: [])
    monkeypatch.setattr(pipeline, "transcribe", lambda audio_path, regions: [])

    events = []

    async def on_event(event):
        events.append(event)

    timeline = await run_pipeline(
        synthetic_video, job_blobs, on_event, client=fake_gemini_client
    )

    assert timeline.job_id == job_blobs.job_id
    assert timeline.duration_sec == pytest.approx(4.0, abs=0.2)
    assert len(timeline.segments) == 2
    assert job_blobs.path("audio.wav").exists()

    for segment in timeline.segments:
        # Every sampled frame is described on its own — there is no shot-level
        # rollup. The Gemini call was stubbed, so each gets the same canned
        # analysis, but each frame carries its own timestamp inside the shot.
        assert segment.frames
        for expected_index, frame in enumerate(segment.frames):
            assert frame.index == expected_index
            assert segment.start <= frame.time <= segment.end
            assert frame.visual is not None
            assert frame.visual.description == "a test scene"
            assert frame.visual.actions == ["an object moves"]
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
            assert segment.ad_narration_key is not None
            assert job_blobs.path("narration/shot_0000.wav").exists()
            # 1.0s of synthesized speech + a 0.22s leading pad (10% of the 2.2s
            # shot) = 1.22s, still within the gap.
            assert segment.ad_narration_duration_sec == 1.22
            assert segment.ad_narration_overflow is False
        else:
            assert segment.ad_narration is None
            assert segment.ad_narration_key is None

    assert any(segment.ad_eligible for segment in timeline.segments)

    # Live progress was streamed: stage markers, the shot/frame skeleton, one
    # event per described frame, a full-timeline snapshot, and per-segment
    # narration audio all showed up as events.
    stages = {e["stage"] for e in events if e.get("type") == "stage"}
    assert {
        "segmentation",
        "vision",
        "audio",
        "timeline",
        "narration",
        "tts",
        "mux",
    } <= stages

    total_frames = sum(len(s.frames) for s in timeline.segments)
    skeletons = [e for e in events if e.get("type") == "shots"]
    assert len(skeletons) == 1
    assert [shot["id"] for shot in skeletons[0]["shots"]] == [0, 1]
    # The skeleton carries every frame's timestamp before any analysis lands.
    assert sum(len(shot["frames"]) for shot in skeletons[0]["shots"]) == total_frames

    frame_events = [e for e in events if e.get("type") == "frame"]
    assert len(frame_events) == total_frames
    assert {(e["shot_id"], e["index"]) for e in frame_events} == {
        (seg.id, frame.index) for seg in timeline.segments for frame in seg.frames
    }
    # Frame events carry job-relative blob keys, resolvable through JobBlobs.
    for event in frame_events:
        assert event["key"].startswith("frames/")
        assert job_blobs.path(event["key"]).exists()

    assert any(e.get("type") == "timeline" for e in events)
    assert any(e.get("type") == "narration_audio" for e in events)

    # The combined AD-only track was assembled, streamed, and recorded on the
    # timeline, spanning the whole video.
    assert timeline.ad_track_key == "narration/ad_track.wav"
    assert job_blobs.path("narration/ad_track.wav").exists()
    ad_track_events = [e for e in events if e.get("type") == "ad_track"]
    assert len(ad_track_events) == 1
    assert ad_track_events[0]["audio"] == "narration/ad_track.wav"

    # …and muxed into a playable video carrying both the original sound and the
    # narration, which is what the frontend serves up at the end.
    assert timeline.described_key == "described.mp4"
    assert job_blobs.path("described.mp4").exists()
    described_events = [e for e in events if e.get("type") == "described_video"]
    assert len(described_events) == 1
    assert described_events[0]["video"] == "described.mp4"


async def test_run_pipeline_skips_mux_without_narration(
    job_blobs, monkeypatch, synthetic_video, fake_gemini_client, fake_kokoro
):
    """No narration clips means nothing to mix: the mux stage closes out without
    a described video rather than handing ffmpeg an empty track."""
    monkeypatch.setattr(pipeline, "detect_speech_regions", lambda audio_path: [])
    monkeypatch.setattr(pipeline, "transcribe", lambda audio_path, regions: [])
    monkeypatch.setattr(pipeline, "build_ad_track", lambda timeline, blobs: None)

    events = []

    async def on_event(event):
        events.append(event)

    timeline = await run_pipeline(
        synthetic_video, job_blobs, on_event, client=fake_gemini_client
    )

    assert timeline.described_key is None
    assert not job_blobs.path("described.mp4").exists()
    assert not any(e.get("type") == "described_video" for e in events)
    mux_done = [
        e for e in events if e.get("stage") == "mux" and e.get("status") == "done"
    ]
    assert mux_done and mux_done[0]["described"] is False


async def test_run_pipeline_handles_video_without_audio(
    job_blobs, monkeypatch, synthetic_video, fake_gemini_client, fake_kokoro
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

    events = []

    async def on_event(event):
        events.append(event)

    timeline = await run_pipeline(
        synthetic_video, job_blobs, on_event, client=fake_gemini_client
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
