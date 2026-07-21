import pytest

import main
import vision_analysis
from timeline import MIN_NARRATABLE_GAP_SEC, load_timeline


def test_process_video_end_to_end(
    tmp_path, monkeypatch, synthetic_video, fake_gemini_client
):
    """Runs the real pipeline glue in main.process_video() over a small
    synthetic clip, stubbing only the external ML/API boundaries (Gemini,
    VAD, Whisper) that would otherwise need network access or slow model
    downloads. Segmentation, audio extraction, and timeline assembly all run
    for real.
    """
    monkeypatch.setattr(vision_analysis.genai, "Client", lambda: fake_gemini_client)
    monkeypatch.setattr(main, "detect_speech_regions", lambda audio_path: [])
    monkeypatch.setattr(main, "transcribe", lambda audio_path, speech_regions: [])

    frames_dir = tmp_path / "frames"
    audio_path = tmp_path / "audio.wav"
    timeline_path = tmp_path / "timeline.json"

    timeline = main.process_video(
        synthetic_video,
        frames_dir=str(frames_dir),
        audio_path=str(audio_path),
        timeline_path=str(timeline_path),
    )

    assert timeline.video_id == synthetic_video
    assert timeline.duration_sec == pytest.approx(4.0, abs=0.2)
    assert len(timeline.segments) == 2
    assert audio_path.exists()
    assert timeline_path.exists()

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
        else:
            assert segment.ad_narration is None

    assert any(segment.ad_eligible for segment in timeline.segments)

    reloaded = load_timeline(str(timeline_path))
    assert reloaded == timeline
