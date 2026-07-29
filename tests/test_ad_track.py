import numpy as np
import soundfile as sf

from ad_track import AD_TRACK_FILENAME, build_ad_track
from timeline import AudioAnalysis, Frame, FrameAnalysis, Segment, Timeline
from tts import SAMPLE_RATE


def _segment(id_, start, end, *, narration_audio=None, start_sec=None, dur=None):
    return Segment(
        id=id_,
        start=start,
        end=end,
        frames=[
            Frame(
                index=0,
                time=start,
                path="k.jpg",
                visual=FrameAnalysis(
                    description="d",
                    entities=[],
                    actions=[],
                    setting="s",
                    on_screen_text=None,
                ),
            )
        ],
        audio=AudioAnalysis(has_speech=False, transcript=None, silence_ratio=1.0),
        ad_eligible=narration_audio is not None,
        narratable_gap_sec=dur,
        narration_start_sec=start_sec,
        ad_narration="line" if narration_audio else None,
        ad_narration_audio=narration_audio,
        ad_narration_duration_sec=dur,
    )


def _write_tone(path, seconds, value=0.5):
    n = int(round(seconds * SAMPLE_RATE))
    sf.write(str(path), np.full(n, value, dtype=np.float32), SAMPLE_RATE)


def test_returns_none_when_no_narration(tmp_path):
    tl = Timeline(
        video_id="v",
        duration_sec=5.0,
        segments=[_segment(0, 0.0, 5.0)],
    )
    assert build_ad_track(tl, str(tmp_path)) is None


def test_places_clips_at_their_gap_start(tmp_path):
    clip_a = tmp_path / "shot_0000.wav"
    clip_b = tmp_path / "shot_0001.wav"
    _write_tone(clip_a, 1.0, value=0.5)
    _write_tone(clip_b, 1.0, value=0.5)

    # Clip A plays at 2.0s, clip B at 7.0s, in a 10s video.
    tl = Timeline(
        video_id="v",
        duration_sec=10.0,
        segments=[
            _segment(0, 0.0, 5.0, narration_audio=str(clip_a), start_sec=2.0, dur=1.0),
            _segment(1, 5.0, 10.0, narration_audio=str(clip_b), start_sec=7.0, dur=1.0),
        ],
    )

    result = build_ad_track(tl, str(tmp_path))
    assert result is not None
    path, duration = result
    assert path.endswith(AD_TRACK_FILENAME)
    assert duration == 10.0

    track, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE
    assert len(track) == int(round(10.0 * SAMPLE_RATE))

    def loud_at(t):
        return abs(track[int(t * SAMPLE_RATE)]) > 0.1

    # Silence at the head, sound only during each clip's window.
    assert not loud_at(0.5)
    assert loud_at(2.5)  # inside clip A (2.0-3.0)
    assert not loud_at(4.0)
    assert loud_at(7.5)  # inside clip B (7.0-8.0)
    assert not loud_at(9.5)


def test_track_extends_past_video_for_overflowing_clip(tmp_path):
    clip = tmp_path / "shot_0000.wav"
    _write_tone(clip, 3.0)  # 3s clip starting at 4.0s runs to 7.0s, past a 5s video

    tl = Timeline(
        video_id="v",
        duration_sec=5.0,
        segments=[
            _segment(0, 0.0, 5.0, narration_audio=str(clip), start_sec=4.0, dur=3.0)
        ],
    )

    result = build_ad_track(tl, str(tmp_path))
    assert result is not None
    _, duration = result
    assert duration == 7.0
