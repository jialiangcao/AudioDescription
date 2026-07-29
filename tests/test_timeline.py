import pytest

import timeline as timeline_module
from timeline import (
    AudioAnalysis,
    Frame,
    FrameAnalysis,
    Segment,
    Timeline,
    _overlap_sec,
    _speech_seconds_within,
    _transcript_within,
    build_timeline,
    load_timeline,
    save_timeline,
)


def test_overlap_sec_no_overlap():
    assert _overlap_sec(0.0, 1.0, 2.0, 3.0) == 0.0


def test_overlap_sec_partial_overlap():
    assert _overlap_sec(0.0, 2.0, 1.0, 3.0) == 1.0


def test_overlap_sec_full_containment():
    assert _overlap_sec(0.0, 5.0, 1.0, 2.0) == 1.0


def test_speech_seconds_within_sums_multiple_regions():
    regions = [(0.0, 1.0), (2.0, 3.5), (10.0, 11.0)]
    assert _speech_seconds_within(0.0, 4.0, regions) == pytest.approx(2.5)


def test_transcript_within_joins_overlapping_segments_in_order():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 5.0, "end": 6.0, "text": "unrelated"},
        {"start": 1.5, "end": 2.0, "text": "world"},
    ]
    assert _transcript_within(0.0, 2.0, segments) == "hello world"


def test_transcript_within_returns_none_when_no_overlap():
    segments = [{"start": 5.0, "end": 6.0, "text": "unrelated"}]
    assert _transcript_within(0.0, 2.0, segments) is None


class _FakeCapture:
    def __init__(self, fps, frame_count):
        self._fps = fps
        self._frame_count = frame_count

    def get(self, prop):
        if prop == timeline_module.cv2.CAP_PROP_FPS:
            return self._fps
        if prop == timeline_module.cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        return 0

    def release(self):
        pass


def _frame(shot_id, index, time):
    return {
        "index": index,
        "time": time,
        "path": f"shot_{shot_id:04d}_{index:02d}.jpg",
        "visual": {
            "description": f"description {shot_id}.{index}",
            "entities": [],
            "actions": ["something happens"],
            "setting": "a setting",
            "on_screen_text": None,
        },
    }


def _shot(id_, start, end):
    return {
        "id": id_,
        "start": start,
        "end": end,
        "frames": [_frame(id_, 0, start), _frame(id_, 1, (start + end) / 2)],
    }


def test_build_timeline_computes_duration_and_ad_eligibility(monkeypatch):
    monkeypatch.setattr(
        timeline_module.cv2,
        "VideoCapture",
        lambda path: _FakeCapture(fps=10.0, frame_count=40),
    )

    shots = [_shot(0, 0.0, 2.2), _shot(1, 2.2, 4.0)]
    speech_regions = [(2.2, 3.0)]
    transcript_segments = [{"start": 2.2, "end": 3.0, "text": "hello"}]

    tl = build_timeline("video.mp4", shots, speech_regions, transcript_segments)

    assert tl.duration_sec == 4.0
    assert len(tl.segments) == 2

    first, second = tl.segments
    assert first.audio is not None
    assert first.audio.has_speech is False
    assert first.audio.transcript is None
    # fully silent 2.2s shot clears the 2.0s narratable-gap threshold
    assert first.ad_eligible is True

    assert second.audio is not None
    assert second.audio.has_speech is True
    assert second.audio.transcript == "hello"
    # 0.8s of speech in a 1.8s shot leaves only ~1.0s of gap, below threshold
    assert second.ad_eligible is False


def test_build_timeline_carries_every_frame_with_its_own_analysis(monkeypatch):
    monkeypatch.setattr(
        timeline_module.cv2,
        "VideoCapture",
        lambda path: _FakeCapture(fps=10.0, frame_count=40),
    )

    tl = build_timeline("video.mp4", [_shot(0, 0.0, 2.0), _shot(1, 2.0, 4.0)])

    for segment in tl.segments:
        assert [f.index for f in segment.frames] == [0, 1]
        for frame in segment.frames:
            # Each frame keeps its own timestamp and its own analysis — there is
            # no shot-level rollup.
            assert segment.start <= frame.time <= segment.end
            assert frame.visual is not None
            assert frame.visual.description == f"description {segment.id}.{frame.index}"
            assert frame.visual.actions == ["something happens"]


def test_build_timeline_sizes_gap_to_longest_silence_not_total(monkeypatch):
    monkeypatch.setattr(
        timeline_module.cv2,
        "VideoCapture",
        lambda path: _FakeCapture(fps=10.0, frame_count=50),
    )

    # A 5s shot chopped by three 0.6s speech bursts: total silence is 3.2s (which
    # would clear the 2.0s threshold under the old sum-of-silence calc), but no
    # single contiguous silent stretch exceeds 1.0s -> not narratable.
    shots = [_shot(0, 0.0, 5.0)]
    speech_regions = [(1.0, 1.6), (2.4, 3.0), (3.8, 4.4)]

    tl = build_timeline("video.mp4", shots, speech_regions, transcript_segments=[])

    seg = tl.segments[0]
    assert seg.narratable_gap_sec == pytest.approx(1.0)
    assert seg.ad_eligible is False


def test_save_and_load_timeline_roundtrip(tmp_path):
    tl = Timeline(
        video_id="video.mp4",
        duration_sec=4.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=2.0,
                frames=[
                    Frame(
                        index=0,
                        time=0.0,
                        path="shot_0000_00.jpg",
                        visual=FrameAnalysis(
                            description="d",
                            entities=[],
                            actions=["a hand lifts"],
                            setting="s",
                            on_screen_text=None,
                        ),
                    )
                ],
                audio=AudioAnalysis(
                    has_speech=False, transcript=None, silence_ratio=1.0
                ),
                ad_eligible=True,
                narratable_gap_sec=2.0,
            )
        ],
    )

    out_path = tmp_path / "timeline.json"
    save_timeline(tl, out_path=str(out_path))
    loaded = load_timeline(str(out_path))

    assert loaded == tl
