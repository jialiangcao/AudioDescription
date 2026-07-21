from timeline import AudioAnalysis, Segment, Timeline, VisualAnalysis
from vision_analysis import (
    _neighbor_transcript,
    analyze_keyframe,
    analyze_shots,
    fill_narration_gaps,
)


def _segment(
    id_, transcript, ad_eligible=None, narratable_gap_sec=None, keyframe="k.jpg"
):
    return Segment(
        id=id_,
        start=float(id_),
        end=float(id_ + 1),
        keyframe=keyframe,
        visual=VisualAnalysis(
            description="d", entities=[], setting="s", on_screen_text=None
        ),
        audio=AudioAnalysis(
            has_speech=bool(transcript), transcript=transcript, silence_ratio=0.0
        ),
        ad_eligible=ad_eligible,
        narratable_gap_sec=narratable_gap_sec,
    )


def test_neighbor_transcript_prefers_previous_segment():
    segments = [_segment(0, "before"), _segment(1, None), _segment(2, "after")]
    assert _neighbor_transcript(segments, 1) == "before"


def test_neighbor_transcript_falls_back_to_next_segment():
    segments = [_segment(0, None), _segment(1, None), _segment(2, "after")]
    assert _neighbor_transcript(segments, 1) == "after"


def test_neighbor_transcript_none_when_no_neighbors_have_transcript():
    segments = [_segment(0, None), _segment(1, None), _segment(2, None)]
    assert _neighbor_transcript(segments, 1) is None


async def test_analyze_keyframe_parses_gemini_response(tmp_path, fake_gemini_client):
    keyframe = tmp_path / "frame.jpg"
    keyframe.write_bytes(b"fake-image-bytes")

    result = await analyze_keyframe(fake_gemini_client, str(keyframe))

    assert result == {
        "description": "a test scene",
        "entities": ["object"],
        "setting": "a test setting",
        "on_screen_text": None,
    }


async def test_analyze_shots_populates_visual_field(tmp_path, fake_gemini_client):
    keyframe = tmp_path / "frame.jpg"
    keyframe.write_bytes(b"fake-image-bytes")

    shots = await analyze_shots(
        [{"id": 0, "keyframe": str(keyframe)}], client=fake_gemini_client
    )

    assert shots[0]["visual"]["description"] == "a test scene"
    assert shots[0]["visual"]["entities"] == ["object"]


async def test_analyze_shots_streams_each_shot_via_callback(
    tmp_path, fake_gemini_client
):
    keyframe = tmp_path / "frame.jpg"
    keyframe.write_bytes(b"fake-image-bytes")

    seen = []

    async def on_shot(shot):
        seen.append(shot["id"])

    shots = [{"id": i, "keyframe": str(keyframe)} for i in range(3)]
    await analyze_shots(shots, on_shot=on_shot, client=fake_gemini_client)

    # every shot is reported, regardless of completion order
    assert sorted(seen) == [0, 1, 2]


async def test_fill_narration_gaps_only_fills_eligible_segments(
    tmp_path, fake_gemini_client
):
    keyframe = tmp_path / "frame.jpg"
    keyframe.write_bytes(b"fake-image-bytes")

    eligible = _segment(
        0, None, ad_eligible=True, narratable_gap_sec=2.0, keyframe=str(keyframe)
    )
    ineligible = _segment(
        1, "dialogue", ad_eligible=False, narratable_gap_sec=0.0, keyframe=str(keyframe)
    )
    tl = Timeline(video_id="v", duration_sec=2.0, segments=[eligible, ineligible])

    result = await fill_narration_gaps(tl, client=fake_gemini_client)

    assert result.segments[0].ad_narration == "A quiet moment unfolds on screen."
    assert result.segments[1].ad_narration is None
