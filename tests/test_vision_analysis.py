from timeline import AudioAnalysis, Frame, FrameAnalysis, Segment, Timeline
from vision_analysis import (
    _current_shot_block,
    _estimated_speech_sec,
    _prior_scenes_block,
    analyze_frame,
    analyze_shots,
    fill_narration_gaps,
    optimize_narration,
    retry_optimize_narration,
)


def _prompt_of(call):
    """The text prompt from a recorded (model, contents, config) Gemini call."""
    _model, contents, _config = call
    return contents[-1]


def _images_of(call):
    """The image parts from a recorded (model, contents, config) Gemini call."""
    _model, contents, _config = call
    return contents[:-1]


def _inline_optimize_calls(client):
    """Recorded calls that used the inline optimization prompt."""
    return [c for c in client.calls if "ORIGINAL DESCRIPTIONS" in _prompt_of(c)]


def _frame(
    index,
    time,
    key="frames/k.jpg",
    description="d",
    actions=None,
    on_screen_text=None,
):
    return Frame(
        index=index,
        time=time,
        key=key,
        visual=FrameAnalysis(
            description=description,
            entities=[],
            actions=actions or [],
            setting="s",
            on_screen_text=on_screen_text,
        ),
    )


def _segment(
    id_,
    transcript,
    ad_eligible=None,
    narratable_gap_sec=None,
    keyframe="frames/k.jpg",
    frame_count=2,
):
    return Segment(
        id=id_,
        start=float(id_),
        end=float(id_ + 1),
        frames=[
            _frame(i, float(id_) + i * 0.5, key=keyframe, description=f"d{i}")
            for i in range(frame_count)
        ],
        audio=AudioAnalysis(
            has_speech=bool(transcript), transcript=transcript, silence_ratio=0.0
        ),
        ad_eligible=ad_eligible,
        narratable_gap_sec=narratable_gap_sec,
    )


def test_current_shot_block_lists_every_frame_in_order_with_timestamps():
    seg = Segment(
        id=0,
        start=0.0,
        end=4.0,
        frames=[
            _frame(0, 0.0, description="a lit room", actions=["a door swings open"]),
            _frame(1, 2.0, description="Maria at the desk", on_screen_text="EXIT"),
        ],
        audio=AudioAnalysis(has_speech=True, transcript="hello", silence_ratio=0.0),
    )
    block = _current_shot_block(seg)

    # Every frame contributes its own timestamped line, oldest first.
    assert block.index("0.00s: a lit room") < block.index("2.00s: Maria at the desk")
    assert "Actions: a door swings open." in block
    assert 'On-screen text: "EXIT"' in block
    assert 'Dialogue in this shot (do not repeat): "hello"' in block


def test_current_shot_block_marks_frames_still_awaiting_analysis():
    seg = Segment(
        id=0,
        start=0.0,
        end=2.0,
        frames=[Frame(index=0, time=0.0, key="frames/k.jpg", visual=None)],
    )
    assert "0.00s: (not analyzed)" in _current_shot_block(seg)


def test_prior_scenes_block_is_empty_marker_when_no_history():
    assert "none" in _prior_scenes_block([]).lower()


def test_prior_scenes_block_lists_frame_descriptions_dialogue_and_narration():
    prior = [
        {
            "id": 0,
            "descriptions": ["Maria enters", "Maria sits"],
            "dialogue": "hi",
            "narration": "Maria waves",
        },
        {
            "id": 1,
            "descriptions": ["an empty hall"],
            "dialogue": None,
            "narration": None,
        },
    ]
    block = _prior_scenes_block(prior)
    # Each earlier shot is summarized by its frame descriptions, in order.
    assert "Shot 0: Maria enters → Maria sits" in block
    assert 'Dialogue: "hi"' in block
    assert 'AD: "Maria waves"' in block
    assert "Shot 1: an empty hall" in block


async def test_analyze_frame_parses_gemini_response(tmp_path, fake_gemini_client):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"fake-image-bytes")

    result = await analyze_frame(fake_gemini_client, str(frame_path), frame_time=1.5)

    assert result == {
        "description": "a test scene",
        "entities": ["object"],
        "actions": ["an object moves"],
        "setting": "a test setting",
        "on_screen_text": None,
    }
    # The frame's own timestamp is what the prompt is anchored to.
    assert "1.50 seconds" in _prompt_of(fake_gemini_client.calls[-1])


async def test_analyze_shots_analyzes_every_frame_not_just_one(
    job_blobs, fake_gemini_client
):
    job_blobs.path("frames/frame.jpg").write_bytes(b"fake-image-bytes")

    shots = [
        {
            "id": 0,
            "start": 0.0,
            "end": 4.0,
            "frames": [
                {"index": i, "time": i * 2.0, "key": "frames/frame.jpg"}
                for i in range(3)
            ],
        }
    ]

    shots = await analyze_shots(shots, job_blobs, client=fake_gemini_client)

    assert len(fake_gemini_client.calls) == 3
    for frame in shots[0]["frames"]:
        assert frame["visual"]["description"] == "a test scene"
        assert frame["visual"]["actions"] == ["an object moves"]


async def test_analyze_shots_streams_each_frame_via_callback(
    job_blobs, fake_gemini_client
):
    job_blobs.path("frames/frame.jpg").write_bytes(b"fake-image-bytes")

    seen = []

    async def on_frame(shot, frame):
        seen.append((shot["id"], frame["index"]))

    shots = [
        {
            "id": shot_id,
            "start": 0.0,
            "end": 2.0,
            "frames": [
                {"index": i, "time": float(i), "key": "frames/frame.jpg"}
                for i in range(2)
            ],
        }
        for shot_id in range(3)
    ]
    await analyze_shots(shots, job_blobs, on_frame=on_frame, client=fake_gemini_client)

    # every frame of every shot is reported, regardless of completion order
    assert sorted(seen) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


async def test_fill_narration_gaps_only_fills_eligible_segments(
    job_blobs, fake_gemini_client
):
    keyframe = "frames/frame.jpg"
    job_blobs.path(keyframe).write_bytes(b"fake-image-bytes")

    eligible = _segment(
        0, None, ad_eligible=True, narratable_gap_sec=2.0, keyframe=keyframe
    )
    ineligible = _segment(
        1, "dialogue", ad_eligible=False, narratable_gap_sec=0.0, keyframe=keyframe
    )
    tl = Timeline(job_id="v", duration_sec=2.0, segments=[eligible, ineligible])

    result = await fill_narration_gaps(tl, job_blobs, client=fake_gemini_client)

    # The eligible segment gets a narration line; the ineligible one does not.
    assert result.segments[0].ad_narration == "A quiet moment unfolds on screen."
    assert result.segments[1].ad_narration is None


async def test_fill_narration_gaps_sends_every_frame_of_the_shot(
    job_blobs, fake_gemini_client
):
    keyframe = "frames/frame.jpg"
    job_blobs.path(keyframe).write_bytes(b"fake-image-bytes")

    seg = _segment(
        0,
        None,
        ad_eligible=True,
        narratable_gap_sec=10.0,
        keyframe=keyframe,
        frame_count=3,
    )
    tl = Timeline(job_id="v", duration_sec=10.0, segments=[seg])

    await fill_narration_gaps(tl, job_blobs, client=fake_gemini_client)

    call = fake_gemini_client.calls[-1]
    # The narration line is written from the whole shot: one image part per
    # sampled frame, plus every frame's description in the prompt.
    assert len(_images_of(call)) == 3
    prompt = _prompt_of(call)
    assert "sampled at 3 frame(s)" in prompt
    for i in range(3):
        assert f"d{i}" in prompt


def test_estimated_speech_sec_scales_with_word_count():
    assert _estimated_speech_sec("one two three four five", 2.5) == 2.0
    assert _estimated_speech_sec("", 2.5) == 0.0


async def test_optimize_narration_uses_inline_prompt(fake_gemini_client):
    out = await optimize_narration(fake_gemini_client, "a rather long line", 2.0)

    assert out == "A quiet moment unfolds on screen."
    prompt = _prompt_of(fake_gemini_client.calls[-1])
    # The paper's inline optimization prompt, filled with our text + gap.
    assert "ORIGINAL DESCRIPTIONS" in prompt
    assert "a rather long line" in prompt
    assert "2.00 seconds" in prompt


async def test_retry_optimize_narration_uses_retry_prompt(fake_gemini_client):
    out = await retry_optimize_narration(fake_gemini_client, "still too long", 3.5, 2.0)

    assert out == "A quiet moment unfolds on screen."
    prompt = _prompt_of(fake_gemini_client.calls[-1])
    # The paper's retry prompt, filled with the measured duration and overshoot.
    assert "PREVIOUS ATTEMPT" in prompt
    assert "still too long" in prompt
    assert "3.50 seconds to speak" in prompt
    assert "2.00 seconds are available" in prompt
    assert "Reduce by 1.50 seconds" in prompt


async def test_fill_narration_gaps_optimizes_when_estimate_exceeds_gap(
    job_blobs, fake_gemini_client
):
    keyframe = "frames/frame.jpg"
    job_blobs.path(keyframe).write_bytes(b"fake-image-bytes")

    # Canned narration is 6 words -> ~2.4s at 2.5 wps, over the 2.0s gap, so the
    # inline optimization pass should fire.
    seg = _segment(0, None, ad_eligible=True, narratable_gap_sec=2.0, keyframe=keyframe)
    tl = Timeline(job_id="v", duration_sec=2.0, segments=[seg])

    result = await fill_narration_gaps(tl, job_blobs, client=fake_gemini_client)

    assert len(_inline_optimize_calls(fake_gemini_client)) == 1
    # The optimized text replaces the generated line.
    assert result.segments[0].ad_narration == "A quiet moment unfolds on screen."


async def test_fill_narration_gaps_skips_optimization_when_line_fits(
    job_blobs, fake_gemini_client
):
    keyframe = "frames/frame.jpg"
    job_blobs.path(keyframe).write_bytes(b"fake-image-bytes")

    # A generous 10s gap easily fits the ~2.4s canned line -> no optimization.
    seg = _segment(
        0, None, ad_eligible=True, narratable_gap_sec=10.0, keyframe=keyframe
    )
    tl = Timeline(job_id="v", duration_sec=10.0, segments=[seg])

    result = await fill_narration_gaps(tl, job_blobs, client=fake_gemini_client)

    assert _inline_optimize_calls(fake_gemini_client) == []
    assert result.segments[0].ad_narration == "A quiet moment unfolds on screen."


async def test_fill_narration_gaps_feeds_prior_scenes_as_continuity_context(
    job_blobs, fake_gemini_client
):
    keyframe = "frames/frame.jpg"
    job_blobs.path(keyframe).write_bytes(b"fake-image-bytes")

    # Shot 0 has dialogue but isn't narrated; shot 1 is. Shot 1's narration prompt
    # should still carry shot 0's frame descriptions + dialogue as continuity
    # context.
    shot0 = _segment(
        0,
        "hello there",
        ad_eligible=False,
        narratable_gap_sec=0.0,
        keyframe=keyframe,
    )
    shot1 = _segment(
        1, None, ad_eligible=True, narratable_gap_sec=10.0, keyframe=keyframe
    )
    tl = Timeline(job_id="v", duration_sec=2.0, segments=[shot0, shot1])

    await fill_narration_gaps(tl, job_blobs, client=fake_gemini_client)

    # The single generation call (shot 1) is the only one carrying the narration
    # prompt; it must include shot 0 as an earlier shot — every one of its frame
    # descriptions, in order — along with its dialogue.
    gen_prompt = _prompt_of(fake_gemini_client.calls[-1])
    assert "EARLIER SHOTS" in gen_prompt
    assert "Shot 0: d0 → d1" in gen_prompt
    assert 'Dialogue: "hello there"' in gen_prompt
