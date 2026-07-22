from timeline import AudioAnalysis, Segment, Timeline, VisualAnalysis
from vision_analysis import (
    _current_shot_block,
    _estimated_speech_sec,
    _prior_scenes_block,
    analyze_keyframe,
    analyze_shots,
    fill_narration_gaps,
    optimize_narration,
    retry_optimize_narration,
)


def _prompt_of(call):
    """The text prompt from a recorded (model, contents, config) Gemini call."""
    _model, contents, _config = call
    return contents[-1]


def _inline_optimize_calls(client):
    """Recorded calls that used the inline optimization prompt."""
    return [c for c in client.calls if "ORIGINAL DESCRIPTIONS" in _prompt_of(c)]


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


def test_current_shot_block_includes_dialogue_and_on_screen_text():
    seg = Segment(
        id=0,
        start=0.0,
        end=1.0,
        keyframe="k.jpg",
        visual=VisualAnalysis(
            description="a lit room", entities=[], setting="s", on_screen_text="EXIT"
        ),
        audio=AudioAnalysis(has_speech=True, transcript="hello", silence_ratio=0.0),
    )
    block = _current_shot_block(seg)
    assert "a lit room" in block
    assert 'On-screen text: "EXIT"' in block
    assert 'Dialogue in this shot (do not repeat): "hello"' in block


def test_prior_scenes_block_is_empty_marker_when_no_history():
    assert "none" in _prior_scenes_block([]).lower()


def test_prior_scenes_block_lists_description_dialogue_and_narration():
    prior = [
        {
            "id": 0,
            "description": "Maria enters",
            "dialogue": "hi",
            "narration": "Maria waves",
        },
        {"id": 1, "description": "an empty hall", "dialogue": None, "narration": None},
    ]
    block = _prior_scenes_block(prior)
    assert "Shot 0: Maria enters" in block
    assert 'Dialogue: "hi"' in block
    assert 'AD: "Maria waves"' in block
    assert "Shot 1: an empty hall" in block


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

    # The eligible segment gets a narration line; the ineligible one does not.
    assert result.segments[0].ad_narration == "A quiet moment unfolds on screen."
    assert result.segments[1].ad_narration is None


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
    tmp_path, fake_gemini_client
):
    keyframe = tmp_path / "frame.jpg"
    keyframe.write_bytes(b"fake-image-bytes")

    # Canned narration is 6 words -> ~2.4s at 2.5 wps, over the 2.0s gap, so the
    # inline optimization pass should fire.
    seg = _segment(
        0, None, ad_eligible=True, narratable_gap_sec=2.0, keyframe=str(keyframe)
    )
    tl = Timeline(video_id="v", duration_sec=2.0, segments=[seg])

    result = await fill_narration_gaps(tl, client=fake_gemini_client)

    assert len(_inline_optimize_calls(fake_gemini_client)) == 1
    # The optimized text replaces the generated line.
    assert result.segments[0].ad_narration == "A quiet moment unfolds on screen."


async def test_fill_narration_gaps_skips_optimization_when_line_fits(
    tmp_path, fake_gemini_client
):
    keyframe = tmp_path / "frame.jpg"
    keyframe.write_bytes(b"fake-image-bytes")

    # A generous 10s gap easily fits the ~2.4s canned line -> no optimization.
    seg = _segment(
        0, None, ad_eligible=True, narratable_gap_sec=10.0, keyframe=str(keyframe)
    )
    tl = Timeline(video_id="v", duration_sec=10.0, segments=[seg])

    result = await fill_narration_gaps(tl, client=fake_gemini_client)

    assert _inline_optimize_calls(fake_gemini_client) == []
    assert result.segments[0].ad_narration == "A quiet moment unfolds on screen."


async def test_fill_narration_gaps_feeds_prior_scenes_as_continuity_context(
    tmp_path, fake_gemini_client
):
    keyframe = tmp_path / "frame.jpg"
    keyframe.write_bytes(b"fake-image-bytes")

    # Shot 0 has dialogue but isn't narrated; shot 1 is. Shot 1's narration prompt
    # should still carry shot 0's description + dialogue as continuity context.
    shot0 = _segment(
        0,
        "hello there",
        ad_eligible=False,
        narratable_gap_sec=0.0,
        keyframe=str(keyframe),
    )
    shot1 = _segment(
        1, None, ad_eligible=True, narratable_gap_sec=10.0, keyframe=str(keyframe)
    )
    tl = Timeline(video_id="v", duration_sec=2.0, segments=[shot0, shot1])

    await fill_narration_gaps(tl, client=fake_gemini_client)

    # The single generation call (shot 1) is the only one carrying the narration
    # prompt; it must include shot 0 as an earlier scene, with its dialogue.
    gen_prompt = _prompt_of(fake_gemini_client.calls[-1])
    assert "EARLIER SCENES" in gen_prompt
    assert "Shot 0:" in gen_prompt
    assert 'Dialogue: "hello there"' in gen_prompt
