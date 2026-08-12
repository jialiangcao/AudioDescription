from conftest import FakeFunctionCall, FakeGeminiResponse

from qa.core_agent import CoreAgent
from qa.llm import ToolContext
from qa.localize_agent import LocalizeAgent
from qa.perception_agent import PerceptionAgent
from qa.prompts import FORCE_ANSWER_PROMPT
from qa.reflection_agent import ReflectionAgent
from qa.subtitle_agent import NO_SUBTITLES, SubtitleAgent
from timeline import AudioAnalysis, Frame, Segment, Timeline

QUESTION = "What color is the car?"


def _content_text(content) -> str:
    """Extract the text of a types.Content built by qa.llm.user_content."""
    return content.parts[0].text


# --------------------------------------------------------------------------- #
# CoreAgent
# --------------------------------------------------------------------------- #


async def test_core_prompt_and_decision(fake_gemini_client):
    agent = CoreAgent(fake_gemini_client, question=QUESTION, video_duration_sec=100.0)
    decision = await agent.run(history=[{"action": "SubtitleAgent", "result": "r"}])

    # Canned CoreDecision output -> a finish decision dict.
    assert decision == {
        "reason": "enough information",
        "agent": "finish",
        "answer": "a canned answer",
    }

    (_, prompt, config) = fake_gemini_client.calls[0]
    assert QUESTION in prompt
    assert 'Video duration: "00:01:40"' in prompt
    assert "<history>" in prompt
    assert '{"action": "SubtitleAgent", "result": "r"}' in prompt
    assert "must** be double-checked by using the PerceptionAgent" in prompt
    assert "relative score >= 3" in prompt
    assert "(A B C D)" not in prompt  # free-form answers, no MCQ leftovers
    assert "Critically" not in config.system_instruction
    assert "THINK" in config.system_instruction


async def test_core_prompt_with_empty_history(fake_gemini_client):
    agent = CoreAgent(fake_gemini_client, question=QUESTION, video_duration_sec=10.0)
    await agent.run(history=[])
    (_, prompt, _) = fake_gemini_client.calls[0]
    assert "No actions have been taken yet." in prompt


async def test_core_falls_back_to_json_repair(fake_gemini_client):
    fake_gemini_client.queue(
        FakeGeminiResponse("not valid json at all"),
        FakeGeminiResponse('{"reason": "r", "agent": "SubtitleAgent"}'),
    )
    agent = CoreAgent(fake_gemini_client, question=QUESTION, video_duration_sec=10.0)
    decision = await agent.run(history=[])
    assert decision == {"reason": "r", "agent": "SubtitleAgent"}
    assert len(fake_gemini_client.calls) == 2  # planner call + repair call


# --------------------------------------------------------------------------- #
# LocalizeAgent
# --------------------------------------------------------------------------- #


def _localize_agent(client, fake_frame_index, fake_blobs):
    return LocalizeAgent(
        client,
        question=QUESTION,
        video_duration_sec=100.0,
        ctx=ToolContext(client=client, frame_index=fake_frame_index, blobs=fake_blobs),
    )


async def test_localize_finish_returns_answer_string(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    fake_gemini_client.queue(
        FakeGeminiResponse(
            None,
            function_calls=[
                FakeFunctionCall("finish", {"answer": "00:01:00-00:02:00"}),
                FakeFunctionCall("retrieve_tool", {"cue": "ignored"}),
            ],
        )
    )
    agent = _localize_agent(fake_gemini_client, fake_frame_index, fake_blobs)
    result = await agent.run()

    assert result == "00:01:00-00:02:00"
    # Only the FIRST tool call executes; the retrieve call never happens.
    assert len(fake_gemini_client.calls) == 1

    prompt = _content_text(fake_gemini_client.calls[0][1][0])
    assert QUESTION in prompt
    assert "00:01:40" in prompt  # VIDEO_LENGTH replaced
    assert "localize_instruction" not in prompt  # phantom param removed


async def test_localize_dispatches_retrieve_tool(
    fake_gemini_client, fake_frame_index, fake_blobs, stub_retriever
):
    fake_gemini_client.queue(
        FakeGeminiResponse(
            None, function_calls=[FakeFunctionCall("retrieve_tool", {"cue": "a car"})]
        )
    )
    agent = _localize_agent(fake_gemini_client, fake_frame_index, fake_blobs)
    result = await agent.run()
    assert result.startswith("The most similar time point:")
    assert stub_retriever[0][1] == "a car"


async def test_localize_retries_until_tool_call_then_gives_up(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    for _ in range(6):
        fake_gemini_client.queue(FakeGeminiResponse("no tools chosen"))
    agent = _localize_agent(fake_gemini_client, fake_frame_index, fake_blobs)
    result = await agent.run()
    assert result == "no tools chosen"
    assert len(fake_gemini_client.calls) == 6


async def test_localize_unknown_tool_name(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    fake_gemini_client.queue(
        FakeGeminiResponse(None, function_calls=[FakeFunctionCall("bogus")])
    )
    agent = _localize_agent(fake_gemini_client, fake_frame_index, fake_blobs)
    assert await agent.run() == "Invalid function name: 'bogus'"


# --------------------------------------------------------------------------- #
# PerceptionAgent
# --------------------------------------------------------------------------- #


def _perception_agent(client, fake_frame_index, fake_blobs, max_iterations=6):
    return PerceptionAgent(
        client,
        ctx=ToolContext(client=client, frame_index=fake_frame_index, blobs=fake_blobs),
        max_iterations=max_iterations,
    )


async def test_perception_tool_then_answer(
    fake_gemini_client, fake_frame_index, fake_blobs, stub_retriever
):
    fake_gemini_client.queue(
        # Turn 1: the model requests one inspection (content text is None).
        FakeGeminiResponse(
            None,
            function_calls=[
                FakeFunctionCall(
                    "frame_inspect_tool",
                    {
                        "question": "what color?",
                        "time_range": ["00:00:10", "00:00:20"],
                        "cue": "car",
                    },
                )
            ],
        ),
        # The tool's own vision call.
        FakeGeminiResponse("the car is red"),
        # Turn 2: the model answers.
        FakeGeminiResponse("[answer] The car is red."),
    )
    agent = _perception_agent(fake_gemini_client, fake_frame_index, fake_blobs)
    result = await agent.run(
        instruct="check [00:00:10, 00:00:20] for (car color)",
        question=QUESTION,
        video_duration=100.0,
    )

    assert result == "[answer] The car is red."
    assert len(fake_gemini_client.calls) == 3
    # The instruct is embedded in the first turn's prompt.
    first_prompt = _content_text(fake_gemini_client.calls[0][1][0])
    assert "check [00:00:10, 00:00:20] for (car color)" in first_prompt
    # Turn 2 sees the model turn + the tool response appended.
    turn2_contents = fake_gemini_client.calls[2][1]
    assert len(turn2_contents) == 3
    assert turn2_contents[2].role == "tool"
    assert "the car is red" in str(
        turn2_contents[2].parts[0].function_response.response
    )


async def test_perception_force_answer_on_last_iteration(
    fake_gemini_client, fake_frame_index, fake_blobs, stub_retriever
):
    inspect_call = FakeFunctionCall(
        "frame_inspect_tool",
        {"question": "q", "time_range": ["00:00:10", "00:00:20"], "cue": "c"},
    )
    fake_gemini_client.queue(
        FakeGeminiResponse(None, function_calls=[inspect_call]),
        FakeGeminiResponse("a scene"),  # tool vision call
        FakeGeminiResponse("still looking", function_calls=[inspect_call]),
        FakeGeminiResponse("a scene"),  # tool vision call
    )
    agent = _perception_agent(
        fake_gemini_client, fake_frame_index, fake_blobs, max_iterations=2
    )
    result = await agent.run(instruct="i", question=QUESTION, video_duration=100.0)

    # The forced-answer turn still tool-called; best-effort text comes back.
    assert result == "still looking"
    last_turn_contents = fake_gemini_client.calls[2][1]
    assert _content_text(last_turn_contents[-1]) == FORCE_ANSWER_PROMPT


async def test_perception_returns_text_when_no_tools_requested(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    for _ in range(6):
        fake_gemini_client.queue(FakeGeminiResponse("cannot determine"))
    agent = _perception_agent(fake_gemini_client, fake_frame_index, fake_blobs)
    result = await agent.run(instruct="i", question=QUESTION, video_duration=100.0)
    assert result == "cannot determine"
    assert len(fake_gemini_client.calls) == 6  # inner retry exhausted


# --------------------------------------------------------------------------- #
# SubtitleAgent
# --------------------------------------------------------------------------- #


def _timeline_with_speech():
    return Timeline(
        job_id="v.mp4",
        duration_sec=10.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=2.0,
                frames=[Frame(index=0, time=0.0, key="frames/f.jpg")],
                audio=AudioAnalysis(
                    has_speech=True, transcript="hello there", silence_ratio=0.2
                ),
            ),
            Segment(
                id=1,
                start=2.0,
                end=5.0,
                frames=[Frame(index=0, time=2.0, key="frames/g.jpg")],
                audio=AudioAnalysis(
                    has_speech=False, transcript=None, silence_ratio=1.0
                ),
            ),
            Segment(
                id=2,
                start=5.0,
                end=8.0,
                frames=[Frame(index=0, time=5.0, key="frames/h.jpg")],
                audio=AudioAnalysis(
                    has_speech=True, transcript="general kenobi", silence_ratio=0.1
                ),
            ),
        ],
    )


async def test_subtitle_formats_transcript(fake_gemini_client):
    agent = SubtitleAgent(
        fake_gemini_client, question=QUESTION, timeline=_timeline_with_speech()
    )
    result = await agent.run()

    assert result == "A quiet moment unfolds on screen."
    (_, prompt, _) = fake_gemini_client.calls[0]
    assert "00:00:00-00:00:02: hello there 00:00:05-00:00:08: general kenobi" in prompt
    assert QUESTION in prompt
    assert "relevant_subtitle_info" in prompt


async def test_subtitle_short_circuits_without_speech(fake_gemini_client):
    timeline = _timeline_with_speech()
    for seg in timeline.segments:
        seg.audio = None
    agent = SubtitleAgent(fake_gemini_client, question=QUESTION, timeline=timeline)
    assert await agent.run() == NO_SUBTITLES
    assert fake_gemini_client.calls == []  # no model call at all


# --------------------------------------------------------------------------- #
# ReflectionAgent
# --------------------------------------------------------------------------- #


async def test_reflection_parses_assessment(fake_gemini_client):
    fake_gemini_client.queue(
        FakeGeminiResponse('{"credible": false, "comment": "the count is wrong"}')
    )
    agent = ReflectionAgent(fake_gemini_client, question=QUESTION)
    assessment = await agent.run(
        proposed_answer="three", history=[{"action": "finish", "answer": "three"}]
    )
    assert assessment == {"credible": False, "comment": "the count is wrong"}

    (_, prompt, _) = fake_gemini_client.calls[0]
    assert QUESTION in prompt
    assert "Proposed Answer: three" in prompt
    assert '{"action": "finish", "answer": "three"}' in prompt


async def test_reflection_fails_open_on_garbage(fake_gemini_client):
    fake_gemini_client.queue(
        FakeGeminiResponse("garbage"),  # reflection output
        FakeGeminiResponse("still garbage"),  # repair attempt
    )
    agent = ReflectionAgent(fake_gemini_client, question=QUESTION)
    assessment = await agent.run(proposed_answer="x", history=[])
    assert assessment["credible"] is True
    assert assessment["comment"].startswith("Fallback")
