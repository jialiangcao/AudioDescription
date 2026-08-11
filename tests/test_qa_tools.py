import ast

from conftest import FakeGeminiResponse

from qa.llm import ToolContext
from qa.tools_localize import localize_tool
from qa.tools_perception import (
    frame_associate_tool,
    frame_inspect_tool,
    interval_summary_tool,
    retrieve_tool,
)


def _judgement(score, caption="a scene"):
    return FakeGeminiResponse(
        f'{{"relevance_score": {score}, "clip_caption": "{caption}", '
        f'"reasoning": "because"}}'
    )


def _frame_count(call):
    """Number of image parts sent in a recorded fake-Gemini call."""
    _, contents, _ = call
    return len(contents) - 1  # last element is the prompt string


# --------------------------------------------------------------------------- #
# localize_tool
# --------------------------------------------------------------------------- #


async def test_localize_scores_every_window(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    result = await localize_tool("what happens?", ctx=ctx)

    # 100s at one frame/2s -> windows starting 0/30/60/90 with 15/15/15/5 frames.
    # Windows are scored concurrently, so assert on the multiset of batch sizes
    # rather than their completion order.
    assert len(fake_gemini_client.calls) == 4
    assert sorted(_frame_count(c) for c in fake_gemini_client.calls) == [5, 15, 15, 15]
    # The canned Judgement scores 1, so every window is filtered out.
    assert result == "The relevance segment:[]"
    # The question lands in every window's prompt.
    assert all("what happens?" in c[1][-1] for c in fake_gemini_client.calls)


async def test_localize_keeps_scores_above_floor(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    fake_gemini_client.queue(_judgement(1), _judgement(2), _judgement(3), _judgement(4))
    result = await localize_tool("q", ctx=ctx)

    # Scripted responses are handed out in call order, and windows are scored
    # concurrently, so which window drew score 1 is not fixed. What matters is
    # that exactly the sub-floor one was dropped and the rest survived.
    assert result.startswith("The relevance segment:")
    kept = ast.literal_eval(result.removeprefix("The relevance segment:"))
    assert sorted(entry["judgement"]["relevance_score"] for entry in kept) == [2, 3, 4]
    assert len({entry["time"] for entry in kept}) == 3
    assert all(
        entry["time"]
        in {
            "00:00:00-00:00:29",
            "00:00:30-00:00:59",
            "00:01:00-00:01:29",
            "00:01:30-00:01:59",
        }
        for entry in kept
    )


async def test_localize_with_no_frames(fake_gemini_client, fake_blobs):
    from qa.frame_index import FrameIndex

    ctx = ToolContext(
        client=fake_gemini_client,
        frame_index=FrameIndex(entries=[], duration_sec=10.0),
        blobs=fake_blobs,
    )
    result = await localize_tool("q", ctx=ctx)
    assert result.startswith("Error:")
    assert fake_gemini_client.calls == []


# --------------------------------------------------------------------------- #
# retrieve_tool
# --------------------------------------------------------------------------- #


async def test_retrieve_returns_timestamps_in_similarity_order(
    fake_gemini_client, fake_frame_index, fake_blobs, stub_retriever
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    result = await retrieve_tool("a red car", ctx=ctx)

    assert result.startswith("The most similar time point:")
    # Stub ranks input order: first 15 frames -> t=0,2,...,28.
    assert "'00:00:00', '00:00:02'" in result
    (paths, cue, top_k) = stub_retriever[0]
    assert cue == "a red car"
    assert top_k == 15
    assert len(paths) == 50
    assert fake_gemini_client.calls == []  # text-only tool, no VLM call


# --------------------------------------------------------------------------- #
# frame_inspect_tool
# --------------------------------------------------------------------------- #


async def test_inspect_unions_uniform_and_retrieved_chronologically(
    fake_gemini_client, fake_frame_index, fake_blobs, stub_retriever
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    answer = await frame_inspect_tool(
        "what color?", ["00:00:10", "00:00:30"], "a car", ctx=ctx
    )

    assert answer == "A quiet moment unfolds on screen."
    # Retrieval ran over exactly the in-range frames (t=10..30 inclusive = 11).
    paths, _, top_k = stub_retriever[0]
    assert len(paths) == 11
    assert top_k == 20
    # All 11 frames sent (uniform covers the range; union dedups), in order.
    (call,) = fake_gemini_client.calls
    assert _frame_count(call) == 11
    assert "what color?" in call[1][-1]


async def test_inspect_clamps_end_to_duration(
    fake_gemini_client, fake_frame_index, fake_blobs, stub_retriever
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    await frame_inspect_tool("q", ["00:01:30", "01:00:00"], "c", ctx=ctx)
    paths, _, _ = stub_retriever[0]
    # 90..100s clamped -> frames at t=90..98.
    assert len(paths) == 5


async def test_inspect_rejects_bad_ranges(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    assert (await frame_inspect_tool("q", ["00:00:10"], "c", ctx=ctx)).startswith(
        "Error:"
    )
    assert (
        await frame_inspect_tool("q", ["00:00:30", "00:00:10"], "c", ctx=ctx)
    ).startswith("Error:")
    assert (
        await frame_inspect_tool("q", ["09:00:00", "09:00:30"], "c", ctx=ctx)
    ).startswith("Error:")
    assert fake_gemini_client.calls == []


# --------------------------------------------------------------------------- #
# interval_summary_tool
# --------------------------------------------------------------------------- #


async def test_summary_samples_at_most_30_frames(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    answer = await interval_summary_tool("overview?", ["00:00:00", "00:01:40"], ctx=ctx)

    assert answer == "A quiet moment unfolds on screen."
    (call,) = fake_gemini_client.calls
    assert 0 < _frame_count(call) <= 30
    assert "overview?" in call[1][-1]


# --------------------------------------------------------------------------- #
# frame_associate_tool
# --------------------------------------------------------------------------- #


async def test_associate_unions_per_cue_hits(
    fake_gemini_client, fake_frame_index, fake_blobs, stub_retriever
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    answer = await frame_associate_tool("which first?", ["a dog", "a cat"], ctx=ctx)

    assert answer == "A quiet moment unfolds on screen."
    assert [cue for _, cue, _ in stub_retriever] == ["a dog", "a cat"]
    assert all(top_k == 10 for _, _, top_k in stub_retriever)
    # Both cues rank the same stub top-10, so the union stays at 10 frames.
    (call,) = fake_gemini_client.calls
    assert _frame_count(call) == 10


async def test_associate_rejects_empty_cues(
    fake_gemini_client, fake_frame_index, fake_blobs
):
    ctx = ToolContext(
        client=fake_gemini_client, frame_index=fake_frame_index, blobs=fake_blobs
    )
    assert (await frame_associate_tool("q", [], ctx=ctx)).startswith("Error:")
