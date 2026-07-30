"""PerceptionAgent tools (plus retrieve_tool, which LocalizeAgent also binds).

Ports of Symphony's perception_tools. Frame selection works against the
job's FrameIndex (authoritative per-frame timestamps) instead of Symphony's
directory listing + 2fps index arithmetic; CLIP retrieval goes through
qa.retriever (open_clip in place of LanguageBind).

Every tool returns a string — an answer/description from the vision model, or
an explanatory error message (Symphony returned error strings too, so the
calling agent can self-correct).
"""

import logging

from google.genai import types

from qa import retriever
from qa.config import (
    ASSOCIATE_TOP_K_PER_CUE,
    INSPECT_RETRIEVE_TOP_K,
    INSPECT_UNIFORM_MAX,
    RETRIEVE_TOP_K,
    SUMMARY_FRAME_COUNT,
)
from qa.llm import ToolContext, generate_vision
from qa.prompts import (
    FRAME_ASSOCIATE_PROMPT,
    FRAME_INSPECT_PROMPT,
    INTERVAL_SUMMARY_PROMPT,
    VISION_TOOL_SYSTEM_PROMPT,
)
from qa.utils import (
    convert_hhmmss_to_seconds,
    convert_seconds_to_hhmmss,
    with_retries,
)

logger = logging.getLogger(__name__)

_TIME_RANGE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(type=types.Type.STRING),
    description=(
        "Start and end time in HH:MM:SS format — exactly two items, "
        'e.g. ["00:01:00", "00:01:45"].'
    ),
)

RETRIEVE_TOOL_DECLARATION = types.FunctionDeclaration(
    name="retrieve_tool",
    description=(
        "Retrieve the most relevant time points in the video for a short "
        "textual cue (CLIP similarity over all frames)."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "cue": types.Schema(
                type=types.Type.STRING,
                description=(
                    "The specific description of scene/objects/people "
                    "contained in the question, used to retrieve relevant "
                    "frames."
                ),
            ),
        },
        required=["cue"],
    ),
)

FRAME_INSPECT_TOOL_DECLARATION = types.FunctionDeclaration(
    name="frame_inspect_tool",
    description=(
        "Inspect the frames within a 5-60s time range in detail and answer "
        "the question from them."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "question": types.Schema(
                type=types.Type.STRING,
                description=(
                    "The specific question about the video content during the "
                    "specified time range. Do not add time ranges and "
                    "subtitles in the question."
                ),
            ),
            "time_range": _TIME_RANGE_SCHEMA,
            "cue": types.Schema(
                type=types.Type.STRING,
                description=(
                    "The specific objects contained in the question, used to "
                    "retrieve relevant frames."
                ),
            ),
        },
        required=["question", "time_range", "cue"],
    ),
)

INTERVAL_SUMMARY_TOOL_DECLARATION = types.FunctionDeclaration(
    name="interval_summary_tool",
    description=(
        "Get a rough overview of a long time range (the entire video, or "
        "more than 3 minutes) from uniformly sampled frames."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "question": types.Schema(
                type=types.Type.STRING,
                description="A question about an overview of the video.",
            ),
            "time_range": _TIME_RANGE_SCHEMA,
        },
        required=["question", "time_range"],
    ),
)

FRAME_ASSOCIATE_TOOL_DECLARATION = types.FunctionDeclaration(
    name="frame_associate_tool",
    description=(
        "Answer a question involving multiple scenes, or the sequence of "
        "scenes, by retrieving frames for each scene description."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "question": types.Schema(
                type=types.Type.STRING,
                description="The specific question containing multiple scenes.",
            ),
            "cue": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description=(
                    "The list of descriptions, each element is a description "
                    "of one scene/object/people contained in the question, "
                    "used to retrieve relevant frames."
                ),
            ),
        },
        required=["question", "cue"],
    ),
)


def _parse_time_range(time_range, duration_sec: float) -> tuple[float, float]:
    """Validate a model-supplied [start, end] pair; raises ValueError."""
    if not isinstance(time_range, list | tuple) or len(time_range) != 2:
        raise ValueError(
            f"time_range must be [start, end] in HH:MM:SS format, got {time_range!r}"
        )
    start = convert_hhmmss_to_seconds(time_range[0])
    end = convert_hhmmss_to_seconds(time_range[1])
    end = min(end, duration_sec)
    if end <= start:
        raise ValueError(f"empty time range: {time_range!r}")
    return start, end


async def retrieve_tool(cue: str, *, ctx: ToolContext) -> str:
    results = await retriever.retrieve_top_k(
        ctx.frame_index.paths(), cue, RETRIEVE_TOP_K
    )
    if not results:
        return "Error: no frames available for this video."
    # Similarity order, not chronological — as in Symphony.
    frame_seconds = [
        convert_seconds_to_hhmmss(ctx.frame_index.timestamp_of(path))
        for path, _ in results
    ]
    logger.debug("retrieve_tool: cue=%r -> %s", cue, frame_seconds)
    return "The most similar time point:" + str(frame_seconds)


async def frame_inspect_tool(
    question: str, time_range, cue: str, *, ctx: ToolContext
) -> str:
    try:
        start, end = _parse_time_range(time_range, ctx.frame_index.duration_sec)
    except ValueError as exc:
        return f"Error: {exc}"

    candidates = ctx.frame_index.in_range(start, end)
    if not candidates:
        return f"Error: no frames available in the time range {time_range!r}."

    # Union of a uniform spread and the CLIP hits for the cue, deduped and in
    # temporal order — Symphony's selection. At this pipeline's 0.5fps sampling
    # the uniform part covers every frame of a <60s range.
    uniform = ctx.frame_index.uniform_sample(
        start, end, min(INSPECT_UNIFORM_MAX, int(end - start))
    )
    retrieved = await retriever.retrieve_top_k(
        [path for _, path in candidates], cue, INSPECT_RETRIEVE_TOP_K
    )
    selected = set(uniform) | {path for path, _ in retrieved}
    frame_paths = sorted(selected, key=ctx.frame_index.timestamp_of)
    logger.debug(
        "frame_inspect_tool: %d frame(s) in [%s, %s]",
        len(frame_paths),
        time_range[0],
        time_range[1],
    )

    return await with_retries(
        lambda: generate_vision(
            ctx.client,
            system=VISION_TOOL_SYSTEM_PROMPT,
            user=FRAME_INSPECT_PROMPT.format(question=question),
            frame_paths=frame_paths,
        )
    )


async def interval_summary_tool(question: str, time_range, *, ctx: ToolContext) -> str:
    try:
        start, end = _parse_time_range(time_range, ctx.frame_index.duration_sec)
    except ValueError as exc:
        return f"Error: {exc}"

    frame_paths = ctx.frame_index.uniform_sample(start, end, SUMMARY_FRAME_COUNT)
    if not frame_paths:
        return f"Error: no frames available in the time range {time_range!r}."
    logger.debug(
        "interval_summary_tool: %d frame(s) in [%s, %s]",
        len(frame_paths),
        time_range[0],
        time_range[1],
    )

    return await with_retries(
        lambda: generate_vision(
            ctx.client,
            system=VISION_TOOL_SYSTEM_PROMPT,
            user=INTERVAL_SUMMARY_PROMPT.format(question=question),
            frame_paths=frame_paths,
        )
    )


async def frame_associate_tool(
    question: str, cue: list[str], *, ctx: ToolContext
) -> str:
    if not cue:
        return "Error: cue must be a non-empty list of scene descriptions."
    all_paths = ctx.frame_index.paths()
    selected: set[str] = set()
    for cue_item in cue:
        results = await retriever.retrieve_top_k(
            all_paths, cue_item, ASSOCIATE_TOP_K_PER_CUE
        )
        selected.update(path for path, _ in results)
    if not selected:
        return "Error: no frames available for this video."
    frame_paths = sorted(selected, key=ctx.frame_index.timestamp_of)
    logger.debug(
        "frame_associate_tool: %d frame(s) across %d cue(s)",
        len(frame_paths),
        len(cue),
    )

    return await with_retries(
        lambda: generate_vision(
            ctx.client,
            system=VISION_TOOL_SYSTEM_PROMPT,
            user=FRAME_ASSOCIATE_PROMPT.format(question=question),
            frame_paths=frame_paths,
        )
    )
