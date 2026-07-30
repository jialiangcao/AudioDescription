"""localize_tool: exhaustive VLM relevance scoring over the whole video.

Port of Symphony's localize_tool: the video is cut into fixed 30s windows and
every window gets one vision call scoring its relevance to the question (1-4).
Windows scoring above the floor come back as
``"The relevance segment:[{'time': ..., 'judgement': {...}}, ...]"``.
Symphony fanned out on a thread pool; here the calls are async under a
semaphore. Per-window failures are logged and skipped, as in Symphony.
"""

import asyncio
import logging

from google.genai import types
from pydantic import BaseModel

from qa.config import LOCALIZE_CONCURRENCY, LOCALIZE_MIN_SCORE, LOCALIZE_WINDOW_SEC
from qa.llm import ToolContext, generate_vision
from qa.prompts import JUDGEMENT_PROMPT, JUDGEMENT_SYSTEM_PROMPT
from qa.utils import convert_seconds_to_hhmmss, with_retries

logger = logging.getLogger(__name__)


class Judgement(BaseModel):
    relevance_score: int
    clip_caption: str
    reasoning: str | None = None


LOCALIZE_TOOL_DECLARATION = types.FunctionDeclaration(
    name="localize_tool",
    description=(
        "Score every 30-second window of the video for relevance to the "
        "question. Use for complex questions requiring scenario understanding."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "question": types.Schema(
                type=types.Type.STRING,
                description="The question to be answered.",
            ),
        },
        required=["question"],
    ),
)


async def _judge_window(
    ctx: ToolContext,
    semaphore: asyncio.Semaphore,
    window_start: float,
    frame_paths: list[str],
    question: str,
) -> dict | None:
    """Score one window; None on failure (the window is then skipped)."""
    time_range = (
        f"{convert_seconds_to_hhmmss(window_start)}"
        f"-{convert_seconds_to_hhmmss(window_start + LOCALIZE_WINDOW_SEC - 1)}"
    )
    prompt = JUDGEMENT_PROMPT.replace("{USER_QUESTION}", question)
    async with semaphore:
        try:
            text = await with_retries(
                lambda: generate_vision(
                    ctx.client,
                    system=JUDGEMENT_SYSTEM_PROMPT,
                    user=prompt,
                    frame_paths=frame_paths,
                    schema=Judgement,
                )
            )
            judgement = Judgement.model_validate_json(text)
        except Exception:
            logger.warning("localize_tool: window %s failed", time_range, exc_info=True)
            return None
    logger.debug(
        "localize_tool: window %s score=%d", time_range, judgement.relevance_score
    )
    return {"time": time_range, "judgement": judgement.model_dump()}


async def localize_tool(question: str, *, ctx: ToolContext) -> str:
    windows = ctx.frame_index.windows(LOCALIZE_WINDOW_SEC)
    if not windows:
        return "Error: no frames available for this video."
    logger.info(
        "localize_tool: scoring %d window(s) of %.0fs for %r",
        len(windows),
        LOCALIZE_WINDOW_SEC,
        question,
    )

    semaphore = asyncio.Semaphore(LOCALIZE_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _judge_window(ctx, semaphore, window_start, frame_paths, question)
            for window_start, _, frame_paths in windows
        )
    )

    relevant = [
        result
        for result in results
        if result and result["judgement"]["relevance_score"] >= LOCALIZE_MIN_SCORE
    ]
    logger.info("localize_tool: %d/%d window(s) relevant", len(relevant), len(windows))
    return "The relevance segment:" + str(relevant)
