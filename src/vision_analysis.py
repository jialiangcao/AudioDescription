import asyncio
import logging
from collections.abc import Awaitable, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from timeline import NARRATION_WORDS_PER_SEC, Segment, Timeline

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash"

# How many Gemini calls to keep in flight at once within a single job's
# shot-analysis / narration loop.
DEFAULT_CONCURRENCY = 4


class ShotAnalysis(BaseModel):
    description: str
    entities: list[str]
    setting: str
    on_screen_text: str | None


async def analyze_keyframe(client, keyframe_path):
    logger.debug("analyze_keyframe: %s -> %s", keyframe_path, MODEL)
    with open(keyframe_path, "rb") as f:
        image_bytes = f.read()

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            "Describe what is visually happening in this video frame.",
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=ShotAnalysis,
            # Reasoning tokens count against max_output_tokens and can starve the
            # structured JSON output; this is a direct extraction task, so skip it.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    if response.text is None:
        logger.error(
            "analyze_keyframe: %s returned no text for %s", MODEL, keyframe_path
        )
        raise RuntimeError(f"{MODEL} returned no text for {keyframe_path}")
    return ShotAnalysis.model_validate_json(response.text).model_dump()


async def analyze_shots(
    shots,
    on_shot: Callable[[dict], Awaitable[None]] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    client=None,
):
    """Attach a ``visual`` analysis to every shot via Gemini, concurrently.

    Shots are analyzed with up to ``concurrency`` Gemini calls in flight at
    once; ``on_shot`` (if given) is awaited with each shot as its analysis
    completes, so callers can stream results. Completion order is not shot
    order, so consumers must key off ``shot["id"]``.
    """
    client = client or genai.Client()
    semaphore = asyncio.Semaphore(concurrency)
    logger.info("analyze_shots: %d shot(s), concurrency=%d", len(shots), concurrency)

    async def _run(shot):
        async with semaphore:
            shot["visual"] = await analyze_keyframe(client, shot["keyframe"])
        if on_shot is not None:
            await on_shot(shot)
        return shot

    await asyncio.gather(*(_run(shot) for shot in shots))
    logger.info("analyze_shots: all %d shot(s) analyzed", len(shots))
    return shots


def _neighbor_transcript(segments, index):
    for offset in (-1, 1):
        neighbor = index + offset
        if 0 <= neighbor < len(segments) and segments[neighbor].audio:
            text = segments[neighbor].audio.transcript
            if text:
                return text
    return None


async def generate_narration(client, segment, max_words, neighbor_transcript=None):
    context = f"Scene: {segment.visual.description}"
    if segment.visual.on_screen_text:
        context += f"\nOn-screen text: {segment.visual.on_screen_text}"
    if neighbor_transcript:
        context += f'\nNearby dialogue (for continuity, do not repeat): "{neighbor_transcript}"'

    with open(segment.keyframe, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "You are writing an audio description (AD) narration line for a blind/low-vision "
        "viewer, to be inserted into a speech-free gap in this video shot.\n\n"
        f"{context}\n\n"
        f"Write a single narration line of no more than {max_words} words that describes "
        "what's visually happening, without repeating information already implied by "
        "dialogue. Do not use phrases like 'we see' or 'the camera shows' — describe the "
        "action and setting directly. Return only the narration text, nothing else."
    )

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=2048,
            # gemini-2.5-flash "thinks" by default, and those reasoning tokens
            # count against max_output_tokens — leaving little to no budget for
            # the actual narration, which came out truncated after a few words.
            # Narration is a short, single-line task that needs no reasoning.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    if response.text is None:
        logger.error(
            "generate_narration: %s returned no text for segment keyframe %s",
            MODEL,
            segment.keyframe,
        )
        raise RuntimeError(f"{MODEL} returned no text for {segment.keyframe}")
    return response.text.strip()


async def fill_narration_gaps(
    timeline: Timeline,
    on_segment: Callable[[Segment], Awaitable[None]] | None = None,
    words_per_sec: float | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    client=None,
):
    """Generate AD narration for every ``ad_eligible`` segment, concurrently.

    ``on_segment`` (if given) is awaited with each segment as its narration
    completes. Completion order is not segment order, so consumers must key
    off ``segment.id``.
    """
    words_per_sec = words_per_sec or NARRATION_WORDS_PER_SEC
    client = client or genai.Client()
    semaphore = asyncio.Semaphore(concurrency)
    eligible = [seg for seg in timeline.segments if seg.ad_eligible]
    logger.info(
        "fill_narration_gaps: %d eligible segment(s), concurrency=%d",
        len(eligible),
        concurrency,
    )

    async def _run(i, segment):
        max_words = max(3, int(segment.narratable_gap_sec * words_per_sec))
        neighbor_transcript = _neighbor_transcript(timeline.segments, i)
        logger.debug(
            "fill_narration_gaps: segment %s, gap=%.2fs, max_words=%d",
            segment.id,
            segment.narratable_gap_sec,
            max_words,
        )
        async with semaphore:
            segment.ad_narration = await generate_narration(
                client, segment, max_words, neighbor_transcript
            )
        if on_segment is not None:
            await on_segment(segment)

    await asyncio.gather(
        *(
            _run(i, segment)
            for i, segment in enumerate(timeline.segments)
            if segment.ad_eligible
        )
    )

    logger.info("fill_narration_gaps: wrote %d narration line(s)", len(eligible))
    return timeline
