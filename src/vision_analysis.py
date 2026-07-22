import asyncio
import logging
from collections.abc import Awaitable, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from prompts import (
    INLINE_OPTIMIZATION_PROMPT,
    NARRATION_GUIDELINES,
    NARRATION_HEADER,
    NARRATION_NO_PRIOR,
    NARRATION_PRIOR_HEADER,
    NO_CONTEXT,
    RETRY_OPTIMIZATION_PROMPT,
    SCENE_GENERATION_PROMPT,
    SYSTEM_INSTRUCTION,
)
from timeline import NARRATION_WORDS_PER_SEC, Segment, Timeline

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash"

# Shared Gemini generation settings for every call in this module. Reasoning
# tokens share MAX_OUTPUT_TOKENS with the visible output, so keep it generous.
# THINKING_LEVEL tunes reasoning effort; HIGH costs more latency/tokens and, for
# these direct perceptual/description tasks, tends to over-condense — LOW or
# MINIMAL usually gives richer, more concrete descriptions.
MAX_OUTPUT_TOKENS = 1024
THINKING_LEVEL = types.ThinkingLevel.LOW

# How many Gemini calls to keep in flight at once within a job's shot-analysis
# loop. Narration generation, by contrast, runs sequentially (each line is fed
# the previous scenes' descriptions/dialogue/narration for continuity).
DEFAULT_CONCURRENCY = 4

# How many preceding scenes to feed a narration line as continuity context.
NARRATION_CONTEXT_SCENES = 6


class ShotAnalysis(BaseModel):
    description: str
    entities: list[str]
    setting: str
    on_screen_text: str | None


async def analyze_keyframe(
    client, keyframe_path, scene_duration=0.0, context=NO_CONTEXT
):
    logger.debug("analyze_keyframe: %s -> %s", keyframe_path, MODEL)
    with open(keyframe_path, "rb") as f:
        image_bytes = f.read()

    prompt = SCENE_GENERATION_PROMPT.format(
        scene_duration=scene_duration, context=context
    )
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            # Reasoning shares the output budget, so keep it generous enough for
            # high-effort thinking plus the structured JSON result.
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=ShotAnalysis,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
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
        scene_duration = shot.get("end", 0.0) - shot.get("start", 0.0)
        async with semaphore:
            shot["visual"] = await analyze_keyframe(
                client, shot["keyframe"], scene_duration=scene_duration
            )
        if on_shot is not None:
            await on_shot(shot)
        return shot

    await asyncio.gather(*(_run(shot) for shot in shots))
    logger.info("analyze_shots: all %d shot(s) analyzed", len(shots))
    return shots


def _current_shot_block(segment):
    """The CURRENT SHOT block: what's on screen plus any dialogue in this shot."""
    lines = [segment.visual.description]
    if segment.visual.on_screen_text:
        lines.append(f'On-screen text: "{segment.visual.on_screen_text}"')
    if segment.audio and segment.audio.transcript:
        lines.append(
            f'Dialogue in this shot (do not repeat): "{segment.audio.transcript}"'
        )
    return "\n".join(lines)


def _prior_scenes_block(prior_scenes):
    """The earlier-scenes continuity block from the accumulated scene history.

    ``prior_scenes`` is a list of ``{id, description, dialogue, narration}`` dicts
    (oldest first). Each preceding scene contributes its visual description, its
    dialogue, and the AD line already written for it, so the model can reuse
    established names, avoid repeating visuals, and keep descriptions continuous.
    """
    if not prior_scenes:
        return NARRATION_NO_PRIOR

    lines = [NARRATION_PRIOR_HEADER]
    for scene in prior_scenes:
        parts = [f"- Shot {scene['id']}: {scene['description']}"]
        if scene.get("dialogue"):
            parts.append(f'Dialogue: "{scene["dialogue"]}"')
        if scene.get("narration"):
            parts.append(f'AD: "{scene["narration"]}"')
        lines.append(" ".join(parts))
    return "\n".join(lines)


async def generate_narration(client, segment, max_words, prior_scenes=None):
    """Write one AD narration line for ``segment``, aware of the preceding scenes.

    Builds the narration prompt from the current shot plus the accumulated
    earlier-scene context (``prior_scenes``) so the line reuses established names,
    doesn't repeat prior visuals, and stays continuous. Returns the raw line; the
    caller may further condense it via ``optimize_narration`` to fit the gap.
    """
    prompt = (
        NARRATION_HEADER.format(gap_sec=segment.narratable_gap_sec, max_words=max_words)
        + "\n\nCURRENT SHOT:\n"
        + _current_shot_block(segment)
        + "\n\n"
        + _prior_scenes_block(prior_scenes)
        + "\n\n"
        + NARRATION_GUIDELINES
    )

    with open(segment.keyframe, "rb") as f:
        image_bytes = f.read()

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            # Reasoning shares the output budget, so keep it generous enough for
            # high-effort thinking plus the narration line.
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
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


async def optimize_narration(client, combined_text, available_duration):
    """Condense ``combined_text`` so it can be spoken within ``available_duration``.

    Uses the paper's inline optimization prompt: when a generated line's estimated
    speech duration exceeds the silence gap it must fit, this rewrites it shorter
    while keeping the action order, rather than relying on TTS speed-up alone.
    """
    prompt = INLINE_OPTIMIZATION_PROMPT.format(
        combined_text=combined_text, available_duration=available_duration
    )
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        ),
    )

    if response.text is None:
        logger.error("optimize_narration: %s returned no text", MODEL)
        raise RuntimeError(f"{MODEL} returned no text optimizing narration")
    return response.text.strip()


async def retry_optimize_narration(
    client, optimized_text, tts_duration, available_duration
):
    """Shorten a line that still overran its gap once synthesized (measured via TTS).

    Uses the paper's retry optimization prompt: unlike the inline pass (which
    works from an *estimated* duration before TTS), this reports the *measured*
    speech duration and the exact overshoot so the model can cut the right amount.
    """
    prompt = RETRY_OPTIMIZATION_PROMPT.format(
        optimized_text=optimized_text,
        tts_duration=tts_duration,
        available_duration=available_duration,
        reduce_by=max(0.0, tts_duration - available_duration),
    )
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        ),
    )

    if response.text is None:
        logger.error("retry_optimize_narration: %s returned no text", MODEL)
        raise RuntimeError(f"{MODEL} returned no text on retry optimization")
    return response.text.strip()


def _estimated_speech_sec(text, words_per_sec):
    return len(text.split()) / words_per_sec if words_per_sec else 0.0


async def fill_narration_gaps(
    timeline: Timeline,
    on_segment: Callable[[Segment], Awaitable[None]] | None = None,
    words_per_sec: float | None = None,
    client=None,
):
    """Generate AD narration for every ``ad_eligible`` segment, in scene order.

    Unlike the shot-analysis stage, this runs sequentially: each narration line
    is written with the preceding scenes' descriptions, dialogue, and narration
    as context (a rolling window of ``NARRATION_CONTEXT_SCENES`` shots), so the
    model reuses established names, avoids repeating earlier visuals, and keeps
    the audio description continuous. Every shot — narrated or not — contributes
    to that history. ``on_segment`` (if given) is awaited per narrated segment,
    in id order.
    """
    words_per_sec = words_per_sec or NARRATION_WORDS_PER_SEC
    client = client or genai.Client()
    eligible = [seg for seg in timeline.segments if seg.ad_eligible]
    logger.info("fill_narration_gaps: %d eligible segment(s)", len(eligible))

    history: list[dict] = []
    for segment in timeline.segments:
        dialogue = segment.audio.transcript if segment.audio else None

        if segment.ad_eligible and segment.narratable_gap_sec is not None:
            gap = segment.narratable_gap_sec
            max_words = max(3, int(gap * words_per_sec))
            prior = history[-NARRATION_CONTEXT_SCENES:]
            logger.debug(
                "fill_narration_gaps: segment %s, gap=%.2fs, %d prior scene(s)",
                segment.id,
                gap,
                len(prior),
            )
            text = await generate_narration(client, segment, max_words, prior)
            # If the line's estimated spoken duration overruns the gap, condense
            # it with the inline optimization prompt so it fits the timing.
            if _estimated_speech_sec(text, words_per_sec) > gap:
                logger.debug(
                    "fill_narration_gaps: segment %s narration overruns %.2fs gap, "
                    "optimizing",
                    segment.id,
                    gap,
                )
                text = await optimize_narration(client, text, gap)
            segment.ad_narration = text
            if on_segment is not None:
                await on_segment(segment)

        history.append(
            {
                "id": segment.id,
                "description": segment.visual.description,
                "dialogue": dialogue,
                "narration": segment.ad_narration,
            }
        )

    logger.info("fill_narration_gaps: wrote %d narration line(s)", len(eligible))
    return timeline
