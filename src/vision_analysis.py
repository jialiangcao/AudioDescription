import asyncio
import logging
from collections.abc import Awaitable, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from prompts import (
    FRAME_ANALYSIS_PROMPT,
    INLINE_OPTIMIZATION_PROMPT,
    NARRATION_FRAMES_HEADER,
    NARRATION_GUIDELINES,
    NARRATION_HEADER,
    NARRATION_NO_PRIOR,
    NARRATION_PRIOR_HEADER,
    NO_CONTEXT,
    RETRY_OPTIMIZATION_PROMPT,
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

# How many Gemini calls to keep in flight at once within a job's frame-analysis
# loop. Every sampled frame is described independently, so this fans out across
# all frames of all shots at once. Narration generation, by contrast, runs
# sequentially (each line is fed the previous shots' frame descriptions,
# dialogue, and narration for continuity).
DEFAULT_CONCURRENCY = 4

# How many preceding shots to feed a narration line as continuity context.
NARRATION_CONTEXT_SCENES = 6


class FrameAnalysis(BaseModel):
    description: str
    entities: list[str]
    actions: list[str]
    setting: str
    on_screen_text: str | None


async def analyze_frame(
    client, frame_path, frame_time=0.0, frame_index=0, shot_duration=0.0
):
    """Describe a single sampled frame, on its own, via Gemini.

    The frame is analyzed in isolation — no neighbouring frames, no prior shot
    history — so the result is a faithful record of just this frame. Returns the
    ``FrameAnalysis`` fields as a dict.
    """
    logger.debug("analyze_frame: %s (t=%.2fs) -> %s", frame_path, frame_time, MODEL)
    with open(frame_path, "rb") as f:
        image_bytes = f.read()

    prompt = FRAME_ANALYSIS_PROMPT.format(
        frame_time=frame_time,
        frame_index=frame_index,
        shot_duration=shot_duration,
        context=NO_CONTEXT,
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
            response_schema=FrameAnalysis,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        ),
    )

    if response.text is None:
        logger.error("analyze_frame: %s returned no text for %s", MODEL, frame_path)
        raise RuntimeError(f"{MODEL} returned no text for {frame_path}")
    return FrameAnalysis.model_validate_json(response.text).model_dump()


async def analyze_shots(
    shots,
    blobs,
    on_frame: Callable[[dict, dict], Awaitable[None]] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    client=None,
):
    """Attach a ``visual`` analysis to every sampled frame of every shot.

    Each frame's jpg is resolved from its blob key through ``blobs`` (fetched
    from object storage only if it isn't already in scratch), then gets its own
    Gemini call, with up to ``concurrency`` in flight across the whole video;
    ``on_frame`` (if given) is awaited with ``(shot, frame)`` as each frame's
    analysis completes, so callers can stream results. Completion order is
    neither shot nor frame order, so consumers must key off ``shot["id"]`` and
    ``frame["index"]``.
    """
    client = client or genai.Client()
    semaphore = asyncio.Semaphore(concurrency)
    jobs = [(shot, frame) for shot in shots for frame in shot.get("frames", [])]
    logger.info(
        "analyze_shots: %d frame(s) across %d shot(s), concurrency=%d",
        len(jobs),
        len(shots),
        concurrency,
    )

    async def _run(shot, frame):
        shot_duration = shot.get("end", 0.0) - shot.get("start", 0.0)
        frame_path = await asyncio.to_thread(blobs.fetch, frame["key"])
        async with semaphore:
            frame["visual"] = await analyze_frame(
                client,
                str(frame_path),
                frame_time=frame["time"],
                frame_index=frame["index"],
                shot_duration=shot_duration,
            )
        if on_frame is not None:
            await on_frame(shot, frame)

    await asyncio.gather(*(_run(shot, frame) for shot, frame in jobs))
    logger.info("analyze_shots: all %d frame(s) analyzed", len(jobs))
    return shots


def _frame_line(frame):
    """One chronological line describing a single frame in a narration prompt."""
    visual = frame.visual
    if visual is None:
        return f"- {frame.time:.2f}s: (not analyzed)"

    parts = [f"- {frame.time:.2f}s: {visual.description}"]
    if visual.actions:
        parts.append(f"Actions: {'; '.join(visual.actions)}.")
    if visual.on_screen_text:
        parts.append(f'On-screen text: "{visual.on_screen_text}"')
    return " ".join(parts)


def _current_shot_block(segment):
    """The CURRENT SHOT block: the shot's frames in order, plus its dialogue.

    Every sampled frame contributes its own description, actions, and on-screen
    text, so the model can read the change across the shot and write a line that
    covers the whole stretch rather than one instant.
    """
    lines = [
        NARRATION_FRAMES_HEADER.format(
            start=segment.start, end=segment.end, frame_count=len(segment.frames)
        )
    ]
    lines.extend(_frame_line(frame) for frame in segment.frames)
    if segment.audio and segment.audio.transcript:
        lines.append(
            f'Dialogue in this shot (do not repeat): "{segment.audio.transcript}"'
        )
    return "\n".join(lines)


def _prior_scenes_block(prior_scenes):
    """The earlier-shots continuity block from the accumulated shot history.

    ``prior_scenes`` is a list of ``{id, descriptions, dialogue, narration}``
    dicts (oldest first), where ``descriptions`` is that shot's per-frame
    descriptions in temporal order. Each preceding shot contributes those
    descriptions, its dialogue, and the AD line already written for it, so the
    model can reuse established names, avoid repeating visuals, and keep
    descriptions continuous.
    """
    if not prior_scenes:
        return NARRATION_NO_PRIOR

    lines = [NARRATION_PRIOR_HEADER]
    for scene in prior_scenes:
        summary = " → ".join(scene.get("descriptions") or []) or "(not analyzed)"
        parts = [f"- Shot {scene['id']}: {summary}"]
        if scene.get("dialogue"):
            parts.append(f'Dialogue: "{scene["dialogue"]}"')
        if scene.get("narration"):
            parts.append(f'AD: "{scene["narration"]}"')
        lines.append(" ".join(parts))
    return "\n".join(lines)


async def generate_narration(client, segment, max_words, blobs, prior_scenes=None):
    """Write one AD narration line for ``segment``, from its whole frame sequence.

    Every frame sampled within the shot is attached as an image, in temporal
    order, alongside the frame-by-frame analyses and the accumulated earlier-shot
    context (``prior_scenes``) — so the line is the culmination of what happens
    across the shot, reuses established names, doesn't repeat prior visuals, and
    stays continuous. Returns the raw line; the caller may further condense it via
    ``optimize_narration`` to fit the gap.
    """
    prompt = (
        NARRATION_HEADER.format(gap_sec=segment.narratable_gap_sec, max_words=max_words)
        + "\n\n"
        + _current_shot_block(segment)
        + "\n\n"
        + _prior_scenes_block(prior_scenes)
        + "\n\n"
        + NARRATION_GUIDELINES
    )

    # Frame images first, in the same chronological order as the prompt's frame
    # list, so the model can line each image up with its description.
    contents = []
    for frame in segment.frames:
        frame_path = await asyncio.to_thread(blobs.fetch, frame.key)
        contents.append(
            types.Part.from_bytes(data=frame_path.read_bytes(), mime_type="image/jpeg")
        )
    contents.append(prompt)

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=contents,
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
            "generate_narration: %s returned no text for segment %s (%d frame(s))",
            MODEL,
            segment.id,
            len(segment.frames),
        )
        raise RuntimeError(f"{MODEL} returned no text for segment {segment.id}")
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
    blobs,
    on_segment: Callable[[Segment], Awaitable[None]] | None = None,
    words_per_sec: float | None = None,
    client=None,
):
    """Generate AD narration for every ``ad_eligible`` segment, in shot order.

    Unlike the frame-analysis stage, this runs sequentially: each narration line
    is written from its shot's whole frame sequence plus the preceding shots'
    frame descriptions, dialogue, and narration as context (a rolling window of
    ``NARRATION_CONTEXT_SCENES`` shots), so the model reuses established names,
    avoids repeating earlier visuals, and keeps the audio description continuous.
    Every shot — narrated or not — contributes to that history. ``on_segment``
    (if given) is awaited per narrated segment, in id order.
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
            text = await generate_narration(client, segment, max_words, blobs, prior)
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
                "descriptions": [
                    frame.visual.description
                    for frame in segment.frames
                    if frame.visual is not None
                ],
                "dialogue": dialogue,
                "narration": segment.ad_narration,
            }
        )

    logger.info("fill_narration_gaps: wrote %d narration line(s)", len(eligible))
    return timeline
