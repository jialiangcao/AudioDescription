"""Narration text-to-speech (stage 7).

Turns each ``ad_narration`` line into a spoken WAV clip with Kokoro-82M. Ported
from the CLI ``synthesize_narration``: instead of writing to a fixed
``src/out/narration`` dir and running synchronously, it writes to a job-scoped
``out_dir`` and is ``async`` so the web pipeline can stream per-segment progress
and keep a worker's event loop responsive. Kokoro itself is a blocking CPU-bound
model, so each synthesis runs in a worker thread via ``asyncio.to_thread``; calls
are serialized (one clip at a time) because the pipeline is a single shared,
non-reentrant model instance.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

import numpy as np
import soundfile as sf

from timeline import Segment, Timeline

logger = logging.getLogger(__name__)

VOICE = "af_heart"
SAMPLE_RATE = 24000

# Blob-key prefix for synthesized clips; keys look like narration/shot_0000.wav.
NARRATION_PREFIX = "narration"

# If a synthesized clip overflows its gap, retry once at a higher speed, capped here
# so the voice doesn't get distorted trying to fit an arbitrarily long line.
MAX_SPEED = 1.3

# Leading silence prepended to each clip, as a fraction of the shot's length, so
# narration eases in a beat after the cut instead of starting abruptly at the top
# of the gap. This delay eats into the gap, so the spoken part gets the remainder.
START_PAD_FRACTION = 0.1

_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        logger.info("loading Kokoro TTS pipeline (voice=%s)", VOICE)
        from kokoro import KPipeline

        _PIPELINE = KPipeline(lang_code="a")  # American English
    return _PIPELINE


def _synthesize(text, speed=1.0):
    pipeline = _load_pipeline()
    chunks = [
        np.asarray(audio)
        for _gs, _ps, audio in pipeline(text, voice=VOICE, speed=speed)
        if audio is not None
    ]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _duration_sec(audio):
    return len(audio) / SAMPLE_RATE


def _prepend_silence(audio, pad_sec):
    if pad_sec <= 0:
        return audio
    pad = np.zeros(int(round(pad_sec * SAMPLE_RATE)), dtype=np.float32)
    return np.concatenate([pad, audio])


def narration_key(segment_id: int) -> str:
    return f"{NARRATION_PREFIX}/shot_{segment_id:04d}.wav"


def _synthesize_segment(segment: Segment, gap_sec: float, blobs) -> None:
    """Synthesize one segment's narration and write its WAV + timing metadata.

    Runs in a worker thread (blocking Kokoro inference + file write). The clip
    gets ``START_PAD_FRACTION`` of the shot's length as leading silence so it
    doesn't start on the cut; that pad eats into the gap, so the spoken part is
    fit to the *remaining* budget. Since ``narratable_gap_sec`` is only an
    estimate of how much speech fits, a spoken part longer than that budget is
    re-synthesized once at a higher speed (capped at ``MAX_SPEED``); if the
    padded clip is still longer than the gap it's kept and flagged via
    ``ad_narration_overflow`` rather than retried again.
    """
    pad_sec = START_PAD_FRACTION * (segment.end - segment.start)
    speech_budget = max(gap_sec - pad_sec, 0.0)

    audio = _synthesize(segment.ad_narration)
    duration = _duration_sec(audio)

    if speech_budget > 0 and duration > speech_budget:
        speed = min(MAX_SPEED, duration / speech_budget)
        logger.debug(
            "segment %s: %.2fs clip over %.2fs speech budget, re-synthesizing at speed=%.2f",
            segment.id,
            duration,
            speech_budget,
            speed,
        )
        audio = _synthesize(segment.ad_narration, speed=speed)

    audio = _prepend_silence(audio, pad_sec)
    duration = _duration_sec(audio)

    key = narration_key(segment.id)
    sf.write(str(blobs.path(key)), audio, SAMPLE_RATE)

    segment.ad_narration_key = key
    segment.ad_narration_duration_sec = round(duration, 3)
    segment.ad_narration_overflow = duration > gap_sec
    if segment.ad_narration_overflow:
        logger.warning(
            "segment %s: narration still overflows gap after speed-up (%.2fs > %.2fs)",
            segment.id,
            duration,
            gap_sec,
        )


async def synthesize_narration(
    timeline: Timeline,
    blobs,
    on_segment: Callable[[Segment], Awaitable[None]] | None = None,
    retry_optimize: Callable[[Segment, float, float], Awaitable[str]] | None = None,
) -> Timeline:
    """Synthesize narration audio for every segment that has ``ad_narration``.

    Clips are written into ``blobs``' scratch dir under ``narration/`` and
    recorded on each segment as ``ad_narration_key``; uploading them is the
    caller's job. ``on_segment`` (if given) is awaited with each segment as its
    clip finishes, so callers can stream results. ``retry_optimize`` (if given)
    is awaited when a synthesized clip still overruns its gap: it receives
    ``(segment, tts_duration, gap_sec)`` and returns a shortened narration line,
    which is then re-synthesized once (the paper's TTS-verified retry
    optimization). Unlike the Gemini stages this runs one clip at a time (the
    Kokoro pipeline is a single shared model), so segments complete in id order.
    """
    synthesized = 0
    for segment in timeline.segments:
        if (
            not segment.ad_eligible
            or not segment.ad_narration
            or segment.narratable_gap_sec is None
        ):
            continue

        gap_sec = segment.narratable_gap_sec
        logger.debug(
            "synthesize_narration: segment %s -> %s",
            segment.id,
            narration_key(segment.id),
        )
        await asyncio.to_thread(_synthesize_segment, segment, gap_sec, blobs)

        # Retry optimization: if the clip still overruns the gap, ask the model
        # for a shorter line using the measured duration, then re-synthesize once.
        if segment.ad_narration_overflow and retry_optimize is not None:
            measured = segment.ad_narration_duration_sec or 0.0
            shorter = await retry_optimize(segment, measured, gap_sec)
            if shorter and shorter != segment.ad_narration:
                logger.debug(
                    "synthesize_narration: segment %s re-synthesizing shortened line",
                    segment.id,
                )
                segment.ad_narration = shorter
                await asyncio.to_thread(_synthesize_segment, segment, gap_sec, blobs)

        synthesized += 1
        if on_segment is not None:
            await on_segment(segment)

    logger.info("synthesize_narration: wrote %d narration clip(s)", synthesized)
    return timeline
