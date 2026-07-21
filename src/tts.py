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
import os
from collections.abc import Awaitable, Callable

import numpy as np
import soundfile as sf

from timeline import Segment, Timeline

VOICE = "af_heart"
SAMPLE_RATE = 24000

# If a synthesized clip overflows its gap, retry once at a higher speed, capped here
# so the voice doesn't get distorted trying to fit an arbitrarily long line.
MAX_SPEED = 1.3

_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
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


def _synthesize_segment(segment: Segment, gap_sec: float, out_dir: str) -> None:
    """Synthesize one segment's narration and write its WAV + timing metadata.

    Runs in a worker thread (blocking Kokoro inference + file write). Since
    ``narratable_gap_sec`` is only an estimate of how much speech fits, a clip
    that comes out longer than its gap is re-synthesized once at a higher speed
    (capped at ``MAX_SPEED``); if it's still too long it's kept and flagged via
    ``ad_narration_overflow`` rather than retried again.
    """
    audio = _synthesize(segment.ad_narration)
    duration = _duration_sec(audio)

    if duration > gap_sec:
        speed = min(MAX_SPEED, duration / gap_sec)
        audio = _synthesize(segment.ad_narration, speed=speed)
        duration = _duration_sec(audio)

    path = os.path.join(out_dir, f"shot_{segment.id:04d}.wav")
    sf.write(path, audio, SAMPLE_RATE)

    segment.ad_narration_audio = path
    segment.ad_narration_duration_sec = round(duration, 3)
    segment.ad_narration_overflow = duration > gap_sec


async def synthesize_narration(
    timeline: Timeline,
    out_dir: str = "narration",
    on_segment: Callable[[Segment], Awaitable[None]] | None = None,
) -> Timeline:
    """Synthesize narration audio for every segment that has ``ad_narration``.

    ``on_segment`` (if given) is awaited with each segment as its clip finishes,
    so callers can stream results. Unlike the Gemini stages this runs one clip at
    a time (the Kokoro pipeline is a single shared model), so segments complete
    in id order.
    """
    os.makedirs(out_dir, exist_ok=True)

    for segment in timeline.segments:
        if (
            not segment.ad_eligible
            or not segment.ad_narration
            or segment.narratable_gap_sec is None
        ):
            continue

        await asyncio.to_thread(
            _synthesize_segment, segment, segment.narratable_gap_sec, out_dir
        )
        if on_segment is not None:
            await on_segment(segment)

    return timeline
