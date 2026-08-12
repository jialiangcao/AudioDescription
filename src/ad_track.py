"""Combined audio-description track.

Stitches every per-segment narration clip (stage 7 output) into a single WAV the
length of the video, each clip laid down at the moment it would play alongside
the source — i.e. at the start of the shot's narratable gap. The result is one
continuous "AD-only" track a viewer can play next to the muted video, or mix
under it, to hear the whole audio description in sync.

Kept separate from ``tts.py`` (which synthesizes the individual clips) because
this is a pure post-processing/assembly step over files already on disk.
"""

import logging

import numpy as np
import soundfile as sf

from timeline import Timeline
from tts import NARRATION_PREFIX, SAMPLE_RATE

logger = logging.getLogger(__name__)

AD_TRACK_KEY = f"{NARRATION_PREFIX}/ad_track.wav"


def _placed_clips(timeline: Timeline, blobs):
    """Yield ``(offset_sec, mono_float32_samples)`` for each narration clip.

    Placement is the segment's ``narration_start_sec`` (start of its silent gap),
    falling back to the shot start if that wasn't recorded.
    """
    for seg in timeline.segments:
        if not seg.ad_narration_key:
            continue
        clip, sr = sf.read(blobs.fetch(seg.ad_narration_key), dtype="float32")
        if clip.ndim > 1:  # collapse any stereo to mono
            clip = clip.mean(axis=1)
        if sr != SAMPLE_RATE:
            logger.warning(
                "ad_track: segment %s clip is %dHz, expected %dHz; placing anyway",
                seg.id,
                sr,
                SAMPLE_RATE,
            )
        offset = seg.narration_start_sec
        if offset is None:
            offset = seg.start
        yield float(offset), clip


def build_ad_track(timeline: Timeline, blobs) -> tuple[str, float] | None:
    """Assemble the combined AD track for ``timeline`` into ``blobs``' scratch.

    Returns ``(key, duration_sec)`` for the written WAV, or ``None`` if no
    segment had synthesized narration (nothing to assemble).
    """
    clips = list(_placed_clips(timeline, blobs))
    if not clips:
        logger.info("build_ad_track: no narration clips, skipping combined track")
        return None

    # Track spans the whole video, extended if any clip (e.g. an overflowing one)
    # would run past the end.
    total_sec = timeline.duration_sec
    for offset, clip in clips:
        total_sec = max(total_sec, offset + len(clip) / SAMPLE_RATE)
    track = np.zeros(int(round(total_sec * SAMPLE_RATE)), dtype=np.float32)

    for offset, clip in clips:
        start = max(0, int(round(offset * SAMPLE_RATE)))
        end = start + len(clip)
        if end > len(track):  # guard against rounding pushing us past the buffer
            track = np.concatenate(
                [track, np.zeros(end - len(track), dtype=np.float32)]
            )
        track[start:end] += clip

    # Overlapping clips (rare, only if an overflow spills into the next gap) can
    # sum past full scale; keep the mix in range rather than clipping harshly.
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 1.0:
        track /= peak

    sf.write(str(blobs.path(AD_TRACK_KEY)), track, SAMPLE_RATE)
    duration = len(track) / SAMPLE_RATE
    logger.info(
        "build_ad_track: wrote %s (%.1fs, %d clip(s))",
        AD_TRACK_KEY,
        duration,
        len(clips),
    )
    return AD_TRACK_KEY, round(duration, 3)
