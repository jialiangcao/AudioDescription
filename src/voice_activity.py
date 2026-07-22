import logging

logger = logging.getLogger(__name__)

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        logger.info("loading Silero VAD model")
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
    return _MODEL


def longest_speech_free_gap(start, end, speech_regions):
    """Length (sec) of the longest contiguous speech-free span within ``[start, end]``.

    Narration for a shot has to be spoken during a single uninterrupted silent
    stretch, so the amount that actually fits is bounded by the *longest* such
    stretch — not by the total silence, which sums up unrelated pauses scattered
    between dialogue and wildly over-estimates how much narration will fit.

    ``speech_regions`` are ``(start_sec, end_sec)`` spans (as returned by
    :func:`detect_speech_regions`); they may overlap or extend outside the
    window and need not be sorted.
    """
    if end <= start:
        return 0.0

    # Clip each speech region to the window, drop non-overlapping ones, sort.
    clipped = sorted(
        (max(start, s), min(end, e))
        for s, e in speech_regions
        if min(end, e) > max(start, s)
    )

    longest = 0.0
    cursor = start
    for s, e in clipped:
        if s > cursor:
            longest = max(longest, s - cursor)
        cursor = max(cursor, e)
    longest = max(longest, end - cursor)
    return longest


def detect_speech_regions(audio_path, threshold=0.5, sample_rate=16000):
    """Run Silero VAD over audio_path.

    Returns a sorted list of non-overlapping (start_sec, end_sec) speech regions.
    """
    from silero_vad import get_speech_timestamps, read_audio

    model = _load_model()
    wav = read_audio(audio_path, sampling_rate=sample_rate)
    timestamps = get_speech_timestamps(
        wav, model, threshold=threshold, sampling_rate=sample_rate, return_seconds=True
    )
    logger.debug(
        "detect_speech_regions: %s -> %d region(s) (threshold=%.2f)",
        audio_path,
        len(timestamps),
        threshold,
    )
    return [(t["start"], t["end"]) for t in timestamps]
