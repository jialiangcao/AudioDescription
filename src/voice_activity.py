import logging

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

logger = logging.getLogger(__name__)

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        logger.info("loading Silero VAD model")
        _MODEL = load_silero_vad()
    return _MODEL


def detect_speech_regions(audio_path, threshold=0.5, sample_rate=16000):
    """Run Silero VAD over audio_path.

    Returns a sorted list of non-overlapping (start_sec, end_sec) speech regions.
    """
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
