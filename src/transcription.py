import logging

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

MODEL_SIZE = "small"

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        logger.info("loading faster-whisper model %r (cpu/int8)", MODEL_SIZE)
        _MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _MODEL


def _overlaps_speech(start, end, speech_regions):
    return any(start < s_end and end > s_start for s_start, s_end in speech_regions)


def transcribe(audio_path, speech_regions):
    """Transcribe audio_path with faster-whisper.

    speech_regions (from Silero VAD) is used to drop Whisper segments that
    don't overlap any detected speech, which suppresses Whisper hallucinating
    text over music/silence.

    Returns a list of {start, end, text} dicts, sorted by start.
    """
    model = _load_model()
    segments, _info = model.transcribe(audio_path, vad_filter=False)

    dialogue = []
    dropped = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if not _overlaps_speech(segment.start, segment.end, speech_regions):
            dropped += 1
            logger.debug(
                "transcribe: dropping non-speech segment [%.2f-%.2f]: %r",
                segment.start,
                segment.end,
                text,
            )
            continue
        dialogue.append({"start": segment.start, "end": segment.end, "text": text})

    logger.debug(
        "transcribe: %s -> %d kept, %d dropped (no VAD overlap)",
        audio_path,
        len(dialogue),
        dropped,
    )
    return dialogue
