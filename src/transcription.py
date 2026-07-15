from faster_whisper import WhisperModel

MODEL_SIZE = "small"

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
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
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if not _overlaps_speech(segment.start, segment.end, speech_regions):
            continue
        dialogue.append({"start": segment.start, "end": segment.end, "text": text})

    return dialogue
