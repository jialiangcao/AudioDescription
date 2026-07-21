import transcription
from transcription import _overlaps_speech, transcribe


def test_overlaps_speech_true_for_overlapping_region():
    assert _overlaps_speech(1.0, 2.0, [(1.5, 3.0)]) is True


def test_overlaps_speech_false_when_disjoint():
    assert _overlaps_speech(1.0, 2.0, [(2.0, 3.0)]) is False


def test_overlaps_speech_false_for_empty_regions():
    assert _overlaps_speech(1.0, 2.0, []) is False


class _FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _FakeWhisperModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio_path, vad_filter):
        return self._segments, None


def test_transcribe_drops_hallucinated_and_empty_segments(monkeypatch):
    segments = [
        _FakeSegment(0.0, 1.0, "  hello there  "),
        _FakeSegment(5.0, 6.0, "hallucinated over silence"),
        _FakeSegment(0.2, 0.4, "   "),
    ]
    monkeypatch.setattr(
        transcription, "_load_model", lambda: _FakeWhisperModel(segments)
    )

    result = transcribe("audio.wav", speech_regions=[(0.0, 1.0)])

    # only the segment overlapping a real speech region and with non-empty
    # text survives; text is stripped
    assert result == [{"start": 0.0, "end": 1.0, "text": "hello there"}]
