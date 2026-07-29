import logging

import cv2
from pydantic import BaseModel, Field

from voice_activity import longest_speech_free_span

logger = logging.getLogger(__name__)


class VisualAnalysis(BaseModel):
    description: str
    entities: list[str]
    setting: str
    on_screen_text: str | None


class AudioAnalysis(BaseModel):
    has_speech: bool
    transcript: str | None
    silence_ratio: float


# Minimum speech-free time within a shot for it to be worth narrating.
MIN_NARRATABLE_GAP_SEC = 2.0

# Standard audio-description pacing, in words per second.
NARRATION_WORDS_PER_SEC = 2.5


class Segment(BaseModel):
    id: int
    start: float
    end: float
    keyframe: str
    # Every frame sampled within the shot (segmentation.py's interval_sec), for CLIP retrieval.
    keyframes: list[str] = Field(default_factory=list)
    visual: VisualAnalysis
    audio: AudioAnalysis | None = None
    ad_eligible: bool | None = None
    narratable_gap_sec: float | None = None
    # Absolute time (sec) where the narratable gap begins — i.e. when this shot's
    # narration would start playing alongside the video. Used to place clips in
    # the combined AD track.
    narration_start_sec: float | None = None
    # Populated by vision_analysis.fill_narration_gaps for ad_eligible segments.
    ad_narration: str | None = None
    # Populated by tts.synthesize_narration for segments with ad_narration set.
    # ad_narration_audio is the server-side path to the synthesized WAV clip.
    ad_narration_audio: str | None = None
    ad_narration_duration_sec: float | None = None
    ad_narration_overflow: bool | None = None


class Timeline(BaseModel):
    video_id: str
    duration_sec: float
    segments: list[Segment]
    # Populated by ad_track.build_ad_track: a single WAV with every narration clip
    # placed at the moment it would play alongside the video. ad_track_audio is
    # the server-side path; pass its basename through the narration endpoint.
    ad_track_audio: str | None = None
    ad_track_duration_sec: float | None = None
    # Populated by mux.mux_described_video: the source video with the AD track
    # mixed into its soundtrack, i.e. the file to watch. Server-side path; the
    # frontend fetches it from the job's described-video endpoint.
    described_video: str | None = None


def _overlap_sec(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _speech_seconds_within(start, end, speech_regions):
    return sum(
        _overlap_sec(start, end, s_start, s_end) for s_start, s_end in speech_regions
    )


def _transcript_within(start, end, transcript_segments):
    texts = [
        seg["text"]
        for seg in transcript_segments
        if _overlap_sec(start, end, seg["start"], seg["end"]) > 0
    ]
    return " ".join(texts) if texts else None


def _build_audio_analysis(start, end, speech_regions, transcript_segments):
    duration = end - start
    speech_sec = _speech_seconds_within(start, end, speech_regions)
    silence_ratio = 1.0 - (speech_sec / duration) if duration > 0 else 1.0

    return AudioAnalysis(
        has_speech=speech_sec > 0,
        transcript=_transcript_within(start, end, transcript_segments),
        silence_ratio=round(silence_ratio, 4),
    )


def build_timeline(video_path, shots, speech_regions=None, transcript_segments=None):
    speech_regions = speech_regions or []
    transcript_segments = transcript_segments or []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if fps else 0.0
    cap.release()
    if not fps:
        logger.warning(
            "build_timeline: could not read fps for %s, duration=0", video_path
        )

    segments = []
    for shot in shots:
        start, end = shot["start"], shot["end"]
        audio = _build_audio_analysis(start, end, speech_regions, transcript_segments)
        # Narration must fit one uninterrupted silent stretch, so size the gap to
        # the longest contiguous speech-free span — not the total silence, which
        # over-estimates the fit and makes narration overrun (and get cut off).
        gap_start, gap_len = longest_speech_free_span(start, end, speech_regions)
        narratable_gap_sec = round(gap_len, 2)

        segments.append(
            Segment(
                id=shot["id"],
                start=start,
                end=end,
                keyframe=shot["keyframe"],
                keyframes=shot.get("keyframes", []),
                visual=VisualAnalysis(**shot["visual"]),
                audio=audio,
                ad_eligible=narratable_gap_sec >= MIN_NARRATABLE_GAP_SEC,
                narratable_gap_sec=narratable_gap_sec,
                narration_start_sec=round(gap_start, 2),
            )
        )

    eligible = sum(1 for s in segments if s.ad_eligible)
    logger.debug(
        "build_timeline: %s -> %d segment(s), %d AD-eligible, duration=%.2fs",
        video_path,
        len(segments),
        eligible,
        duration_sec,
    )
    return Timeline(video_id=video_path, duration_sec=duration_sec, segments=segments)


def save_timeline(timeline, out_path="timeline.json"):
    with open(out_path, "w") as f:
        f.write(timeline.model_dump_json(indent=2))
    return out_path


def load_timeline(path="timeline.json"):
    with open(path) as f:
        return Timeline.model_validate_json(f.read())
