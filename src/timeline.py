import logging

from pydantic import BaseModel, Field

from voice_activity import longest_speech_free_span

logger = logging.getLogger(__name__)


class FrameAnalysis(BaseModel):
    """The vision model's description of one sampled frame, on its own."""

    description: str
    entities: list[str]
    # Short phrases, one per distinct action visible in the frame.
    actions: list[str] = Field(default_factory=list)
    setting: str
    on_screen_text: str | None


class Frame(BaseModel):
    """One sampled frame of a shot, with its own independent analysis.

    ``index`` is the frame's position within its shot, ``time`` its absolute
    timestamp in the video, and ``key`` the job-relative blob key of the jpg
    (e.g. ``frames/shot_0000_00.jpg``) — resolve it through ``blobs.JobBlobs``
    to read the bytes, or presign it to hand the browser a URL. ``visual`` is
    None until the frame's Gemini call lands.
    """

    index: int
    time: float
    key: str
    visual: FrameAnalysis | None = None


class AudioAnalysis(BaseModel):
    has_speech: bool
    transcript: str | None
    silence_ratio: float


# Minimum speech-free time within a shot for it to be worth narrating.
MIN_NARRATABLE_GAP_SEC = 2.0

# Audio-description pacing, in words per second — the narration word budget is
# this times the gap the line has to fit in.
#
# Measured, not assumed: across 134 CMD-AD reference lines the human describers
# average 3.46 w/s (median 3.72). At the previous 2.5, 82% of reference lines
# would not have fit the budget we gave ourselves, and our narration came out
# systematically shorter than the reference whether or not it described the right
# thing. Raising it to 3.4 moved CIDEr 29.2 -> 41.3 and LLM-AD-eval 1.38 -> 1.66
# over the same rows (better on 40 pairs, worse on 19, sign test p = 0.004).
# See `bench/` for the harness that produced those numbers.
NARRATION_WORDS_PER_SEC = 3.4


class Segment(BaseModel):
    id: int
    start: float
    end: float
    # Every frame sampled within the shot (segmentation.py's interval_sec), in
    # temporal order, each carrying its own independent analysis. There is no
    # shot-level rollup: narration is written from the whole frame sequence, and
    # Q&A retrieves over these frames.
    frames: list[Frame] = Field(default_factory=list)
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
    # ad_narration_key is the job-relative blob key of the synthesized WAV clip.
    ad_narration_key: str | None = None
    ad_narration_duration_sec: float | None = None
    ad_narration_overflow: bool | None = None


class Timeline(BaseModel):
    # The job this timeline belongs to; blob keys below are relative to it.
    job_id: str
    duration_sec: float
    segments: list[Segment]
    # Populated by ad_track.build_ad_track: a single WAV with every narration clip
    # placed at the moment it would play alongside the video.
    ad_track_key: str | None = None
    ad_track_duration_sec: float | None = None
    # Populated by mux.mux_described_video: the source video with the AD track
    # mixed into its soundtrack, i.e. the file to watch.
    described_key: str | None = None


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


def build_audio_analysis(start, end, speech_regions, transcript_segments):
    """The speech picture for one window: does it talk, what is said, how quiet.

    Public because segments are not always shots — the benchmark harness builds
    them from ground-truth AD windows and needs the same dialogue context the
    real pipeline gives a narration line.
    """
    duration = end - start
    speech_sec = _speech_seconds_within(start, end, speech_regions)
    silence_ratio = 1.0 - (speech_sec / duration) if duration > 0 else 1.0

    return AudioAnalysis(
        has_speech=speech_sec > 0,
        transcript=_transcript_within(start, end, transcript_segments),
        silence_ratio=round(silence_ratio, 4),
    )


def build_timeline(
    job_id, duration_sec, shots, speech_regions=None, transcript_segments=None
):
    """Merge shots, speech regions and transcript into a Timeline.

    ``duration_sec`` comes from the segmentation stage's probe rather than being
    re-read here, so this function needs no access to the source video — which
    is what lets it run on a worker that never downloads it.
    """
    speech_regions = speech_regions or []
    transcript_segments = transcript_segments or []

    segments = []
    for shot in shots:
        start, end = shot["start"], shot["end"]
        audio = build_audio_analysis(start, end, speech_regions, transcript_segments)
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
                frames=[Frame(**frame) for frame in shot.get("frames", [])],
                audio=audio,
                ad_eligible=narratable_gap_sec >= MIN_NARRATABLE_GAP_SEC,
                narratable_gap_sec=narratable_gap_sec,
                narration_start_sec=round(gap_start, 2),
            )
        )

    eligible = sum(1 for s in segments if s.ad_eligible)
    logger.debug(
        "build_timeline: job %s -> %d segment(s), %d AD-eligible, duration=%.2fs",
        job_id,
        len(segments),
        eligible,
        duration_sec,
    )
    return Timeline(job_id=job_id, duration_sec=duration_sec, segments=segments)


def save_timeline(timeline, out_path="timeline.json"):
    with open(out_path, "w") as f:
        f.write(timeline.model_dump_json(indent=2))
    return out_path


def load_timeline(path="timeline.json"):
    with open(path) as f:
        return Timeline.model_validate_json(f.read())
