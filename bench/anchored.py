"""Building a Timeline whose segments are the ground-truth AD windows.

The pipeline normally decides *where* to speak: one candidate per shot, kept if
its longest speech-free span clears ``MIN_NARRATABLE_GAP_SEC``. The CSV records
where a human describer chose to speak. Scoring free-running output against the
CSV would therefore mix two questions — "did we describe this well?" and "did we
happen to speak at the same moment?" — and a low score would not say which.

So the harness forces the windows. Each CSV row becomes one ``Segment`` spanning
that row's ``[scaled_start, scaled_end]``, carrying the frames sampled around it
and the dialogue spoken during it, with the row's ``duration`` standing in for
the narratable gap. ``vision_analysis.fill_narration_gaps`` then runs completely
unmodified and yields exactly one line per reference line: a 1:1 pairing that
needs no temporal matching, which is what makes CIDEr and LLM-AD-eval mean what
the paper means by them.
"""

import logging

from bench.config import FRAME_PAD_SEC
from bench.dataset import AdRow
from timeline import Frame, Segment, Timeline, build_audio_analysis

logger = logging.getLogger(__name__)


def frames_for_window(
    frames: list[dict], start: float, end: float, pad_sec: float = FRAME_PAD_SEC
) -> list[dict]:
    """The sampled frames a describer would have been reacting to, in time order.

    Padded either side because a describer watches the action leading into the
    gap, not only what is on screen during it. If the padded window still
    catches nothing — a 2-second reference against 2-second sampling can fall
    between two frames — fall back to the single nearest frame: every segment
    must carry at least one image, since ``generate_narration`` attaches the
    frames as the model's only view of the video.
    """
    selected = [f for f in frames if start - pad_sec <= f["time"] <= end + pad_sec]
    if not selected and frames:
        center = (start + end) / 2
        selected = [min(frames, key=lambda f: abs(f["time"] - center))]
    return sorted(selected, key=lambda f: f["time"])


def build_anchored_timeline(
    video_id: str,
    duration_sec: float,
    shots: list[dict],
    rows: list[AdRow],
    speech_regions=None,
    transcript_segments=None,
    pad_sec: float = FRAME_PAD_SEC,
) -> Timeline:
    """One segment per ground-truth AD row, in temporal order.

    ``shots`` is segmentation's output for the whole clip — the segments here
    cut across it, drawing frames by timestamp rather than by shot membership,
    because a human AD window has no reason to respect a camera cut.
    """
    all_frames = sorted(
        (frame for shot in shots for frame in shot["frames"]),
        key=lambda f: f["time"],
    )

    segments = []
    for index, row in enumerate(sorted(rows, key=lambda r: r.scaled_start)):
        window = frames_for_window(
            all_frames, row.scaled_start, row.scaled_end, pad_sec
        )
        segments.append(
            Segment(
                # Renumbered 0..n-1 rather than carrying a CSV index: the
                # narration stage's rolling context window walks segments in id
                # order, so ids must be dense and temporal.
                id=index,
                start=row.scaled_start,
                end=row.scaled_end,
                frames=[Frame(**frame) for frame in window],
                audio=build_audio_analysis(
                    row.scaled_start,
                    row.scaled_end,
                    speech_regions or [],
                    transcript_segments or [],
                ),
                # Forced: we want a prediction for every reference line,
                # including ones our own gap heuristic would have declined.
                ad_eligible=True,
                # The human's own window becomes our word budget
                # (max_words = gap × NARRATION_WORDS_PER_SEC), so the two lines
                # are held to the same length. A text-similarity metric between
                # a 6-word reference and a 40-word prediction measures nothing.
                narratable_gap_sec=round(row.duration, 2),
                narration_start_sec=round(row.scaled_start, 2),
            )
        )

    frameless = sum(1 for seg in segments if not seg.frames)
    if frameless:
        logger.warning(
            "anchored: %d/%d segment(s) for %s have no frames at all — the clip's "
            "timestamps may not match this upload",
            frameless,
            len(segments),
            video_id,
        )
    logger.info(
        "anchored: %s -> %d segment(s), %.1f frame(s) each on average",
        video_id,
        len(segments),
        sum(len(seg.frames) for seg in segments) / len(segments) if segments else 0.0,
    )
    return Timeline(job_id=video_id, duration_sec=duration_sec, segments=segments)
