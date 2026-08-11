"""Stage orchestration for the audio-description pipeline.

Extracted from the old ``main.py::process_video`` CLI. Instead of printing to
the console and reading/writing fixed ``src/in``/``src/out`` paths, this drives
the same eight stages against explicit job-scoped paths and reports progress via
an async ``on_event`` callback, so the web layer can stream results live.

The function is intentionally ignorant of FastAPI/JobStore — it just awaits a
callback — which keeps it unit-testable in isolation.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from ad_track import build_ad_track
from audio_extract import AUDIO_KEY, NoAudioStreamError, extract_audio
from mux import DESCRIBED_KEY, mux_described_video
from segmentation import probe_video, segment_video
from timeline import Timeline, build_timeline
from transcription import transcribe
from tts import synthesize_narration
from vision_analysis import (
    analyze_shots,
    fill_narration_gaps,
    retry_optimize_narration,
)
from voice_activity import detect_speech_regions

logger = logging.getLogger(__name__)

OnEvent = Callable[[dict], Awaitable[None]]


def _shots_event(shots: list[dict]) -> dict:
    """The shot/frame skeleton, emitted before any frame has been described.

    Lets the client show every extracted frame with its timestamp immediately;
    the per-frame ``frame`` events then fill in each frame's analysis.
    """
    return {
        "type": "shots",
        "shots": [
            {
                "id": shot["id"],
                "start": shot["start"],
                "end": shot["end"],
                "frames": [
                    {
                        "index": frame["index"],
                        "time": frame["time"],
                        "key": frame["key"],
                    }
                    for frame in shot["frames"]
                ],
            }
            for shot in shots
        ],
    }


def _frame_event(shot: dict, frame: dict) -> dict:
    return {
        "type": "frame",
        "shot_id": shot["id"],
        "index": frame["index"],
        "time": frame["time"],
        "key": frame["key"],
        "visual": frame.get("visual"),
    }


async def run_pipeline(
    video_path: Path,
    blobs,
    on_event: OnEvent,
    client=None,
) -> Timeline:
    """Run the full AD pipeline for one job, in one process.

    This is the reference implementation of the stage sequence and the path used
    for local runs and tests; the deployed system runs the same stage functions
    as separate Celery tasks. Artifacts are addressed by job-relative blob key
    and written into ``blobs``' scratch dir.

    ``on_event`` is awaited on the event loop with dict events: ``stage``
    (start/done markers), ``shots`` (the shot/frame skeleton with timestamps),
    ``frame`` (one frame's vision result), ``timeline`` (the assembled timeline
    before narration), ``narration`` (per-segment AD text), ``narration_audio``
    (per-segment synthesized clip: key, duration, overflow), ``ad_track``
    (the combined AD-only track) and ``described_video`` (the final video with
    narration mixed in). Frame and narration events arrive out of order —
    consumers must key off ``shot_id``/``index`` and ``segment_id``.
    """
    video_path = Path(video_path)
    audio_path = blobs.path(AUDIO_KEY)
    logger.info("pipeline start: %s", video_path)

    # 1. Shot segmentation (CPU-bound: PySceneDetect + OpenCV).
    logger.info("stage 1/8 segmentation: detecting shots + sampling frames")
    await on_event({"type": "stage", "stage": "segmentation", "status": "start"})
    t0 = time.monotonic()
    meta = await asyncio.to_thread(probe_video, str(video_path))
    shots = await asyncio.to_thread(segment_video, str(video_path), blobs)
    frame_count = sum(len(shot["frames"]) for shot in shots)
    logger.info(
        "stage 1/8 segmentation done: %d shot(s), %d frame(s) in %.1fs",
        len(shots),
        frame_count,
        time.monotonic() - t0,
    )
    # Publish the skeleton first so the client can show every extracted frame and
    # its timestamp while the descriptions are still being generated.
    await on_event(_shots_event(shots))
    await on_event(
        {
            "type": "stage",
            "stage": "segmentation",
            "status": "done",
            "count": len(shots),
            "frame_count": frame_count,
        }
    )

    # 2. Vision analysis: one Gemini call per sampled frame, run concurrently.
    logger.info("stage 2/8 vision: describing %d frame(s) via Gemini", frame_count)
    await on_event({"type": "stage", "stage": "vision", "status": "start"})
    t0 = time.monotonic()

    async def _on_frame(shot: dict, frame: dict) -> None:
        logger.debug("shot %s frame %s described", shot["id"], frame["index"])
        await on_event(_frame_event(shot, frame))

    shots = await analyze_shots(shots, blobs, on_frame=_on_frame, client=client)
    logger.info("stage 2/8 vision done in %.1fs", time.monotonic() - t0)
    await on_event({"type": "stage", "stage": "vision", "status": "done"})

    # 3-5. Audio: extraction (CPU/ffmpeg) → VAD → transcription. A video with
    # no audio track skips straight past these with empty results, exactly as
    # the original CLI did.
    speech_regions: list = []
    transcript_segments: list = []
    logger.info("stage 3-5/8 audio: extract -> VAD -> transcribe")
    await on_event({"type": "stage", "stage": "audio", "status": "start"})
    t0 = time.monotonic()
    try:
        await asyncio.to_thread(
            extract_audio, str(video_path), out_path=str(audio_path)
        )
        speech_regions = await asyncio.to_thread(detect_speech_regions, str(audio_path))
        transcript_segments = await asyncio.to_thread(
            transcribe, str(audio_path), speech_regions
        )
        logger.info(
            "stage 3-5/8 audio done in %.1fs: %d speech region(s), %d dialogue segment(s)",
            time.monotonic() - t0,
            len(speech_regions),
            len(transcript_segments),
        )
        await on_event(
            {
                "type": "stage",
                "stage": "audio",
                "status": "done",
                "has_audio": True,
                "speech_regions": len(speech_regions),
                "dialogue_segments": len(transcript_segments),
            }
        )
    except NoAudioStreamError:
        logger.warning(
            "stage 3-5/8 audio: no audio stream, skipping VAD + transcription"
        )
        await on_event(
            {"type": "stage", "stage": "audio", "status": "done", "has_audio": False}
        )

    # 6. Timeline assembly (CPU-bound) then narration generation (Gemini).
    logger.info("stage 6/8 timeline: assembling segments")
    await on_event({"type": "stage", "stage": "timeline", "status": "start"})
    timeline = await asyncio.to_thread(
        build_timeline,
        blobs.job_id,
        meta["duration_sec"],
        shots,
        speech_regions,
        transcript_segments,
    )
    eligible = sum(1 for s in timeline.segments if s.ad_eligible)
    logger.info(
        "stage 6/8 timeline done: %d segment(s), %d AD-eligible",
        len(timeline.segments),
        eligible,
    )
    await on_event(
        {
            "type": "timeline",
            "timeline": timeline.model_dump(),
        }
    )
    await on_event({"type": "stage", "stage": "timeline", "status": "done"})

    logger.info(
        "stage 6/8 narration: writing AD lines for %d eligible segment(s)", eligible
    )
    await on_event({"type": "stage", "stage": "narration", "status": "start"})
    t0 = time.monotonic()

    async def _on_segment(segment) -> None:
        logger.debug("segment %s narration: %r", segment.id, segment.ad_narration)
        await on_event(
            {
                "type": "narration",
                "segment_id": segment.id,
                "text": segment.ad_narration,
            }
        )

    timeline = await fill_narration_gaps(
        timeline, blobs, on_segment=_on_segment, client=client
    )
    logger.info("stage 6/8 narration done in %.1fs", time.monotonic() - t0)
    await on_event({"type": "stage", "stage": "narration", "status": "done"})

    # 7. Narration text-to-speech (CPU-bound Kokoro, one clip at a time).
    logger.info("stage 7/8 tts: synthesizing narration audio")
    await on_event({"type": "stage", "stage": "tts", "status": "start"})
    t0 = time.monotonic()

    async def _on_narration_audio(segment) -> None:
        if segment.ad_narration_overflow:
            logger.warning(
                "segment %s narration overflows its gap (%.2fs clip)",
                segment.id,
                segment.ad_narration_duration_sec,
            )
        else:
            logger.debug(
                "segment %s narration audio: %.2fs",
                segment.id,
                segment.ad_narration_duration_sec,
            )
        await on_event(
            {
                "type": "narration_audio",
                "segment_id": segment.id,
                "audio": segment.ad_narration_key,
                "duration_sec": segment.ad_narration_duration_sec,
                "overflow": segment.ad_narration_overflow,
            }
        )

    async def _retry_optimize(segment, tts_duration, gap_sec) -> str:
        return await retry_optimize_narration(
            client, segment.ad_narration, tts_duration, gap_sec
        )

    # Only wire the TTS-verified retry pass when we actually have a Gemini client.
    retry_optimize: Callable[..., Awaitable[str]] | None = (
        _retry_optimize if client is not None else None
    )

    timeline = await synthesize_narration(
        timeline,
        blobs,
        on_segment=_on_narration_audio,
        retry_optimize=retry_optimize,
    )
    logger.info("stage 7/8 tts done in %.1fs", time.monotonic() - t0)

    # 8. Assemble every narration clip into one video-length "AD-only" track (each
    # clip placed where it would play alongside the source), then mux that track
    # into the video so it can be watched with narration and original sound
    # together. Nothing to do if no segment produced narration.
    logger.info("stage 8/8 mux: building AD track + described video")
    await on_event({"type": "stage", "stage": "mux", "status": "start"})
    t0 = time.monotonic()

    result = await asyncio.to_thread(build_ad_track, timeline, blobs)
    if result is None:
        logger.info("stage 8/8 mux: no narration to mux, skipping described video")
        await on_event(
            {"type": "stage", "stage": "mux", "status": "done", "described": False}
        )
        logger.info("pipeline complete: %s", video_path)
        return timeline

    ad_track_key, ad_track_duration = result
    timeline.ad_track_key = ad_track_key
    timeline.ad_track_duration_sec = ad_track_duration
    await on_event(
        {
            "type": "ad_track",
            "audio": ad_track_key,
            "duration_sec": ad_track_duration,
        }
    )

    await asyncio.to_thread(
        mux_described_video,
        video_path,
        blobs.fetch(ad_track_key),
        blobs.path(DESCRIBED_KEY),
    )
    timeline.described_key = DESCRIBED_KEY
    logger.info("stage 8/8 mux done in %.1fs", time.monotonic() - t0)
    await on_event({"type": "described_video", "video": DESCRIBED_KEY})
    await on_event(
        {"type": "stage", "stage": "mux", "status": "done", "described": True}
    )

    logger.info("pipeline complete: %s", video_path)

    return timeline
