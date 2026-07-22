"""Stage orchestration for the audio-description pipeline.

Extracted from the old ``main.py::process_video`` CLI. Instead of printing to
the console and reading/writing fixed ``src/in``/``src/out`` paths, this drives
the same six stages against explicit job-scoped paths and reports progress via
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
from audio_extract import NoAudioStreamError, extract_audio
from segmentation import segment_video
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


def _shot_event(shot: dict) -> dict:
    return {
        "type": "shot",
        "shot": {
            "id": shot["id"],
            "start": shot["start"],
            "end": shot["end"],
            "keyframe": shot["keyframe"],
            "visual": shot.get("visual"),
        },
    }


async def run_pipeline(
    video_path: Path,
    job_dir: Path,
    on_event: OnEvent,
    client=None,
) -> Timeline:
    """Run the full AD pipeline for one job.

    ``on_event`` is awaited on the event loop with dict events: ``stage``
    (start/done markers), ``shot`` (per-shot vision result), ``timeline`` (the
    assembled timeline before narration), ``narration`` (per-segment AD text),
    and ``narration_audio`` (per-segment synthesized clip: filename, duration,
    overflow). Shot and narration events arrive out of order — consumers must
    key off the ``id`` field.
    """
    video_path = Path(video_path)
    frames_dir = job_dir / "frames"
    audio_path = job_dir / "audio.wav"
    narration_dir = job_dir / "narration"
    logger.info("pipeline start: %s", video_path)

    # 1. Shot segmentation (CPU-bound: PySceneDetect + OpenCV).
    logger.info("stage 1/7 segmentation: detecting shots + keyframes")
    await on_event({"type": "stage", "stage": "segmentation", "status": "start"})
    t0 = time.monotonic()
    shots = await asyncio.to_thread(
        segment_video, str(video_path), out_dir=str(frames_dir)
    )
    logger.info(
        "stage 1/7 segmentation done: %d shot(s) in %.1fs",
        len(shots),
        time.monotonic() - t0,
    )
    await on_event(
        {
            "type": "stage",
            "stage": "segmentation",
            "status": "done",
            "count": len(shots),
        }
    )

    # 2. Vision analysis (network-bound Gemini calls, run concurrently).
    logger.info("stage 2/7 vision: describing %d shot(s) via Gemini", len(shots))
    await on_event({"type": "stage", "stage": "vision", "status": "start"})
    t0 = time.monotonic()

    async def _on_shot(shot: dict) -> None:
        logger.debug("shot %s described", shot["id"])
        await on_event(_shot_event(shot))

    shots = await analyze_shots(shots, on_shot=_on_shot, client=client)
    logger.info("stage 2/7 vision done in %.1fs", time.monotonic() - t0)
    await on_event({"type": "stage", "stage": "vision", "status": "done"})

    # 3-5. Audio: extraction (CPU/ffmpeg) → VAD → transcription. A video with
    # no audio track skips straight past these with empty results, exactly as
    # the original CLI did.
    speech_regions: list = []
    transcript_segments: list = []
    logger.info("stage 3-5/7 audio: extract -> VAD -> transcribe")
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
            "stage 3-5/7 audio done in %.1fs: %d speech region(s), %d dialogue segment(s)",
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
            "stage 3-5/7 audio: no audio stream, skipping VAD + transcription"
        )
        await on_event(
            {"type": "stage", "stage": "audio", "status": "done", "has_audio": False}
        )

    # 6. Timeline assembly (CPU-bound) then narration generation (Gemini).
    logger.info("stage 6/7 timeline: assembling segments")
    await on_event({"type": "stage", "stage": "timeline", "status": "start"})
    timeline = await asyncio.to_thread(
        build_timeline,
        str(video_path),
        shots,
        speech_regions,
        transcript_segments,
    )
    eligible = sum(1 for s in timeline.segments if s.ad_eligible)
    logger.info(
        "stage 6/7 timeline done: %d segment(s), %d AD-eligible",
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
        "stage 6/7 narration: writing AD lines for %d eligible segment(s)", eligible
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
        timeline, on_segment=_on_segment, client=client
    )
    logger.info("stage 6/7 narration done in %.1fs", time.monotonic() - t0)
    await on_event({"type": "stage", "stage": "narration", "status": "done"})

    # 7. Narration text-to-speech (CPU-bound Kokoro, one clip at a time).
    logger.info("stage 7/7 tts: synthesizing narration audio")
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
                "audio": Path(segment.ad_narration_audio).name,
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
        out_dir=str(narration_dir),
        on_segment=_on_narration_audio,
        retry_optimize=retry_optimize,
    )
    logger.info("stage 7/7 tts done in %.1fs", time.monotonic() - t0)

    # Assemble every narration clip into one video-length "AD-only" track, each
    # clip placed where it would play alongside the source.
    result = await asyncio.to_thread(build_ad_track, timeline, str(narration_dir))
    if result is not None:
        ad_track_path, ad_track_duration = result
        timeline.ad_track_audio = ad_track_path
        timeline.ad_track_duration_sec = ad_track_duration
        await on_event(
            {
                "type": "ad_track",
                "audio": Path(ad_track_path).name,
                "duration_sec": ad_track_duration,
            }
        )

    logger.info("pipeline complete: %s", video_path)

    return timeline
