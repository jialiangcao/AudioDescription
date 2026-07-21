"""Stage orchestration for the audio-description pipeline.

Extracted from the old ``main.py::process_video`` CLI. Instead of printing to
the console and reading/writing fixed ``src/in``/``src/out`` paths, this drives
the same six stages against explicit job-scoped paths and reports progress via
an async ``on_event`` callback, so the web layer can stream results live.

The function is intentionally ignorant of FastAPI/JobStore — it just awaits a
callback — which keeps it unit-testable in isolation.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from audio_extract import NoAudioStreamError, extract_audio
from segmentation import segment_video
from timeline import Timeline, build_timeline
from transcription import transcribe
from vision_analysis import analyze_shots, fill_narration_gaps
from voice_activity import detect_speech_regions

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
    assembled timeline before narration), and ``narration`` (per-segment AD
    text). Shot and narration events arrive out of order — consumers must key
    off the ``id`` field.
    """
    video_path = Path(video_path)
    frames_dir = job_dir / "frames"
    audio_path = job_dir / "audio.wav"

    # 1. Shot segmentation (CPU-bound: PySceneDetect + OpenCV).
    await on_event({"type": "stage", "stage": "segmentation", "status": "start"})
    shots = await asyncio.to_thread(
        segment_video, str(video_path), out_dir=str(frames_dir)
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
    await on_event({"type": "stage", "stage": "vision", "status": "start"})

    async def _on_shot(shot: dict) -> None:
        await on_event(_shot_event(shot))

    shots = await analyze_shots(shots, on_shot=_on_shot, client=client)
    await on_event({"type": "stage", "stage": "vision", "status": "done"})

    # 3-5. Audio: extraction (CPU/ffmpeg) → VAD → transcription. A video with
    # no audio track skips straight past these with empty results, exactly as
    # the original CLI did.
    speech_regions: list = []
    transcript_segments: list = []
    await on_event({"type": "stage", "stage": "audio", "status": "start"})
    try:
        await asyncio.to_thread(
            extract_audio, str(video_path), out_path=str(audio_path)
        )
        speech_regions = await asyncio.to_thread(detect_speech_regions, str(audio_path))
        transcript_segments = await asyncio.to_thread(
            transcribe, str(audio_path), speech_regions
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
        await on_event(
            {"type": "stage", "stage": "audio", "status": "done", "has_audio": False}
        )

    # 6. Timeline assembly (CPU-bound) then narration generation (Gemini).
    await on_event({"type": "stage", "stage": "timeline", "status": "start"})
    timeline = await asyncio.to_thread(
        build_timeline,
        str(video_path),
        shots,
        speech_regions,
        transcript_segments,
    )
    await on_event(
        {
            "type": "timeline",
            "timeline": timeline.model_dump(),
        }
    )
    await on_event({"type": "stage", "stage": "timeline", "status": "done"})

    await on_event({"type": "stage", "stage": "narration", "status": "start"})

    async def _on_segment(segment) -> None:
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
    await on_event({"type": "stage", "stage": "narration", "status": "done"})

    return timeline
