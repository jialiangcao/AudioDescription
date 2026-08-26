"""``bench run`` — the pipeline over the downloaded clips, one prediction per AD row.

Calls the stage functions directly, the way ``pipeline.run_pipeline`` does,
against a ``JobBlobs`` with ``store=None``. That is what keeps the harness free
of Postgres, Redis and object storage: every artifact stays in a local scratch
directory, and uploads become no-ops.

Each clip's expensive intermediates go through the same ``job_state`` helpers
the Celery tasks use, so they land in that scratch directory and a second run
reuses them. This matters more than it looks: the whole point of the harness is
to iterate on the *narration* prompt, and without the cache every iteration would
re-pay for one Gemini call per sampled frame.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import job_state
from audio_extract import AUDIO_KEY, NoAudioStreamError, extract_audio
from bench.anchored import build_anchored_timeline
from bench.config import FRAME_PAD_SEC, INTERVAL_SEC
from bench.dataset import AdRow, group_by_clip, load_rows, spread_across_movies
from bench.fetch import clip_path
from blobs import JobBlobs
from segmentation import probe_video, segment_video
from timeline import NARRATION_WORDS_PER_SEC
from transcription import transcribe
from vision_analysis import fill_narration_gaps
from voice_activity import detect_speech_regions

logger = logging.getLogger(__name__)


def clip_blobs(video_id: str, work_dir: Path) -> JobBlobs:
    """A local-only ``JobBlobs`` for one clip, in its **own** scratch directory.

    ``JobBlobs`` takes ``root`` as the directory itself — the job id namespaces
    the *bucket* key, not the local path — so each clip has to be handed a
    distinct root. Sharing one directory across clips made every clip after the
    first "reuse" the first clip's cached shots, frames and vision results, and
    silently describe the wrong video.

    ``store=None``: purely local, no bucket, no credentials.
    """
    return JobBlobs(video_id, root=Path(work_dir) / video_id, store=None)


def _prepare_clip(video_path: Path, blobs: JobBlobs, interval_sec: float):
    """Stages 1 and 3-5 for one clip: shots + frames, then speech + transcript.

    Both halves are cached, and both are pure CPU — no Gemini call happens here.
    """
    try:
        shots, meta = job_state.load_shots(blobs)
        logger.info("run: %s reusing cached shots", blobs.job_id)
    except (FileNotFoundError, OSError, ValueError):
        meta = probe_video(str(video_path))
        shots = segment_video(str(video_path), blobs, interval_sec=interval_sec)
        job_state.save_shots(blobs, shots, meta)

    try:
        _, speech_regions, transcript_segments = job_state.load_audio(blobs)
        logger.info("run: %s reusing cached audio", blobs.job_id)
    except (FileNotFoundError, OSError, ValueError):
        try:
            extract_audio(str(video_path), out_path=str(blobs.path(AUDIO_KEY)))
            speech_regions = detect_speech_regions(str(blobs.path(AUDIO_KEY)))
            transcript_segments = transcribe(str(blobs.path(AUDIO_KEY)), speech_regions)
            job_state.save_audio(blobs, True, speech_regions, transcript_segments)
        except NoAudioStreamError:
            logger.warning("run: %s has no audio stream", blobs.job_id)
            speech_regions, transcript_segments = [], []
            job_state.save_audio(blobs, False, [], [])

    return shots, meta, speech_regions, transcript_segments


async def _describe_frames(shots: list[dict], blobs: JobBlobs, client) -> list[dict]:
    """Stage 2, cached per shot. The expensive one: one Gemini call per frame."""
    from vision_analysis import analyze_shots

    cached = job_state.load_vision_into(blobs, [dict(shot) for shot in shots])
    if all(frame.get("visual") for shot in cached for frame in shot["frames"]):
        logger.info("run: %s reusing cached frame analyses", blobs.job_id)
        return cached

    analyzed = await analyze_shots(shots, blobs, client=client)
    for shot in analyzed:
        job_state.save_shot_vision(blobs, shot)
    return analyzed


async def run_clip(
    video_id: str,
    rows: list[AdRow],
    clips_dir: Path,
    work_dir: Path,
    interval_sec: float = INTERVAL_SEC,
    pad_sec: float = FRAME_PAD_SEC,
    words_per_sec: float | None = None,
    client=None,
) -> list[dict]:
    """Predict one AD line per row of ``rows``. Returns the prediction records."""
    video_path = clip_path(clips_dir, video_id)
    if not video_path.exists():
        raise FileNotFoundError(f"clip {video_id} not downloaded ({video_path})")

    blobs = clip_blobs(video_id, work_dir)
    t0 = time.monotonic()

    shots, meta, speech_regions, transcript_segments = await asyncio.to_thread(
        _prepare_clip, video_path, blobs, interval_sec
    )
    shots = await _describe_frames(shots, blobs, client)

    timeline = build_anchored_timeline(
        video_id,
        meta["duration_sec"],
        shots,
        rows,
        speech_regions,
        transcript_segments,
        pad_sec=pad_sec,
    )
    await fill_narration_gaps(
        timeline, blobs, words_per_sec=words_per_sec, client=client
    )

    ordered = sorted(rows, key=lambda r: r.scaled_start)
    records = [
        {
            "video_id": video_id,
            "cmd_filename": row.cmd_filename,
            "imdbid": row.imdbid,
            "movie_title": row.movie_title,
            "scaled_start": row.scaled_start,
            "scaled_end": row.scaled_end,
            "duration": row.duration,
            "frames": len(segment.frames),
            "words_per_sec": words_per_sec or NARRATION_WORDS_PER_SEC,
            "gt": row.text,
            "pred": segment.ad_narration,
        }
        for row, segment in zip(ordered, timeline.segments, strict=True)
    ]
    logger.info(
        "run: %s -> %d prediction(s) in %.1fs",
        video_id,
        len(records),
        time.monotonic() - t0,
    )
    return records


def _already_done(preds_path: Path) -> set[str]:
    if not preds_path.exists():
        return set()
    done = set()
    with open(preds_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["video_id"])
    return done


async def run_benchmark(
    csv_path: Path,
    clips_dir: Path,
    work_dir: Path,
    preds_path: Path,
    split: str | None = None,
    limit: int | None = None,
    interval_sec: float = INTERVAL_SEC,
    pad_sec: float = FRAME_PAD_SEC,
    words_per_sec: float | None = None,
    force: bool = False,
) -> dict:
    """Run every downloaded clip, appending predictions to ``preds_path``."""
    from google import genai

    clips = group_by_clip(load_rows(csv_path, split=split))
    preds_path = Path(preds_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    if force and preds_path.exists():
        preds_path.unlink()
    done = _already_done(preds_path)

    ordered = spread_across_movies(clips)
    pending = [
        video_id
        for video_id in ordered
        if clip_path(clips_dir, video_id).exists() and video_id not in done
    ]
    missing = [v for v in ordered if not clip_path(clips_dir, video_id=v).exists()]
    if limit:
        pending = pending[:limit]
    if missing:
        logger.warning(
            "run: %d clip(s) in the CSV are not downloaded; run `bench fetch` first",
            len(missing),
        )

    client = genai.Client()
    written, failed = 0, 0
    with open(preds_path, "a", encoding="utf-8") as out:
        for i, video_id in enumerate(pending, 1):
            logger.info("run %d/%d: %s", i, len(pending), video_id)
            try:
                records = await run_clip(
                    video_id,
                    clips[video_id],
                    clips_dir,
                    work_dir,
                    interval_sec=interval_sec,
                    pad_sec=pad_sec,
                    words_per_sec=words_per_sec,
                    client=client,
                )
            except Exception:
                # One bad clip (a broken download, a Gemini refusal) must not
                # cost the whole run — everything before it is already on disk.
                failed += 1
                logger.exception("run: clip %s failed, continuing", video_id)
                continue
            for record in records:
                out.write(json.dumps(record) + "\n")
                written += 1
            out.flush()

    summary = {
        "clips_run": len(pending) - failed,
        "clips_failed": failed,
        "clips_skipped": len(done),
        "predictions": written,
        "preds_path": str(preds_path),
    }
    logger.info(
        "run: %d prediction(s) over %d clip(s) -> %s",
        written,
        len(pending) - failed,
        preds_path,
    )
    return summary
