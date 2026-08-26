"""``bench fetch`` — pull the CMD clips named by the CSV down from YouTube.

Each ``cmd_filename`` is downloaded once, however many AD rows point at it, and
a clip already on disk is left alone, so this is safe to re-run.

Coverage is reported rather than assumed. CMD's uploads date to around 2020 and
a real share of them are now deleted, private or geo-blocked; without a count,
those clips would just quietly shrink the eval set and make two runs
incomparable for a reason nothing in the output mentions.
"""

import json
import logging
from pathlib import Path

from bench.dataset import group_by_clip, load_rows, spread_across_movies, watch_url
from ingest import IngestError, download_youtube

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"


def clip_path(clips_dir: Path, video_id: str) -> Path:
    return Path(clips_dir) / f"{video_id}.mp4"


def fetch_clips(
    csv_path: Path,
    clips_dir: Path,
    split: str | None = None,
    limit: int | None = None,
    max_height: int = 720,
) -> dict:
    """Download up to ``limit`` distinct clips. Returns a coverage summary."""
    clips = group_by_clip(load_rows(csv_path, split=split))
    # Widest movie coverage first, so --limit never means "the first two movies".
    ordered = spread_across_movies(clips)
    video_ids = ordered[:limit] if limit else ordered
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest = clips_dir / MANIFEST_NAME

    ok, failed, skipped = 0, 0, 0
    with open(manifest, "a", encoding="utf-8") as log:
        for i, video_id in enumerate(video_ids, 1):
            dest = clip_path(clips_dir, video_id)
            if dest.exists():
                skipped += 1
                continue
            url = watch_url(video_id)
            logger.info("fetch %d/%d: %s", i, len(video_ids), url)
            record = {"video_id": video_id, "url": url, "ad_rows": len(clips[video_id])}
            try:
                meta = download_youtube(url, dest, max_height=max_height)
            except IngestError as exc:
                failed += 1
                record |= {"status": "failed", "reason": str(exc)}
                logger.warning("fetch: %s unavailable (%s)", video_id, exc)
            else:
                ok += 1
                record |= {"status": "ok", **meta}
            log.write(json.dumps(record) + "\n")
            log.flush()

    summary = {
        "requested": len(video_ids),
        "downloaded": ok,
        "already_present": skipped,
        "failed": failed,
        "clips_dir": str(clips_dir),
    }
    available = ok + skipped
    logger.info(
        "fetch: %d/%d clip(s) available (%d new, %d already here, %d unavailable)",
        available,
        len(video_ids),
        ok,
        skipped,
        failed,
    )
    if failed:
        logger.warning(
            "fetch: %.0f%% of requested clips could not be downloaded — the eval "
            "set is smaller than the CSV suggests",
            100 * failed / len(video_ids) if video_ids else 0,
        )
    return summary
