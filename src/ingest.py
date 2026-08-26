"""Fetching a source video from a URL (stage 0).

The browser normally PUTs the source straight to R2 and the pipeline starts at
segmentation. This module is the other way in: given a YouTube URL, a worker
downloads the video itself. Two callers share it — ``tasks.fetch_source`` (the
``POST /api/jobs/from-url`` route) and the offline benchmark harness, which
pulls Condensed Movies clips by their video id.

``yt-dlp`` lives in the ``media`` optional extra and is imported **lazily**, for
the same reason ``segmentation`` and ``transcription`` defer theirs: ``tasks.py``
must stay importable in the slim image, which is the premise of the two-image
split (CI asserts it).
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Hosts we are willing to fetch from. Deliberately a allowlist, not a blocklist:
# this runs on a worker with network access, so an arbitrary URL is an SSRF
# surface as much as it is a media source.
ALLOWED_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)

# Cap the download: 720p is well past what per-frame vision analysis needs, and
# the ceiling keeps one long 4K video from filling a worker's scratch volume.
DEFAULT_MAX_HEIGHT = 720

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class IngestError(Exception):
    """A URL could not be fetched — unavailable, private, geo-blocked, rejected."""


def is_supported_url(url: str) -> bool:
    try:
        return (urlparse(url).hostname or "").lower() in ALLOWED_HOSTS
    except ValueError:
        return False


def video_id(url_or_id: str) -> str | None:
    """The 11-character YouTube id in ``url_or_id``, or None.

    Accepts a bare id, a ``watch?v=`` URL, or a ``youtu.be/`` short link. The
    benchmark's CSV addresses clips as ``2011/_SQr8I3lcW8``, so a bare id is the
    common case there.
    """
    if _VIDEO_ID_RE.match(url_or_id):
        return url_or_id
    parsed = urlparse(url_or_id)
    if (parsed.hostname or "").lower() == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    else:
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]
    return candidate if _VIDEO_ID_RE.match(candidate) else None


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _base_opts() -> dict[str, Any]:
    """yt-dlp options shared by probe and download.

    The cookie hooks matter in production, not locally: YouTube's bot check
    fires on datacenter IPs (Fly) long before it fires on a residential one, and
    the only fix is to hand yt-dlp a real session.
    """
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "noprogress": True,
        # yt-dlp logs through its own logger by default; route it at ours so
        # ADESC_LOG_LEVEL governs it like everything else.
        "logger": logging.getLogger("yt_dlp"),
    }
    browser = os.environ.get("ADESC_YTDLP_COOKIES_FROM_BROWSER")
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    cookiefile = os.environ.get("ADESC_YTDLP_COOKIEFILE")
    if cookiefile:
        opts["cookiefile"] = cookiefile
    return opts


def _info_dict(info) -> dict:
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "duration_sec": float(info["duration"]) if info.get("duration") else None,
        "extractor": info.get("extractor"),
    }


def probe_youtube(url: str) -> dict:
    """Metadata for ``url`` without downloading it: ``id``, ``title``, ``duration_sec``.

    Lets a caller reject a video on length *before* paying for the bytes — the
    upload path checks size at ``/start``, but a URL has no size until it has
    been fetched.
    """
    if not is_supported_url(url):
        raise IngestError(f"unsupported URL: {url}")

    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    try:
        # yt-dlp types its options as a large private TypedDict, but ours is
        # assembled conditionally and is a plain dict by construction.
        with YoutubeDL(cast(Any, _base_opts())) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise IngestError(f"could not read {url}: {exc}") from exc
    if info is None:
        raise IngestError(f"could not read {url}")
    return _info_dict(info)


def download_youtube(
    url: str, dest: Path | str, *, max_height: int = DEFAULT_MAX_HEIGHT
) -> dict:
    """Download ``url`` to exactly ``dest``. Returns the same dict as ``probe_youtube``.

    The format string prefers a separate video+audio pair (which yt-dlp muxes
    into mp4 with ffmpeg) and falls back to a pre-muxed stream, so a video with
    no adaptive rendition at ``max_height`` still downloads rather than failing.
    """
    if not is_supported_url(url):
        raise IngestError(f"unsupported URL: {url}")

    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    opts = _base_opts() | {
        "format": (f"bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b"),
        "merge_output_format": "mp4",
        # A fixed template, not a pattern: callers address the file by blob key
        # and must not have to guess what yt-dlp named it.
        "outtmpl": str(dest.with_suffix("")) + ".%(ext)s",
    }

    logger.info("ingest: downloading %s -> %s", url, dest)
    try:
        with YoutubeDL(cast(Any, opts)) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        raise IngestError(f"could not download {url}: {exc}") from exc
    if info is None:
        raise IngestError(f"could not download {url}")

    if not dest.exists():
        # merge_output_format is a request, not a guarantee — a stream yt-dlp
        # cannot remux lands under its own extension. Move it into place so the
        # caller's key is still correct.
        produced = sorted(dest.parent.glob(dest.stem + ".*"))
        if not produced:
            raise IngestError(f"{url} downloaded but produced no file at {dest}")
        produced[0].rename(dest)
        logger.debug("ingest: renamed %s -> %s", produced[0].name, dest.name)

    meta = _info_dict(info)
    logger.info(
        "ingest: %s -> %s (%.1fMB, %s)",
        meta["id"],
        dest.name,
        dest.stat().st_size / (1024 * 1024),
        meta["title"],
    )
    return meta
