"""Described-video mux (stage 8).

Takes the source video plus the combined AD track from ``ad_track.py`` and writes
a single playable file whose audio is the original soundtrack with the narration
mixed in — so a viewer watches the video once and hears both, instead of playing
an AD-only track alongside a muted picture.

The original audio is *ducked* under the narration rather than mixed flat: an
``sidechaincompress`` filter keyed off the AD track pulls the soundtrack down
while a line is spoken and lets it back up afterwards, which is how broadcast AD
mixes are made. Narration is placed in speech-free gaps by earlier stages, so in
practice the ducking only bites into music and ambience, not dialogue.

The video stream is stream-copied when the container allows it (the common case:
an mp4/mov upload), and re-encoded to H.264 only if the copy is rejected — e.g. a
source codec that mp4 can't hold.
"""

import logging
import os
import subprocess

from audio_extract import has_audio_stream

logger = logging.getLogger(__name__)

# Job-relative blob key of the muxed result; also its filename in scratch.
DESCRIBED_KEY = "described.mp4"

# Output audio format. 48kHz stereo AAC is the safe common denominator for
# browser <video> playback; both inputs are resampled into it before mixing.
SAMPLE_RATE = 48000
AUDIO_BITRATE = "192k"

# Ducking envelope, keyed off the narration track. The threshold is a linear
# amplitude (~-26dBFS): narration above it pulls the soundtrack down, easing back
# out over RELEASE ms so the original doesn't snap up between words. Measured on
# a typical narration clip this ratio works out to roughly a 12dB duck, which is
# the usual broadcast-AD depth — loud enough to hear over, quiet enough to follow.
DUCK_THRESHOLD = 0.05
DUCK_RATIO = 3
DUCK_ATTACK_MS = 20
DUCK_RELEASE_MS = 400

# Gain on the narration going into the mix, so it sits clearly above the ducked
# bed. Kokoro's output is quiet — real narration tracks peak around 0.37 — so 2x
# (+6dB) still leaves headroom, and the limiter below catches anything hotter.
# Note this is applied to the mix branch only, *not* to the sidechain key: the
# duck depth is deliberately independent of how loud the narration is played.
NARRATION_GAIN = 2.0

# Safety net for a hot source: only engages above this level, so a normal mix
# passes through untouched. ``level=disabled`` keeps it from auto-normalizing
# (which would drag quiet passages up).
LIMIT = 0.95

_AFORMAT = f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"

# Duck the source audio under the narration, then mix the narration back on top.
# ``apad`` extends the AD track indefinitely so it can never run out from under
# the sidechain (which would cut the soundtrack short at the last spoken line);
# ``amix=duration=first`` then trims the result back to the source's length.
_DUCK_FILTER = (
    f"[0:a]{_AFORMAT}[orig];"
    f"[1:a]{_AFORMAT},apad,asplit=2[ad_pre][ad_key];"
    f"[ad_pre]volume={NARRATION_GAIN}[ad_mix];"
    f"[orig][ad_key]sidechaincompress="
    f"threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}"
    f":attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}[ducked];"
    f"[ducked][ad_mix]amix=inputs=2:duration=first:normalize=0,"
    f"alimiter=limit={LIMIT}:level=disabled[aout]"
)

# Nothing to duck or mix against: the narration *is* the soundtrack, at the same
# playback gain it would get in a full mix.
_AD_ONLY_FILTER = (
    f"[1:a]{_AFORMAT},volume={NARRATION_GAIN},"
    f"alimiter=limit={LIMIT}:level=disabled[aout]"
)


def _ffmpeg_args(video_path, ad_track_path, out_path, source_has_audio, copy_video):
    """Build the ffmpeg argv for one mux attempt.

    ``-nostdin`` matters here: this runs on a worker thread with whatever stdin
    the server inherited, and ffmpeg's interactive key handling has been seen to
    block at exit rather than finish the mux.
    """
    args = ["ffmpeg", "-nostdin", "-y", "-i", video_path, "-i", ad_track_path]

    filters = _DUCK_FILTER if source_has_audio else _AD_ONLY_FILTER
    args += ["-filter_complex", filters, "-map", "0:v:0", "-map", "[aout]"]
    if not source_has_audio:
        args += ["-shortest"]

    args += ["-c:v", "copy" if copy_video else "libx264"]
    if not copy_video:
        args += ["-preset", "veryfast", "-crf", "20"]
    args += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart", out_path]
    return args


def mux_described_video(video_path, ad_track_path, out_path=DESCRIBED_KEY) -> str:
    """Mux ``ad_track_path`` into ``video_path``, writing a described video.

    All three arguments are real local paths — ffmpeg needs files on disk, so
    resolving blob keys to scratch is the caller's job. Returns the output path.
    Raises ``subprocess.CalledProcessError`` if ffmpeg fails even after falling
    back to re-encoding the video stream, and ``FileNotFoundError`` if the AD
    track is missing.
    """
    video_path, ad_track_path, out_path = (
        str(video_path),
        str(ad_track_path),
        str(out_path),
    )
    if not os.path.isfile(ad_track_path):
        raise FileNotFoundError(f"AD track not found: {ad_track_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    source_has_audio = has_audio_stream(video_path)
    logger.debug(
        "mux_described_video: %s + %s -> %s (source_has_audio=%s)",
        video_path,
        ad_track_path,
        out_path,
        source_has_audio,
    )

    for copy_video in (True, False):
        args = _ffmpeg_args(
            video_path, ad_track_path, out_path, source_has_audio, copy_video
        )
        result = subprocess.run(
            args, capture_output=True, text=True, stdin=subprocess.DEVNULL
        )
        if result.returncode == 0:
            logger.info(
                "mux_described_video: wrote %s (video %s)",
                out_path,
                "copied" if copy_video else "re-encoded",
            )
            return out_path
        if copy_video:
            logger.warning(
                "mux_described_video: stream copy failed for %s, re-encoding video: %s",
                video_path,
                result.stderr.strip().splitlines()[-1:] or "",
            )
        else:
            logger.error("mux_described_video: ffmpeg failed: %s", result.stderr)
            raise subprocess.CalledProcessError(
                result.returncode, args, result.stdout, result.stderr
            )

    raise AssertionError("unreachable")  # pragma: no cover
