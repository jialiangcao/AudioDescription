import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Job-relative blob key of the extracted 16kHz mono track.
AUDIO_KEY = "audio.wav"


class NoAudioStreamError(Exception):
    pass


def has_audio_stream(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    logger.debug("has_audio_stream: %s -> %d audio stream(s)", video_path, len(streams))
    return len(streams) > 0


def extract_audio(video_path, out_path="audio.wav", sample_rate=16000):
    """Extract a mono wav track from video_path via ffmpeg.

    16kHz mono is the standard input format for Whisper and Silero VAD.
    """
    if not has_audio_stream(video_path):
        logger.info("extract_audio: %s has no audio stream", video_path)
        raise NoAudioStreamError(f"{video_path} has no audio stream")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    logger.debug(
        "extract_audio: %s -> %s (%dHz mono)", video_path, out_path, sample_rate
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-acodec",
                "pcm_s16le",
                out_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("extract_audio: ffmpeg failed (%s): %s", video_path, exc.stderr)
        raise
    return out_path
