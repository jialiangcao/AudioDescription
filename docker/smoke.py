"""Exercise every stage of the media image that can only fail at runtime.

CI proves the image *builds*, and `warm_models.py` proves the weights download.
Neither proves the image *runs*: the stages here reach native code — OpenCV,
ffmpeg, libsndfile, espeak-ng, torch's shared libraries — through paths that a
`docker build` never touches. The bug that motivated this script was exactly
that shape: `silero_vad.read_audio` went through torchaudio into TorchCodec,
which wanted FFmpeg's shared libraries rather than the `ffmpeg` binary this
image installs, so it imported fine and blew up in stage 4 of a real job.

Run from docker/Dockerfile.media as the final build layer, so a break fails the
build instead of a deploy. It is also the fastest way to check a suspect image
by hand — no build, no deploy:

    fly ssh console -a adesc-worker-media -C "python /app/docker/smoke.py"
    docker run --rm -v "$PWD/src:/app/src:ro" adesc-media python docker/smoke.py

Nothing is caught: any failure here means the image cannot do its job.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("smoke")

# Long enough for PySceneDetect to see a clip and Whisper to have something to
# chew on, short enough that the whole script stays a few seconds of build time.
CLIP_SECONDS = 3


def require_binaries() -> None:
    """ffmpeg/ffprobe (audio extraction, mux) and espeak-ng (Kokoro's G2P)."""
    for binary in ("ffmpeg", "ffprobe", "espeak-ng"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"{binary} is not on PATH")
    logger.info("binaries ok: ffmpeg, ffprobe, espeak-ng")


def make_clip(path: Path) -> None:
    """A tiny synthetic video with an audio track, via ffmpeg's lavfi sources."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={CLIP_SECONDS}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={CLIP_SECONDS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )  # fmt: skip


def main() -> None:
    require_binaries()

    import soundfile as sf

    from audio_extract import extract_audio
    from blobs import JobBlobs
    from mux import mux_described_video
    from segmentation import probe_video, segment_video
    from transcription import transcribe
    from tts import SAMPLE_RATE, _synthesize
    from voice_activity import detect_speech_regions

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        video = work / "source.mp4"
        make_clip(video)

        # Stage 1: OpenCV + PySceneDetect, writing frames through JobBlobs.
        probe = probe_video(str(video))
        if probe["duration_sec"] <= 0:
            raise RuntimeError(f"OpenCV could not decode the clip: {probe}")
        blobs = JobBlobs("smoke", root=work / "blobs", store=None)
        shots = segment_video(str(video), blobs)
        frames = [frame for shot in shots for frame in shot["frames"]]
        if not frames:
            raise RuntimeError("segmentation produced no frames")
        if not blobs.path(frames[0]["key"]).exists():
            raise RuntimeError(f"frame {frames[0]['key']} was not written")
        logger.info(
            "segmentation ok: %.1ffps, %.1fs, %d shot(s), %d frame(s)",
            probe["fps"], probe["duration_sec"], len(shots), len(frames),
        )  # fmt: skip

        # Stage 3: ffmpeg shells out for the 16kHz mono WAV.
        audio = work / "audio.wav"
        extract_audio(str(video), out_path=str(audio))
        logger.info("audio extraction ok: %s", audio.name)

        # Stage 4: Silero VAD. Reads the WAV through libsndfile — the step that
        # used to drag in torchaudio/TorchCodec. A tone yields no speech, and
        # that is a fine result: what matters is that it returns at all.
        regions = detect_speech_regions(str(audio))
        logger.info("vad ok: %d region(s)", len(regions))

        # Stage 5: faster-whisper over the same track.
        segments = transcribe(str(audio), regions)
        logger.info("transcription ok: %d segment(s)", len(segments))

        # Stage 7: Kokoro, which needs espeak-ng for its G2P fallback.
        clip = _synthesize("The smoke test speaks one short line.")
        if len(clip) == 0:
            raise RuntimeError("Kokoro synthesized an empty clip")
        ad_track = work / "ad_track.wav"
        sf.write(str(ad_track), clip, SAMPLE_RATE)
        logger.info("tts ok: %.2fs of audio", len(clip) / SAMPLE_RATE)

        # Stage 8: the mux, including its sidechaincompress ducking filter.
        described = work / "described.mp4"
        mux_described_video(video, ad_track, described)
        if described.stat().st_size == 0:
            raise RuntimeError("mux produced an empty file")
        logger.info("mux ok: %d bytes", described.stat().st_size)

    logger.info("media image smoke test passed")


if __name__ == "__main__":
    main()
