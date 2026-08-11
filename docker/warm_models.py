"""Download every model weight into the image at build time.

These three were fetched lazily on first use, inside the request path, so the
first job after a cold start waited on roughly 1.2GB of downloads before it
could begin. Baking them in trades image size — which Fly pays once per deploy —
for latency, which every cold start would otherwise pay.

Run from docker/Dockerfile.media. Each download is independent; a failure here
should fail the build rather than silently produce an image that stalls on its
first job, so nothing is caught.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("warm_models")


def warm_whisper() -> None:
    """faster-whisper `small`, int8 on CPU (~250MB). See transcription.py."""
    from faster_whisper import WhisperModel

    logger.info("downloading faster-whisper small…")
    WhisperModel("small", device="cpu", compute_type="int8")


def warm_kokoro() -> None:
    """Kokoro-82M plus the af_heart voice pack (~330MB). See tts.py."""
    from kokoro import KPipeline

    logger.info("downloading Kokoro-82M…")
    pipeline = KPipeline(lang_code="a")
    # Loading the pipeline does not fetch the voice; synthesizing one short
    # line does, and also proves espeak-ng is present for the G2P fallback.
    for _ in pipeline("Warming up.", voice="af_heart", speed=1.0):
        break


def warm_clip() -> None:
    """open_clip ViT-B-32 / laion2b_s34b_b79k (~600MB). See qa/retriever.py."""
    import open_clip

    logger.info("downloading CLIP ViT-B-32…")
    open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    open_clip.get_tokenizer("ViT-B-32")


def warm_silero() -> None:
    """Silero VAD (~2MB, but it is one more first-run download). See voice_activity.py."""
    from silero_vad import load_silero_vad

    logger.info("downloading Silero VAD…")
    load_silero_vad()


if __name__ == "__main__":
    warm_whisper()
    warm_kokoro()
    warm_clip()
    warm_silero()
    logger.info("all model weights baked into the image")
