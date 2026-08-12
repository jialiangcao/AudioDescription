"""Intermediate state handed between pipeline stages.

``run_pipeline`` could keep the shot list, the speech regions and the transcript
in local variables because it was one function in one process. As Celery tasks
the stages run on different machines, so whatever one stage produces and a later
one consumes has to be written down.

It lives in the job's blob store as JSON rather than in Postgres: it is bulky,
write-once, read-once, and already job-scoped, so it rides the same R2 lifecycle
rule that expires the frames.

Vision results are stored one blob per *shot*, not one per job. The vision stage
fans out across workers, and a shot-sized blob is the largest unit a single task
owns end to end — so no two tasks ever write the same key.
"""

import json
import logging

logger = logging.getLogger(__name__)

SHOTS_KEY = "state/shots.json"
AUDIO_KEY = "state/audio.json"


def vision_key(shot_id: int) -> str:
    return f"state/vision/shot_{shot_id:04d}.json"


def _write(blobs, key: str, doc) -> str:
    blobs.path(key).write_text(json.dumps(doc))
    blobs.put(key)
    return key


def _read(blobs, key: str):
    return json.loads(blobs.fetch(key).read_text())


# --- segmentation -----------------------------------------------------------


def save_shots(blobs, shots: list[dict], meta: dict) -> None:
    """The shot/frame skeleton plus the video's probed fps/duration."""
    _write(blobs, SHOTS_KEY, {"shots": shots, "meta": meta})


def load_shots(blobs) -> tuple[list[dict], dict]:
    doc = _read(blobs, SHOTS_KEY)
    return doc["shots"], doc["meta"]


# --- vision -----------------------------------------------------------------


def save_shot_vision(blobs, shot: dict) -> None:
    """One shot's frames, each carrying the ``visual`` analysis just produced."""
    _write(blobs, vision_key(shot["id"]), shot["frames"])


def load_vision_into(blobs, shots: list[dict]) -> list[dict]:
    """Attach each shot's stored frame analyses back onto the skeleton.

    A shot whose vision blob is missing keeps its unanalyzed frames rather than
    failing the job: ``visual`` is already Optional everywhere downstream, and
    losing one shot's descriptions is a far better outcome than losing the run.
    """
    for shot in shots:
        try:
            shot["frames"] = _read(blobs, vision_key(shot["id"]))
        except (FileNotFoundError, OSError, ValueError):
            logger.warning(
                "job %s: no vision results for shot %s, leaving it unanalyzed",
                blobs.job_id,
                shot["id"],
            )
    return shots


# --- audio ------------------------------------------------------------------


def save_audio(blobs, has_audio: bool, speech_regions, transcript_segments) -> None:
    _write(
        blobs,
        AUDIO_KEY,
        {
            "has_audio": has_audio,
            # JSON has no tuples; VAD regions come back as [start, end] lists
            # and every consumer unpacks them positionally.
            "speech_regions": [list(region) for region in speech_regions],
            "transcript_segments": transcript_segments,
        },
    )


def load_audio(blobs) -> tuple[bool, list, list]:
    doc = _read(blobs, AUDIO_KEY)
    regions = [tuple(region) for region in doc["speech_regions"]]
    return doc["has_audio"], regions, doc["transcript_segments"]
