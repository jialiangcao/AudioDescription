import json

import cv2
from pydantic import BaseModel


class VisualAnalysis(BaseModel):
    description: str
    entities: list[str]
    setting: str
    on_screen_text: str | None


class SoundTag(BaseModel):
    label: str
    confidence: float


class AudioAnalysis(BaseModel):
    has_speech: bool
    transcript: str | None
    sound_tags: list[SoundTag]
    music: bool
    silence_ratio: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    keyframe: str
    visual: VisualAnalysis
    # Populated once the audio pipeline exists (VAD/Whisper/sound tagging).
    audio: AudioAnalysis | None = None
    # Derived from `audio` once it's populated — left null until then.
    ad_eligible: bool | None = None
    narratable_gap_sec: float | None = None


class Timeline(BaseModel):
    video_id: str
    duration_sec: float
    segments: list[Segment]


def build_timeline(video_path, shots):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if fps else 0.0
    cap.release()

    segments = [
        Segment(
            id=shot["id"],
            start=shot["start"],
            end=shot["end"],
            keyframe=shot["keyframe"],
            visual=VisualAnalysis(**shot["visual"]),
        )
        for shot in shots
    ]

    return Timeline(video_id=video_path, duration_sec=duration_sec, segments=segments)


def save_timeline(timeline, out_path="timeline.json"):
    with open(out_path, "w") as f:
        f.write(timeline.model_dump_json(indent=2))
    return out_path


def load_timeline(path="timeline.json"):
    with open(path) as f:
        return Timeline.model_validate_json(f.read())