from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from timeline import NARRATION_WORDS_PER_SEC

load_dotenv()

MODEL = "gemini-2.5-flash"


class ShotAnalysis(BaseModel):
    description: str
    entities: list[str]
    setting: str
    on_screen_text: str | None


def analyze_keyframe(client, keyframe_path):
    with open(keyframe_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            "Describe what is visually happening in this video frame.",
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=ShotAnalysis,
        ),
    )

    if response.text is None:
        raise RuntimeError(f"{MODEL} returned no text for {keyframe_path}")
    return ShotAnalysis.model_validate_json(response.text).model_dump()


def analyze_shots(shots):
    client = genai.Client()
    for shot in shots:
        shot["visual"] = analyze_keyframe(client, shot["keyframe"])
    return shots


def _neighbor_transcript(segments, index):
    for offset in (-1, 1):
        neighbor = index + offset
        if 0 <= neighbor < len(segments) and segments[neighbor].audio:
            text = segments[neighbor].audio.transcript
            if text:
                return text
    return None


def generate_narration(client, segment, max_words, neighbor_transcript=None):
    context = f"Scene: {segment.visual.description}"
    if segment.visual.on_screen_text:
        context += f"\nOn-screen text: {segment.visual.on_screen_text}"
    if neighbor_transcript:
        context += f'\nNearby dialogue (for continuity, do not repeat): "{neighbor_transcript}"'

    with open(segment.keyframe, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "You are writing an audio description (AD) narration line for a blind/low-vision "
        "viewer, to be inserted into a speech-free gap in this video shot.\n\n"
        f"{context}\n\n"
        f"Write a single narration line of no more than {max_words} words that describes "
        "what's visually happening, without repeating information already implied by "
        "dialogue. Do not use phrases like 'we see' or 'the camera shows' — describe the "
        "action and setting directly. Return only the narration text, nothing else."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(max_output_tokens=256),
    )

    if response.text is None:
        raise RuntimeError(f"{MODEL} returned no text for {segment.keyframe}")
    return response.text.strip()


def fill_narration_gaps(timeline, words_per_sec=None):
    words_per_sec = words_per_sec or NARRATION_WORDS_PER_SEC
    client = genai.Client()

    for i, segment in enumerate(timeline.segments):
        if not segment.ad_eligible:
            continue
        max_words = max(3, int(segment.narratable_gap_sec * words_per_sec))
        neighbor_transcript = _neighbor_transcript(timeline.segments, i)
        segment.ad_narration = generate_narration(
            client, segment, max_words, neighbor_transcript
        )

    return timeline
