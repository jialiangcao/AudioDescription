import base64

import anthropic
from dotenv import load_dotenv

from timeline import NARRATION_WORDS_PER_SEC

load_dotenv()

MODEL = "claude-sonnet-4-6"

SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "A concise, vivid description of what is visually happening in this shot — subject, action, and setting.",
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Notable people, objects, or creatures visible in the shot.",
        },
        "setting": {
            "type": "string",
            "description": "Where/when the shot appears to take place.",
        },
        "on_screen_text": {
            "type": ["string", "null"],
            "description": "Any text visible on screen (titles, signs, captions), or null if none.",
        },
    },
    "required": ["description", "entities", "setting", "on_screen_text"],
    "additionalProperties": False,
}


def analyze_keyframe(client, keyframe_path):
    with open(keyframe_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe what is visually happening in this video frame.",
                    },
                ],
            }
        ],
    )

    import json

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def analyze_shots(shots):
    client = anthropic.Anthropic()
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
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    prompt = (
        "You are writing an audio description (AD) narration line for a blind/low-vision "
        "viewer, to be inserted into a speech-free gap in this video shot.\n\n"
        f"{context}\n\n"
        f"Write a single narration line of no more than {max_words} words that describes "
        "what's visually happening, without repeating information already implied by "
        "dialogue. Do not use phrases like 'we see' or 'the camera shows' — describe the "
        "action and setting directly. Return only the narration text, nothing else."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return next(b.text for b in response.content if b.type == "text").strip()


def fill_narration_gaps(timeline, words_per_sec=None):
    words_per_sec = words_per_sec or NARRATION_WORDS_PER_SEC
    client = anthropic.Anthropic()

    for i, segment in enumerate(timeline.segments):
        if not segment.ad_eligible:
            continue
        max_words = max(3, int(segment.narratable_gap_sec * words_per_sec))
        neighbor_transcript = _neighbor_transcript(timeline.segments, i)
        segment.ad_narration = generate_narration(
            client, segment, max_words, neighbor_transcript
        )

    return timeline
