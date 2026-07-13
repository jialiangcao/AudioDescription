import base64

import anthropic

MODEL = "claude-opus-4-8"

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
        messages=[{
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
                {"type": "text", "text": "Describe what is visually happening in this video frame."},
            ],
        }],
    )

    import json
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def analyze_shots(shots):
    client = anthropic.Anthropic()
    for shot in shots:
        shot["visual"] = analyze_keyframe(client, shot["keyframe"])
    return shots


if __name__ == "__main__":
    from segmentation import segment_video

    shots = segment_video("test.mp4")
    shots = analyze_shots(shots)
    for shot in shots:
        print(shot)
