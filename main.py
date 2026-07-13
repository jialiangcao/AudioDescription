import os

from audio_extract import NoAudioStreamError, extract_audio
from segmentation import segment_video
from timeline import build_timeline, load_timeline, save_timeline
from vision_analysis import analyze_shots

# question testing
import anthropic
from dotenv import load_dotenv
load_dotenv
MODEL = "claude-opus-4-8"

def process_video(video_path, frames_dir="frames", audio_path="audio.wav", timeline_path="timeline.json"):
    if os.path.exists(timeline_path):
        return load_timeline(timeline_path)

    shots = segment_video(video_path, out_dir=frames_dir)
    shots = analyze_shots(shots)

    try:
        extract_audio(video_path, out_path=audio_path)
    except NoAudioStreamError:
        pass

    timeline = build_timeline(video_path, shots)
    save_timeline(timeline, out_path=timeline_path)
    return timeline


if __name__ == "__main__":
    timeline = process_video("test.mp4")
    question = "What does the woman look like?"
    timestamp = 31
    client = anthropic.Anthropic()
    for seg in timeline.segments:
        if not (seg.start < timestamp and seg.end > timestamp):
            continue
        else:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": f"Answer {question} using {seg.visual}"}
                ]
            )
            print(response)


