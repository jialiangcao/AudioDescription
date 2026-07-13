from audio_extract import NoAudioStreamError, extract_audio
from segmentation import segment_video
from timeline import build_timeline, save_timeline
from vision_analysis import analyze_shots


def process_video(video_path, frames_dir="frames", audio_path="audio.wav", timeline_path="timeline.json"):
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
    print(timeline.model_dump_json(indent=2))
