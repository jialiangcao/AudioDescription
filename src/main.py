import os
from pathlib import Path

from audio_extract import NoAudioStreamError, extract_audio
from segmentation import segment_video
from timeline import build_timeline, load_timeline, save_timeline
from transcription import transcribe
from vision_analysis import analyze_shots, fill_narration_gaps
from voice_activity import detect_speech_regions

ROOT_DIR = Path(__file__).resolve().parent.parent
IN_DIR = ROOT_DIR / "in"
OUT_DIR = ROOT_DIR / "out"


def print_timeline(timeline):
    print(
        f"\n=== Timeline: {timeline.video_id} ({timeline.duration_sec:.2f}s, "
        f"{len(timeline.segments)} segments) ==="
    )
    for seg in timeline.segments:
        print(f"\n[{seg.id}] {seg.start:.2f}s - {seg.end:.2f}s")
        print(f"  visual: {seg.visual.description}")
        if seg.audio and seg.audio.transcript:
            print(f'  dialogue: "{seg.audio.transcript}"')
        if seg.audio:
            print(
                f"  silence_ratio: {seg.audio.silence_ratio:.2f}  "
                f"narratable_gap_sec: {seg.narratable_gap_sec:.2f}  "
                f"ad_eligible: {seg.ad_eligible}"
            )
        if seg.ad_narration:
            print(f'  ad_narration: "{seg.ad_narration}"')
    print()


def process_video(video_path, frames_dir=None, audio_path=None, timeline_path=None):
    frames_dir = frames_dir or str(OUT_DIR / "frames")
    audio_path = audio_path or str(OUT_DIR / "audio.wav")
    timeline_path = timeline_path or str(OUT_DIR / "timeline.json")

    if os.path.exists(timeline_path):
        print(
            f"[cache] found existing timeline at {timeline_path}, loading it instead of reprocessing"
        )
        timeline = load_timeline(timeline_path)
        print_timeline(timeline)
        return timeline

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[1/6] segmenting shots from {video_path} ...")
    shots = segment_video(video_path, out_dir=frames_dir)
    print(f"      found {len(shots)} shots, keyframes written to {frames_dir}/")

    print("[2/6] running vision analysis on each keyframe ...")
    shots = analyze_shots(shots)
    print(f"      described {len(shots)} shots")

    speech_regions = []
    transcript_segments = []
    print(f"[3/6] extracting audio from {video_path} ...")
    try:
        extract_audio(video_path, out_path=audio_path)
        print(f"      wrote {audio_path}")

        print("[4/6] running voice activity detection ...")
        speech_regions = detect_speech_regions(audio_path)
        print(f"      found {len(speech_regions)} speech regions: {speech_regions}")

        print("[5/6] transcribing detected speech ...")
        transcript_segments = transcribe(audio_path, speech_regions)
        for seg in transcript_segments:
            print(f'      [{seg["start"]:.2f}s - {seg["end"]:.2f}s] "{seg["text"]}"')
        print(f"      transcribed {len(transcript_segments)} dialogue segments")
    except NoAudioStreamError:
        print("      no audio stream in video, skipping VAD/transcription")

    print(
        "[6/6] building timeline and generating audio-description narration for gaps ..."
    )
    timeline = build_timeline(video_path, shots, speech_regions, transcript_segments)
    ad_eligible_count = sum(1 for seg in timeline.segments if seg.ad_eligible)
    print(
        f"      {ad_eligible_count}/{len(timeline.segments)} segments are ad_eligible, generating narration ..."
    )
    timeline = fill_narration_gaps(timeline)

    save_timeline(timeline, out_path=timeline_path)
    print(f"      saved timeline to {timeline_path}")

    print_timeline(timeline)
    return timeline


if __name__ == "__main__":
    timeline = process_video(str(IN_DIR / "test.mp4"))
