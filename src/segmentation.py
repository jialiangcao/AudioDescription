import os

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


def detect_shots(video_path, threshold=27.0):
    """Detect shot (camera cut) boundaries in a video.

    Returns a list of (start_sec, end_sec) tuples covering the whole video.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        return [(0.0, video.duration)]

    return [(start.seconds, end.seconds) for start, end in scene_list]


def extract_keyframes(video_path, shots, out_dir="frames", interval_sec=2.0):
    """Extract frames sampled every `interval_sec` within each shot.

    Returns a list of shot dicts: {id, start, end, keyframe, keyframes}, where
    `keyframe` is the single sampled frame closest to the shot's midpoint (kept
    for backward compatibility with the rest of the pipeline) and `keyframes` is
    every sampled frame path in the shot, in temporal order.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    shot_records = []
    for i, (start, end) in enumerate(shots):
        timestamps = []
        t = start
        while t <= end:
            timestamps.append(t)
            t += interval_sec
        if not timestamps:
            timestamps = [start]

        midpoint = (start + end) / 2
        paths = []
        for t in timestamps:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            path = os.path.join(out_dir, f"shot_{i:04d}_{len(paths):02d}.jpg")
            cv2.imwrite(path, frame)
            paths.append((t, path))

        if not paths:
            continue

        keyframe = min(paths, key=lambda tp: abs(tp[0] - midpoint))[1]
        shot_records.append(
            {
                "id": i,
                "start": round(start, 2),
                "end": round(end, 2),
                "keyframe": keyframe,
                "keyframes": [p for _, p in paths],
            }
        )

    cap.release()
    return shot_records


def segment_video(video_path, out_dir="frames", threshold=27.0):
    shots = detect_shots(video_path, threshold=threshold)
    return extract_keyframes(video_path, shots, out_dir=out_dir)
