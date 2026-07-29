import logging
import os

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

logger = logging.getLogger(__name__)


def detect_shots(video_path, threshold=27.0):
    """Detect shot (camera cut) boundaries in a video.

    Returns a list of (start_sec, end_sec) tuples covering the whole video.
    """
    logger.debug("detect_shots: %s (threshold=%.1f)", video_path, threshold)
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        logger.warning(
            "detect_shots: no cuts found, treating whole video as one shot (%s)",
            video.duration,
        )
        # video.duration is a FrameTimecode; unwrap it to plain float seconds so
        # callers get the same shape as the scene-list branch below.
        duration = video.duration.seconds if video.duration else 0.0
        return [(0.0, duration)]

    logger.debug("detect_shots: %d shot(s) detected", len(scene_list))
    return [(start.seconds, end.seconds) for start, end in scene_list]


def extract_keyframes(video_path, shots, out_dir="frames", interval_sec=2.0):
    """Extract frames sampled every `interval_sec` within each shot.

    Returns a list of shot dicts: `{id, start, end, frames}`, where `frames` is
    every sampled frame in the shot in temporal order, each recorded as
    `{index, time, path}` — `index` is its position within the shot and `time`
    its absolute timestamp in the video. No frame is privileged: every one is
    described on its own by `vision_analysis.analyze_shots`.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.debug(
        "extract_keyframes: %d shot(s) @ %.2ffps, every %.1fs -> %s",
        len(shots),
        fps,
        interval_sec,
        out_dir,
    )

    shot_records = []
    for i, (start, end) in enumerate(shots):
        timestamps = []
        t = start
        while t <= end:
            timestamps.append(t)
            t += interval_sec
        if not timestamps:
            timestamps = [start]

        frames = []
        for t in timestamps:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            index = len(frames)
            path = os.path.join(out_dir, f"shot_{i:04d}_{index:02d}.jpg")
            cv2.imwrite(path, frame)
            frames.append({"index": index, "time": round(t, 2), "path": path})

        if not frames:
            logger.warning("extract_keyframes: shot %d yielded no readable frames", i)
            continue

        shot_records.append(
            {
                "id": i,
                "start": round(start, 2),
                "end": round(end, 2),
                "frames": frames,
            }
        )

    cap.release()
    logger.debug(
        "extract_keyframes: produced %d shot record(s), %d frame(s) total",
        len(shot_records),
        sum(len(r["frames"]) for r in shot_records),
    )
    return shot_records


def segment_video(video_path, out_dir="frames", threshold=27.0):
    shots = detect_shots(video_path, threshold=threshold)
    return extract_keyframes(video_path, shots, out_dir=out_dir)
