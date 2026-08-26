import logging

logger = logging.getLogger(__name__)

# OpenCV and PySceneDetect are imported inside the functions that need them
# rather than at module scope, so the slim worker image — which carries no
# torch, no OpenCV and no scenedetect — can still import tasks.py and run the
# Gemini-only stages. Only the media queue ever calls into this module.

# Blob-key prefix for sampled frames; keys look like frames/shot_0000_00.jpg.
FRAMES_PREFIX = "frames"


def frame_key(shot_id: int, index: int) -> str:
    return f"{FRAMES_PREFIX}/shot_{shot_id:04d}_{index:02d}.jpg"


def probe_video(video_path):
    """Read fps / frame count / duration without decoding the video.

    Split out from ``build_timeline`` (which used to open the video itself) so
    the duration can be recorded once, by the one stage that already has the
    source file on disk, and carried through the rest of the pipeline as data.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    # OpenCV reports -1 (not 0) for a file it can't decode, so a plain falsiness
    # check lets it through and yields a *negative* duration downstream.
    readable = fps > 0 and frame_count > 0
    if not readable:
        logger.warning(
            "probe_video: could not read fps/frame count for %s (fps=%s, frames=%s), "
            "duration=0",
            video_path,
            fps,
            frame_count,
        )
    return {
        "fps": fps if fps > 0 else 0.0,
        "frame_count": frame_count if frame_count > 0 else 0.0,
        "duration_sec": frame_count / fps if readable else 0.0,
    }


def detect_shots(video_path, threshold=27.0):
    """Detect shot (camera cut) boundaries in a video.

    Returns a list of (start_sec, end_sec) tuples covering the whole video.
    """
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

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


def extract_keyframes(video_path, shots, blobs, interval_sec=1.0):
    """Extract frames sampled every `interval_sec` within each shot.

    Frames are written into `blobs`' scratch dir and returned as shot dicts:
    `{id, start, end, frames}`, where `frames` is every sampled frame in the
    shot in temporal order, each recorded as `{index, time, key}` — `index` is
    its position within the shot, `time` its absolute timestamp in the video,
    and `key` its job-relative blob key. No frame is privileged: every one is
    described on its own by `vision_analysis.analyze_shots`.

    Uploading the written frames is the caller's job (it knows whether the run
    is local or backed by a bucket); the keys returned here are what to upload.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.debug(
        "extract_keyframes: %d shot(s) @ %.2ffps, every %.1fs -> %s",
        len(shots),
        fps,
        interval_sec,
        blobs.root,
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
            key = frame_key(i, index)
            cv2.imwrite(str(blobs.path(key)), frame)
            frames.append({"index": index, "time": round(t, 2), "key": key})

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


def segment_video(video_path, blobs, threshold=27.0, interval_sec=None):
    """Detect shots and sample frames within each. ``interval_sec`` is the cost knob.

    Stage 2 issues one Gemini call per sampled frame, so this interval — not the
    shot count — is what a job's API cost scales with. It is a parameter rather
    than a constant so the benchmark harness can trade frame density against
    spend from the command line; ``None`` keeps ``extract_keyframes``' default.
    """
    shots = detect_shots(video_path, threshold=threshold)
    if interval_sec is None:
        return extract_keyframes(video_path, shots, blobs)
    return extract_keyframes(video_path, shots, blobs, interval_sec=interval_sec)
