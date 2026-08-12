import pytest

from segmentation import detect_shots, extract_keyframes, probe_video, segment_video


def test_probe_video_reports_fps_and_duration(synthetic_video):
    """Duration used to be derived inside build_timeline; it now comes from here.

    The 4s synthetic clip is muxed at 10fps, so duration is frame_count / fps.
    """
    meta = probe_video(synthetic_video)

    assert meta["fps"] == pytest.approx(10.0)
    assert meta["duration_sec"] == pytest.approx(4.0, abs=0.2)
    assert meta["frame_count"] == pytest.approx(40, abs=2)


def test_probe_video_reports_zero_duration_when_fps_is_unreadable(tmp_path):
    """A file OpenCV can't decode yields fps=0; duration must not divide by it."""
    not_a_video = tmp_path / "broken.mp4"
    not_a_video.write_bytes(b"not actually a video")

    meta = probe_video(str(not_a_video))

    assert meta["fps"] == 0
    assert meta["duration_sec"] == 0.0


def test_detect_shots_finds_the_color_cut(synthetic_video):
    shots = detect_shots(synthetic_video)

    assert len(shots) == 2
    (s0_start, s0_end), (s1_start, s1_end) = shots
    assert s0_start == pytest.approx(0.0)
    assert s0_end == pytest.approx(2.0, abs=0.5)
    assert s1_end == pytest.approx(4.0, abs=0.2)


def test_detect_shots_falls_back_to_plain_float_seconds_when_no_cuts(
    single_shot_video,
):
    """A cut-free video yields one shot whose bounds are plain float seconds.

    Regression: the no-cuts fallback used to hand back PySceneDetect's
    `video.duration` FrameTimecode, which has no `__round__`, so
    extract_keyframes blew up with a TypeError on any single-shot video.
    """
    shots = detect_shots(single_shot_video)

    assert len(shots) == 1
    (start, end) = shots[0]
    assert isinstance(start, float)
    assert isinstance(end, float)
    assert end == pytest.approx(3.0, abs=0.5)


def test_segment_video_handles_a_video_with_no_cuts(job_blobs, single_shot_video):
    records = segment_video(single_shot_video, job_blobs)

    assert len(records) == 1
    record = records[0]
    assert record["frames"]
    for frame in record["frames"]:
        assert job_blobs.path(frame["key"]).exists()
        assert record["start"] <= frame["time"] <= record["end"]


def test_segment_video_writes_frames_and_shot_records(job_blobs, synthetic_video):
    records = segment_video(synthetic_video, job_blobs)

    assert len(records) == 2
    for record in records:
        assert record["frames"]
        for expected_index, frame in enumerate(record["frames"]):
            assert job_blobs.path(frame["key"]).exists()
            # index is the frame's position within its shot...
            assert frame["index"] == expected_index
            # ...and time its absolute timestamp in the video, inside the shot.
            assert record["start"] <= frame["time"] <= record["end"]


def test_extract_keyframes_samples_at_the_requested_interval(
    job_blobs, synthetic_video
):
    shots = detect_shots(synthetic_video)
    records = extract_keyframes(synthetic_video, shots, job_blobs, interval_sec=0.5)

    for record in records:
        times = [frame["time"] for frame in record["frames"]]
        assert times == sorted(times)
        # A ~2s shot sampled every 0.5s yields multiple frames, each 0.5s apart.
        assert len(times) > 1
        gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
        assert all(gap == pytest.approx(0.5, abs=0.01) for gap in gaps)
