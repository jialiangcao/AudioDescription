import os

import pytest

from segmentation import detect_shots, extract_keyframes, segment_video


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


def test_segment_video_handles_a_video_with_no_cuts(tmp_path, single_shot_video):
    records = segment_video(single_shot_video, out_dir=str(tmp_path / "frames"))

    assert len(records) == 1
    record = records[0]
    assert record["frames"]
    for frame in record["frames"]:
        assert os.path.exists(frame["path"])
        assert record["start"] <= frame["time"] <= record["end"]


def test_segment_video_writes_frames_and_shot_records(tmp_path, synthetic_video):
    out_dir = tmp_path / "frames"

    records = segment_video(synthetic_video, out_dir=str(out_dir))

    assert len(records) == 2
    for record in records:
        assert record["frames"]
        for expected_index, frame in enumerate(record["frames"]):
            assert os.path.exists(frame["path"])
            # index is the frame's position within its shot...
            assert frame["index"] == expected_index
            # ...and time its absolute timestamp in the video, inside the shot.
            assert record["start"] <= frame["time"] <= record["end"]


def test_extract_keyframes_samples_at_the_requested_interval(tmp_path, synthetic_video):
    out_dir = tmp_path / "frames"

    shots = detect_shots(synthetic_video)
    records = extract_keyframes(
        synthetic_video, shots, out_dir=str(out_dir), interval_sec=0.5
    )

    for record in records:
        times = [frame["time"] for frame in record["frames"]]
        assert times == sorted(times)
        # A ~2s shot sampled every 0.5s yields multiple frames, each 0.5s apart.
        assert len(times) > 1
        gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
        assert all(gap == pytest.approx(0.5, abs=0.01) for gap in gaps)
