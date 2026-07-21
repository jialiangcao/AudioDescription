import os

import pytest

from segmentation import detect_shots, segment_video


def test_detect_shots_finds_the_color_cut(synthetic_video):
    shots = detect_shots(synthetic_video)

    assert len(shots) == 2
    (s0_start, s0_end), (s1_start, s1_end) = shots
    assert s0_start == pytest.approx(0.0)
    assert s0_end == pytest.approx(2.0, abs=0.5)
    assert s1_end == pytest.approx(4.0, abs=0.2)


def test_segment_video_writes_keyframes_and_shot_records(tmp_path, synthetic_video):
    out_dir = tmp_path / "frames"

    records = segment_video(synthetic_video, out_dir=str(out_dir))

    assert len(records) == 2
    for record in records:
        assert os.path.exists(record["keyframe"])
        assert record["keyframes"]
        for path in record["keyframes"]:
            assert os.path.exists(path)
