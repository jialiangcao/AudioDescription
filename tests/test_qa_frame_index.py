from qa.frame_index import FrameIndex
from timeline import Frame, Segment, Timeline


def _timeline():
    return Timeline(
        video_id="v.mp4",
        duration_sec=10.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=4.0,
                frames=[
                    Frame(index=0, time=0.0, path="/f/shot_0000_00.jpg"),
                    Frame(index=1, time=2.0, path="/f/shot_0000_01.jpg"),
                ],
            ),
            Segment(
                id=1,
                start=4.0,
                end=10.0,
                frames=[
                    Frame(index=0, time=4.0, path="/f/shot_0001_00.jpg"),
                    Frame(index=1, time=6.0, path="/f/shot_0001_01.jpg"),
                    Frame(index=2, time=8.0, path="/f/shot_0001_02.jpg"),
                ],
            ),
        ],
    )


def test_from_timeline_uses_frame_times_and_paths():
    index = FrameIndex.from_timeline(_timeline())
    assert index.duration_sec == 10.0
    assert index.entries == [
        (0.0, "/f/shot_0000_00.jpg"),
        (2.0, "/f/shot_0000_01.jpg"),
        (4.0, "/f/shot_0001_00.jpg"),
        (6.0, "/f/shot_0001_01.jpg"),
        (8.0, "/f/shot_0001_02.jpg"),
    ]
    assert index.paths() == [path for _, path in index.entries]
    assert index.timestamp_of("/f/shot_0001_01.jpg") == 6.0


def test_in_range_bounds_are_inclusive():
    index = FrameIndex.from_timeline(_timeline())
    assert index.in_range(2.0, 6.0) == [
        (2.0, "/f/shot_0000_01.jpg"),
        (4.0, "/f/shot_0001_00.jpg"),
        (6.0, "/f/shot_0001_01.jpg"),
    ]
    assert index.in_range(2.5, 3.5) == []
    assert index.in_range(-5.0, 100.0) == index.entries


def test_windows_bucket_on_fixed_grid_and_drop_empty(fake_frame_index):
    # Frames at t=0,2,...,98 over 100s -> four 30s windows, none empty.
    windows = fake_frame_index.windows(30.0)
    assert [(start, end) for start, end, _ in windows] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 90.0),
        (90.0, 120.0),
    ]
    assert [len(paths) for _, _, paths in windows] == [15, 15, 15, 5]

    # A gap in coverage drops the empty middle window but keeps grid starts.
    sparse = FrameIndex(entries=[(1.0, "a.jpg"), (65.0, "b.jpg")], duration_sec=70.0)
    windows = sparse.windows(30.0)
    assert [(start, paths) for start, _, paths in windows] == [
        (0.0, ["a.jpg"]),
        (60.0, ["b.jpg"]),
    ]


def test_uniform_sample_spreads_and_dedups(fake_frame_index):
    path_at = dict(fake_frame_index.entries)

    # Asking for more samples than frames collapses to every frame in range
    # (the range is inclusive, so t=20 appears too: it is nearest to the last
    # sample instants).
    all_in_range = fake_frame_index.uniform_sample(0.0, 20.0, 70)
    assert all_in_range == [path_at[float(t)] for t in range(0, 22, 2)]

    # A small count spreads across the range without duplicates.
    picked = fake_frame_index.uniform_sample(0.0, 100.0, 5)
    assert len(picked) == 5
    assert len(set(picked)) == 5
    times = [int(p.split("_")[-1].split(".")[0]) for p in picked]
    assert times == sorted(times)
    # endpoint=False: the last sample instant is 80.0, not 100.0.
    assert times[-1] == 80

    assert fake_frame_index.uniform_sample(0.0, 100.0, 0) == []
    assert fake_frame_index.uniform_sample(200.0, 300.0, 5) == []
