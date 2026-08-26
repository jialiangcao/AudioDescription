"""The GT-anchored timeline: CSV rows in, one narratable Segment each out."""

import pytest

from bench.anchored import build_anchored_timeline, frames_for_window
from bench.dataset import AdRow, group_by_clip, spread_across_movies
from timeline import NARRATION_WORDS_PER_SEC


def _row(text, start, end, video="2011/_SQr8I3lcW8"):
    return AdRow(
        text=text,
        cmd_filename=video,
        video_id=video.rsplit("/", 1)[-1],
        scaled_start=start,
        scaled_end=end,
        duration=end - start,
        imdbid="tt0020629",
        movie_title="All Quiet on the Western Front",
        cmd_clip_idx=4,
        split="train",
    )


def _shots(times, per_shot=4):
    """Shots holding frames at ``times``, chunked so the segments cut across them."""
    shots, frames = [], list(enumerate(times))
    for shot_id, chunk_start in enumerate(range(0, len(frames), per_shot)):
        chunk = frames[chunk_start : chunk_start + per_shot]
        shots.append(
            {
                "id": shot_id,
                "start": chunk[0][1],
                "end": chunk[-1][1],
                "frames": [
                    {
                        "index": i,
                        "time": float(t),
                        "key": f"frames/shot_{shot_id:04d}_{i:02d}.jpg",
                    }
                    for i, (_, t) in enumerate(chunk)
                ],
            }
        )
    return shots


# --------------------------------------------------------------------------- #
# frame selection
# --------------------------------------------------------------------------- #


def test_window_selects_the_padded_span_in_time_order():
    frames = [{"index": i, "time": float(i), "key": f"f{i}"} for i in range(20)]
    selected = frames_for_window(frames, 10.0, 12.0, pad_sec=2.0)

    assert [f["time"] for f in selected] == [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]


def test_empty_window_falls_back_to_the_nearest_frame():
    """A 2s reference against coarse sampling can fall between two frames, and a
    segment with no images gives the model no view of the video at all."""
    frames = [
        {"index": 0, "time": 0.0, "key": "a"},
        {"index": 1, "time": 30.0, "key": "b"},
    ]
    selected = frames_for_window(frames, 14.0, 16.0, pad_sec=1.0)

    assert len(selected) == 1
    assert selected[0]["time"] == 0.0  # centre 15.0 is nearer 0.0 than 30.0


def test_no_frames_at_all_yields_nothing_rather_than_raising():
    assert frames_for_window([], 1.0, 2.0) == []


# --------------------------------------------------------------------------- #
# the timeline
# --------------------------------------------------------------------------- #


def test_one_segment_per_row_renumbered_in_temporal_order():
    rows = [
        _row("third", 30.0, 32.0),
        _row("first", 5.0, 7.0),
        _row("second", 12.0, 15.0),
    ]
    timeline = build_anchored_timeline("vid", 60.0, _shots(range(0, 40)), rows)

    assert [seg.id for seg in timeline.segments] == [0, 1, 2]
    assert [seg.start for seg in timeline.segments] == [5.0, 12.0, 30.0]
    assert timeline.job_id == "vid"
    assert timeline.duration_sec == 60.0


def test_every_segment_is_forced_eligible_and_carries_frames():
    rows = [_row("a", 5.0, 7.0), _row("b", 20.0, 24.0)]
    timeline = build_anchored_timeline("vid", 60.0, _shots(range(0, 40)), rows)

    for segment in timeline.segments:
        assert segment.ad_eligible is True
        assert segment.frames


def test_gap_comes_from_the_reference_duration_not_from_silence():
    """The word budget must match what the human describer had to fit."""
    row = _row("Paul nudges Muller.", 100.0, 104.4)
    timeline = build_anchored_timeline("vid", 200.0, _shots(range(90, 120)), [row])
    segment = timeline.segments[0]

    assert segment.narratable_gap_sec == pytest.approx(4.4)
    assert segment.narration_start_sec == pytest.approx(100.0)
    # This is the number fill_narration_gaps turns into max_words. Derived from
    # the constant, not hardcoded: the assertion is that the reference window
    # sets the budget, not what the pacing happens to be tuned to today.
    assert int(segment.narratable_gap_sec * NARRATION_WORDS_PER_SEC) == int(
        4.4 * NARRATION_WORDS_PER_SEC
    )


def test_dialogue_spoken_during_the_window_is_attached():
    row = _row("a", 10.0, 14.0)
    timeline = build_anchored_timeline(
        "vid",
        60.0,
        _shots(range(0, 30)),
        [row],
        speech_regions=[(10.0, 12.0)],
        transcript_segments=[
            {"start": 10.0, "end": 12.0, "text": "Get down!"},
            {"start": 40.0, "end": 42.0, "text": "Elsewhere entirely."},
        ],
    )
    audio = timeline.segments[0].audio

    assert audio is not None
    assert audio.transcript == "Get down!"
    assert audio.has_speech is True


def test_segments_draw_frames_across_shot_boundaries():
    """A human AD window has no reason to respect a camera cut."""
    shots = _shots(range(0, 20), per_shot=4)  # cuts at 4, 8, 12, 16
    timeline = build_anchored_timeline("vid", 60.0, shots, [_row("a", 7.0, 9.0)])

    keys = [frame.key for frame in timeline.segments[0].frames]
    assert len({key.split("_")[1] for key in keys}) > 1


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #


def test_group_by_clip_keys_on_the_youtube_id_and_sorts_by_time():
    rows = [
        _row("b", 20.0, 22.0),
        _row("a", 5.0, 7.0),
        _row("other", 1.0, 3.0, video="2012/AAAAAAAAAAA"),
    ]
    grouped = group_by_clip(rows)

    assert set(grouped) == {"_SQr8I3lcW8", "AAAAAAAAAAA"}
    assert [r.text for r in grouped["_SQr8I3lcW8"]] == ["a", "b"]


def test_clips_are_ordered_to_cover_as_many_movies_as_possible():
    """Regression: the CSV is ordered by movie, so slicing it directly made
    ``--limit 15`` mean "the first two movies" — a useless eval sample."""
    rows = []
    for movie, imdb in (("A", "tt1"), ("B", "tt2"), ("C", "tt3")):
        for clip in range(3):
            rows.append(
                AdRow(
                    text="x",
                    cmd_filename=f"2011/{movie}{clip}aaaaaaaaa",
                    video_id=f"{movie}{clip}aaaaaaaaa",
                    scaled_start=1.0,
                    scaled_end=3.0,
                    duration=2.0,
                    imdbid=imdb,
                    movie_title=movie,
                    cmd_clip_idx=clip,
                    split="eval",
                )
            )
    ordered = spread_across_movies(group_by_clip(rows))

    assert len(ordered) == 9
    # The first three cover all three movies, rather than three clips of movie A.
    assert {video_id[0] for video_id in ordered[:3]} == {"A", "B", "C"}
    assert {video_id[0] for video_id in ordered[3:6]} == {"A", "B", "C"}


def test_spreading_an_empty_catalogue_is_not_an_error():
    assert spread_across_movies({}) == []
