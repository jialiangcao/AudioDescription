import pytest

from voice_activity import longest_speech_free_gap


def test_no_speech_returns_full_window():
    assert longest_speech_free_gap(0.0, 10.0, []) == pytest.approx(10.0)


def test_returns_longest_gap_not_total_silence():
    # Speech chops the window into pieces. Total silence is 1+1+1+1+2 = 6s, but
    # the longest *contiguous* silent stretch is only the final 2s (8.0-10.0).
    regions = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    assert longest_speech_free_gap(0.0, 10.0, regions) == pytest.approx(2.0)


def test_merges_overlapping_regions():
    # (1,5) and (3,7) overlap into one 1.0-7.0 span; longest gap is 7.0-10.0.
    assert longest_speech_free_gap(
        0.0, 10.0, [(1.0, 5.0), (3.0, 7.0)]
    ) == pytest.approx(3.0)


def test_clips_regions_extending_outside_window():
    # Regions spill past both edges; only the 3.0-7.0 gap inside the window counts.
    regions = [(0.0, 3.0), (7.0, 10.0)]
    assert longest_speech_free_gap(2.0, 8.0, regions) == pytest.approx(4.0)


def test_unsorted_regions_are_handled():
    assert longest_speech_free_gap(
        0.0, 10.0, [(5.0, 6.0), (1.0, 2.0)]
    ) == pytest.approx(4.0)


def test_empty_window_returns_zero():
    assert longest_speech_free_gap(5.0, 5.0, []) == 0.0
    assert longest_speech_free_gap(5.0, 4.0, []) == 0.0


def test_speech_filling_window_returns_zero():
    assert longest_speech_free_gap(0.0, 3.0, [(0.0, 3.0)]) == pytest.approx(0.0)
