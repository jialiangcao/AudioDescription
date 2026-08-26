"""Per-clip isolation in the benchmark harness."""

import json

from bench.run import _already_done, clip_blobs


def test_each_clip_gets_its_own_scratch_directory(tmp_path):
    """Regression: JobBlobs takes ``root`` as the directory *itself* — the job id
    namespaces the bucket key, not the local path. Handing every clip the same
    root made each one reuse the first clip's cached shots, frames and vision
    results, so every prediction after the first clip described the wrong video.
    """
    first = clip_blobs("aaaaaaaaaaa", tmp_path)
    second = clip_blobs("bbbbbbbbbbb", tmp_path)

    assert first.root != second.root
    first.path("state/shots.json").write_text('{"shots": []}')
    assert not second.path("state/shots.json").exists()


def test_clip_scratch_is_reused_across_runs_of_the_same_clip(tmp_path):
    """The cache is the point: re-running must not re-pay for vision."""
    blobs = clip_blobs("aaaaaaaaaaa", tmp_path)
    blobs.path("state/shots.json").write_text('{"shots": []}')

    assert clip_blobs("aaaaaaaaaaa", tmp_path).path("state/shots.json").exists()


def test_clips_already_predicted_are_skipped(tmp_path):
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        json.dumps({"video_id": "aaaaaaaaaaa"})
        + "\n"
        + json.dumps({"video_id": "aaaaaaaaaaa"})
        + "\n"
        + json.dumps({"video_id": "bbbbbbbbbbb"})
        + "\n"
    )

    assert _already_done(preds) == {"aaaaaaaaaaa", "bbbbbbbbbbb"}


def test_no_predictions_file_yet_is_not_an_error(tmp_path):
    assert _already_done(tmp_path / "nothing.jsonl") == set()


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def _pair(gt, pred, judge, dur=2.0, movie="A Film"):
    return {
        "video_id": "aaaaaaaaaaa",
        "movie_title": movie,
        "scaled_start": 1.0,
        "scaled_end": 1.0 + dur,
        "duration": dur,
        "frames": 3,
        "words_per_sec": 2.5,
        "gt": gt,
        "pred": pred,
        "llm_ad_eval": judge,
        "cider": 10.0,
    }


def test_summary_measures_the_word_budget_gap():
    """The headline finding the report exists to surface: human describers speak
    faster than our budget allows, so our lines come out systematically short."""
    from bench.report import summarize

    pairs = [
        # 10 words in a 2s window -> the human spoke at 5 w/s; our budget is 5 words.
        _pair("one two three four five six seven eight nine ten", "one two", 1),
        _pair("one two three four five six seven eight nine ten", "one two", 2),
    ]
    s = summarize(pairs, {"cider": 12.0, "llm_ad_eval": 1.5})

    assert s["gt_words_mean"] == 10.0
    assert s["pred_words_mean"] == 2.0
    assert s["human_wps_mean"] == 5.0
    assert s["over_budget"] == 2


def test_summary_counts_character_naming_on_both_sides():
    from bench.report import summarize

    pairs = [
        _pair("Paul nudges Muller.", "He nudges the man.", 3),
        _pair("The others glare.", "The others glare.", 4),
    ]
    s = summarize(pairs, {})

    assert s["gt_named"] == 1
    assert s["pred_named"] == 0


def test_report_embeds_every_pair(tmp_path):
    """The pairs are the deliverable — none may be dropped by a filter default."""
    import json as _json

    from bench.report import build_report

    preds = tmp_path / "p.jsonl"
    preds.write_text(
        "\n".join(
            _json.dumps(_pair(f"reference {i}", f"prediction {i}", i % 6))
            for i in range(12)
        )
    )
    out = build_report(preds, tmp_path / "missing_results.json", tmp_path / "r.html")
    html = out.read_text()

    for i in range(12):
        assert f"reference {i}" in html
        assert f"prediction {i}" in html
    assert "<title>" in html


def test_report_survives_a_pair_with_no_prediction(tmp_path):
    import json as _json

    from bench.report import build_report

    preds = tmp_path / "p.jsonl"
    bad = _pair("a reference", None, None)
    preds.write_text(_json.dumps(bad))

    html = build_report(preds, tmp_path / "none.json", tmp_path / "r.html").read_text()
    assert "a reference" in html
