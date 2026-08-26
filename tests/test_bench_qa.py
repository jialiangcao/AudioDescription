"""Scoring a prose-answering agent against multiple choice."""

import json

import pytest

from bench.qa_dataset import QaItem, group_by_video
from bench.qa_report import summarize_run
from bench.qa_run import _record, parse_choice

OPTIONS = ["A. Apples.", "B. Candles.", "C. Berries.", "D. All the same."]


def _item(**over):
    base = dict(
        dataset="video-mme",
        question_id="001-1",
        video_id="abcdefghijk",
        url="https://youtu.be/abcdefghijk",
        question="Which decoration is most numerous?",
        options=OPTIONS,
        answer="C",
        task_type="Counting Problem",
        domain="Knowledge",
        duration="short",
    )
    return QaItem(**(base | over))


# --------------------------------------------------------------------------- #
# parsing an answer out of prose
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "answer",
    [
        "C",
        "The answer is C.",
        "Looking at the frames, berries dominate.\n\nC",
        "**C**",
        "After inspecting frame 12, I conclude option C is correct.",
        "Answer: C",
    ],
)
def test_letters_are_found_however_the_agent_phrases_it(answer):
    assert parse_choice(answer, OPTIONS) == "C"


def test_the_last_letter_wins_when_the_agent_changes_its_mind():
    """The planner loop revises; the committed answer is the final one."""
    answer = "Initially the answer is A.\nOn reflection that was wrong.\n\nC"
    assert parse_choice(answer, OPTIONS) == "C"


def test_a_prose_answer_with_no_letter_falls_back_to_the_option_text():
    """Naming the right thing without labelling it is a correct answer; scoring
    it wrong would measure instruction-following, not video understanding."""
    assert parse_choice("There are more berries than anything else.", OPTIONS) == "C"


def test_an_answer_matching_nothing_is_left_uncommitted():
    assert parse_choice("I could not determine this from the video.", OPTIONS) is None
    assert parse_choice("", OPTIONS) is None
    assert parse_choice(None, OPTIONS) is None


def test_generic_word_overlap_does_not_count_as_a_match():
    """ "The" appearing in both is not evidence of anything."""
    assert parse_choice("The video is unclear about the.", OPTIONS) is None


# --------------------------------------------------------------------------- #
# records and scoring
# --------------------------------------------------------------------------- #


def test_a_record_marks_correctness_against_the_gold_letter():
    right = _record(_item(), "agent", "The answer is C.", 3, 12.5)
    wrong = _record(_item(), "agent", "The answer is A.", 3, 12.5)

    assert right["predicted"] == "C" and right["correct"] is True
    assert wrong["predicted"] == "A" and wrong["correct"] is False
    assert right["cycles"] == 3 and right["seconds"] == 12.5


def test_an_uncommitted_answer_counts_as_wrong_not_as_missing():
    """Dropping unparsed answers from the denominator would inflate accuracy."""
    record = _record(_item(), "agent", "I don't know.", 1, 1.0)
    assert record["predicted"] is None
    assert record["correct"] is False

    stats = summarize_run([record, _record(_item(), "agent", "C", 1, 1.0)])
    assert stats["questions"] == 2
    assert stats["unparsed"] == 1
    assert stats["accuracy"] == 50.0


def test_summary_reports_chance_from_the_option_count():
    """A result is only readable against its own floor: 4-way is 25%, 5-way 20%."""
    four = summarize_run([_record(_item(), "agent", "C", 1, 1.0)])
    five = summarize_run(
        [_record(_item(options=[*OPTIONS, "E. None."]), "agent", "C", 1, 1.0)]
    )

    assert four["chance"] == 25.0
    assert five["chance"] == 20.0


def test_summary_breaks_accuracy_down_by_task_type():
    records = [
        _record(_item(task_type="Counting Problem"), "agent", "C", 1, 1.0),
        _record(_item(task_type="Counting Problem"), "agent", "A", 1, 1.0),
        _record(_item(task_type="OCR Problems"), "agent", "C", 1, 1.0),
    ]
    by_task = dict((t, acc) for t, acc, _ in summarize_run(records)["by_task"])

    assert by_task["Counting Problem"] == 50.0
    assert by_task["OCR Problems"] == 100.0


def test_report_separates_the_modes_it_is_given(tmp_path):
    from bench.qa_report import build_report, load_runs

    path = tmp_path / "preds.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_record(_item(), mode, letter, 1, 1.0))
            for mode, letter in (("agent", "C"), ("blind", "A"), ("subtitles", "C"))
        )
    )
    runs = load_runs([path])

    assert set(runs) == {"agent", "blind", "subtitles"}
    html = build_report([path], tmp_path / "r.html").read_text()
    assert "Blind (no video)" in html
    assert "<title>" in html


def test_questions_group_by_video_so_a_clip_is_analysed_once():
    items = [
        _item(question_id="1"),
        _item(question_id="2"),
        _item(video_id="zzzzzzzzzzz"),
    ]
    grouped = group_by_video(items)

    assert len(grouped["abcdefghijk"]) == 2
    assert len(grouped["zzzzzzzzzzz"]) == 1


# --------------------------------------------------------------------------- #
# dead videos are missing measurements, not wrong answers
# --------------------------------------------------------------------------- #


def test_an_unfetchable_video_is_excluded_from_accuracy():
    """Video-MME's YouTube sources rot. Scoring a question wrong because its
    video is gone would penalise the video modes against the blind control,
    which needs no video, and make the two numbers incomparable."""
    good = _record(_item(question_id="1"), "agent", "C", 2, 5.0)
    dead = _record(_item(question_id="2"), "agent", None, None, 0.0, unavailable=True)

    stats = summarize_run([good, dead])

    assert stats["questions"] == 1
    assert stats["unavailable"] == 1
    assert stats["accuracy"] == 100.0


def test_legacy_records_without_the_flag_are_still_recognised():
    from bench.qa_report import is_unavailable

    legacy = _record(_item(), "agent", None, None, 0.0)
    assert is_unavailable(legacy)
    # A blind answer has no video by design and must never be treated this way.
    assert not is_unavailable(_record(_item(), "blind", None, None, 0.0))


def test_modes_are_compared_over_the_questions_all_of_them_measured():
    from bench.qa_report import common_questions

    runs = {
        "agent": [
            _record(_item(question_id="1"), "agent", "C", 1, 1.0),
            _record(_item(question_id="2"), "agent", None, None, 0.0, unavailable=True),
        ],
        "blind": [
            _record(_item(question_id="1"), "blind", "A", 0, 1.0),
            _record(_item(question_id="2"), "blind", "C", 0, 1.0),
        ],
    }
    shared = common_questions(runs)

    assert shared == {"1"}
    # Blind got question 2 right, but the agent never saw it — including it
    # would flatter the control.
    assert summarize_run(runs["blind"], only=shared)["accuracy"] == 0.0
    assert summarize_run(runs["agent"], only=shared)["accuracy"] == 100.0
