"""CIDEr and the LLM-AD-eval judge."""

import json

import pytest

from bench import metrics

pytest.importorskip(
    "pycocoevalcap", reason="install the bench extra: uv sync --extra bench"
)


def _pair(gt, pred):
    return {"gt": gt, "pred": pred}


# --------------------------------------------------------------------------- #
# tokenization + CIDEr
# --------------------------------------------------------------------------- #


def test_tokenize_lowercases_and_strips_punctuation():
    assert (
        metrics.tokenize("Paul nudges Müller, then stops!")
        == "paul nudges m ller then stops"
    )


def test_identical_text_scores_far_above_disjoint_text():
    """CIDEr is corpus-relative, so the meaningful assertion is the ordering."""
    corpus = [
        _pair("a man walks into the room", "a man walks into the room"),
        _pair("she opens the heavy door", "she opens the heavy door"),
        _pair("the dog runs across the field", "a spaceship explodes in orbit"),
    ]
    metrics.cider(corpus)

    assert corpus[0]["cider"] > corpus[2]["cider"]
    assert corpus[2]["cider"] == pytest.approx(0.0, abs=1e-6)


def test_cider_skips_rows_with_no_prediction():
    pairs = [_pair("a man walks", "a man walks"), _pair("something", None)]
    result = metrics.cider(pairs)

    assert result["n"] == 1
    assert "cider" not in pairs[1]


def test_cider_on_an_empty_corpus_is_zero_not_a_crash():
    assert metrics.cider([]) == {"cider": 0.0, "n": 0}


# --------------------------------------------------------------------------- #
# LLM-AD-eval
# --------------------------------------------------------------------------- #


def test_judge_prompt_is_the_papers_prompt():
    """The 0-5 scale only means what the paper means by it if the instructions
    that produced it are the paper's — in particular the two clauses that make
    this metric complementary to CRITIC rather than a second CIDEr."""
    assert "pronouns" in metrics.JUDGE_SYSTEM_PROMPT
    assert (
        "Consider different character names as valid matches"
        in metrics.JUDGE_SYSTEM_PROMPT
    )
    assert "actions, objects and interactions" in metrics.JUDGE_SYSTEM_PROMPT
    assert "between 0 and 5" in metrics.JUDGE_USER_PROMPT


class _ScriptedJudge:
    """A judge whose answer depends on the *pair*, not on call order.

    The pairs are judged concurrently, so a FIFO queue of canned responses makes
    the assertions depend on which coroutine happens to reach the client first —
    which passed alone and failed in a full suite run.
    """

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config=None):
        from tests.conftest import FakeGeminiResponse

        prompt = "".join(str(part) for part in contents)
        for needle, reply in self.answers.items():
            if needle in prompt:
                return FakeGeminiResponse(reply)
        raise AssertionError(f"no scripted answer for prompt: {prompt[:200]}")


async def test_judge_scores_every_pair():
    client = _ScriptedJudge(
        {"a man walks\n\n": '{"score": 5}', "a cat sleeps": '{"score": 1}'}
    )
    pairs = [_pair("a man walks", "a man walks"), _pair("a man walks", "a cat sleeps")]

    result = await metrics.llm_ad_eval(pairs, client=client)

    assert [p["llm_ad_eval"] for p in pairs] == [5, 1]
    assert result["llm_ad_eval"] == pytest.approx(3.0)
    assert result["histogram"] == {"0": 0, "1": 1, "2": 0, "3": 0, "4": 0, "5": 1}


async def test_judge_clamps_an_out_of_range_score():
    client = _ScriptedJudge({"Predicted": '{"score": 9}'})
    pairs = [_pair("a", "b")]

    await metrics.llm_ad_eval(pairs, client=client)

    assert pairs[0]["llm_ad_eval"] == 5


async def test_an_unparseable_judgement_is_reported_not_counted():
    client = _ScriptedJudge(
        {
            "Predicted Audio Description: b": "I'd rather not say.",
            "Predicted Audio Description: d": '{"score": 4}',
        }
    )
    pairs = [_pair("a", "b"), _pair("c", "d")]

    result = await metrics.llm_ad_eval(pairs, client=client)

    assert result["n"] == 1
    assert result["unscored"] == 1
    assert result["llm_ad_eval"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


async def test_score_predictions_writes_scores_back_onto_the_rows(
    tmp_path, monkeypatch
):
    """Per-row scores land back in the JSONL, so bad pairs can be read by hand —
    which is the more informative output when n is small."""
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        json.dumps({"video_id": "v", "gt": "a man walks", "pred": "a man walks"}) + "\n"
    )
    results_path = tmp_path / "results.json"

    real_judge = metrics.llm_ad_eval
    judge = _ScriptedJudge({"Predicted": '{"score": 3}'})
    monkeypatch.setattr(
        metrics,
        "llm_ad_eval",
        lambda pairs, client=None: real_judge(pairs, client=judge),
    )
    results = await metrics.score_predictions(preds, results_path)

    written = json.loads(preds.read_text().strip())
    assert "cider" in written and "llm_ad_eval" in written
    assert json.loads(results_path.read_text())["cider"] == results["cider"]


def test_report_names_the_papers_reference_figures():
    """A bare CIDEr number invites over-reading; the human ceiling is 69.8."""
    report = metrics.format_report(
        {
            "n": 10,
            "predictions": 10,
            "cider": 18.0,
            "llm_ad_eval": 2.4,
            "judge_model": "gemini-3.5-flash",
            "histogram": {"0": 1, "5": 2},
        }
    )

    assert "69.8" in report
    assert "3.06" in report
    assert "gemini-3.5-flash" in report


# --------------------------------------------------------------------------- #
# Recall@k/N — the discriminative metric
# --------------------------------------------------------------------------- #

bert = pytest.importorskip("bert_score", reason="install the bench extra")


def _clip_pairs(texts, preds):
    """One clip's worth of pairs, one second apart."""
    return [
        {
            "video_id": "aaaaaaaaaaa",
            "scaled_start": float(i),
            "scaled_end": float(i) + 1.0,
            "gt": gt,
            "pred": pred,
        }
        for i, (gt, pred) in enumerate(zip(texts, preds, strict=True))
    ]


REFS = [
    "a man opens the safe",
    "she climbs the staircase",
    "the dog runs across a field",
    "he lights a cigarette",
    "a car pulls up outside",
]


def test_perfect_predictions_retrieve_their_own_reference():
    result = metrics.recall_at_k(_clip_pairs(REFS, REFS))

    assert result["recall@1/5"] == 100.0
    assert result["recall_n"] == 5


def test_a_wrong_moment_description_scores_zero():
    """The whole reason for this metric: copying the *neighbouring* reference is
    fluent, on-register and completely wrong, and CIDEr gives it most of the
    credit. Retrieval gives it none."""
    neighbours = REFS[1:] + REFS[:1]
    result = metrics.recall_at_k(_clip_pairs(REFS, neighbours))

    assert result["recall@1/5"] == 0.0


def test_a_clip_with_fewer_rows_than_the_window_is_skipped():
    """Padding a short window would make retrieval easier there and the number
    incomparable between runs."""
    result = metrics.recall_at_k(_clip_pairs(REFS[:3], REFS[:3]))

    assert result["recall_n"] == 0
    assert result["recall@1/5"] is None


def test_bertscore_ranks_a_paraphrase_above_an_unrelated_line():
    pairs = [
        {"gt": "he picks up the leather boots", "pred": "he lifts the leather boots"},
        {"gt": "he picks up the leather boots", "pred": "a spaceship explodes"},
    ]
    metrics.bertscore(pairs)

    assert pairs[0]["bertscore"] > pairs[1]["bertscore"]


def test_retrieval_skips_records_with_no_timing_rather_than_raising():
    """A pair with no clip or timestamp cannot be given neighbours; scoring the
    rest of the run matters more than failing on it."""
    result = metrics.recall_at_k([{"gt": "a", "pred": "b"}])

    assert result["recall_n"] == 0
