"""Scoring predicted AD against the CMD-AD reference text.

Two metrics from *AutoAD III* (arXiv 2404.14412, §6.2 and Appendix B.2):

**CIDEr** — TF-IDF-weighted n-gram agreement, the field's usual caption metric.
Reference points on CMD-AD-Eval: AutoAD-III 25.0, StrAD-FT 36.3 (the 2026 state
of the art) — while **two human describers watching the same frames only agree
at 69.8**. That is the ceiling, and it is why a middling CIDEr here says less
than reading a dozen pairs by hand.

CIDEr is also the weakest metric here, measurably: on our own corpus a line
copied from the *neighbouring* reference AD — right register, wrong moment —
scores 30.4 against a real run's 38.2. Prefer ``recall_at_k`` below.

**LLM-AD-eval** — an LLM judge scoring the match 0-5, using the paper's prompt
verbatim. It exists because n-gram overlap punishes a correct description
written in different words, which is most of them.

Two deliberate departures from the reference implementation, both noted in the
report so nobody reads these as the paper's numbers:

- the judge is Gemini, not ``gpt-3.5-turbo``, so absolute values are not
  comparable to the paper's table — and a Gemini judge scoring Gemini-written
  lines carries a self-preference bias. Use it to compare *our* runs.
- tokenization is done here rather than by ``PTBTokenizer``, which shells out to
  Java. The effect on CIDEr is small; the dependency is not.
"""

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import gemini_limits
from bench.config import JUDGE_CONCURRENCY, JUDGE_MODEL

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def tokenize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_PUNCT_RE.sub(" ", (text or "").lower()).split())


# --------------------------------------------------------------------------- #
# CIDEr
# --------------------------------------------------------------------------- #


def cider(pairs: list[dict]) -> dict:
    """Corpus-level CIDEr-D over ``{"gt": ..., "pred": ...}`` records, ×100.

    The scorer returns the conventional 0-1-ish figure; papers report it ×100,
    so we do too, to be directly readable against the paper's table.

    CIDEr's IDF weights are estimated from the reference corpus *passed in*, so
    a score is only comparable against another run over the same rows — which is
    why the report prints ``n`` next to it.
    """
    from pycocoevalcap.cider.cider import Cider

    scored = [p for p in pairs if p.get("pred") and p.get("gt")]
    if not scored:
        return {"cider": 0.0, "n": 0}

    gts = {i: [tokenize(p["gt"])] for i, p in enumerate(scored)}
    res = {i: [tokenize(p["pred"])] for i, p in enumerate(scored)}
    score, per_item = Cider().compute_score(gts, res)
    for pair, item in zip(scored, per_item, strict=True):
        pair["cider"] = round(float(item) * 100, 2)
    return {"cider": round(float(score) * 100, 2), "n": len(scored)}


# --------------------------------------------------------------------------- #
# LLM-AD-eval
# --------------------------------------------------------------------------- #

# Verbatim from Appendix B.2, Algorithm 2. Kept word for word — the score scale
# is only meaningful against the instructions that produced it, and the "ignore
# character names, count pronouns as matches" clauses are what make this metric
# complementary to CRITIC rather than a second CIDEr.
JUDGE_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the quality of "
    "generative outputs for movie audio descriptions. "
    "Your task is to compare the predicted audio descriptions with the correct "
    "audio descriptions and determine its level of match, considering mainly "
    "the visual elements like actions, objects and interactions. Here's how you "
    "can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Check if the predicted audio description covers the main visual events "
    "from the movie, especially focusing on the verbs and nouns.\n"
    "- Evaluate whether the predicted audio description includes specific "
    "details rather than just generic points. It should provide comprehensive "
    "information that is tied to specific elements of the video.\n"
    "- Consider synonyms or paraphrases as valid matches. Consider pronouns "
    "like 'he' or 'she' as valid matches with character names. Consider "
    "different character names as valid matches. \n"
    "- Provide a single evaluation score that reflects the level of match of "
    "the prediction, considering the visual elements like actions, objects and "
    "interactions."
)

JUDGE_USER_PROMPT = (
    "Please evaluate the following movie audio description pair:\n\n"
    "Correct Audio Description: {gt}\n"
    "Predicted Audio Description: {pred}\n\n"
    "Provide your evaluation only as a matching score where the matching score "
    "is an integer value between 0 and 5, with 5 indicating the highest level "
    "of match."
)


async def _judge_one(client, gt: str, pred: str) -> int | None:
    from google.genai import types

    response = await gemini_limits.with_retries(
        lambda: client.aio.models.generate_content(
            model=JUDGE_MODEL,
            contents=[JUDGE_USER_PROMPT.format(gt=gt, pred=pred)],
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                # The paper's script asks for a Python dict string and runs
                # ast.literal_eval over it. Structured output gets the same
                # {"score": int} without parsing free text.
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {"score": {"type": "integer"}},
                    "required": ["score"],
                },
                temperature=0,
            ),
        )
    )
    if response.text is None:
        return None
    try:
        return max(0, min(5, int(json.loads(response.text)["score"])))
    except (ValueError, KeyError, TypeError):
        logger.warning("judge returned an unusable score: %r", response.text)
        return None


async def llm_ad_eval(pairs: list[dict], client=None) -> dict:
    """Score every pair 0-5 with the judge. Writes ``llm_ad_eval`` onto each pair."""
    from google import genai

    client = client or genai.Client()
    scored = [p for p in pairs if p.get("pred") and p.get("gt")]
    if not scored:
        return {"llm_ad_eval": 0.0, "n": 0, "histogram": {}}

    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def _run(pair: dict) -> None:
        async with semaphore:
            pair["llm_ad_eval"] = await _judge_one(client, pair["gt"], pair["pred"])

    await asyncio.gather(*(_run(pair) for pair in scored))

    scores = [p["llm_ad_eval"] for p in scored if p.get("llm_ad_eval") is not None]
    histogram = Counter(scores)
    return {
        "llm_ad_eval": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "n": len(scores),
        "unscored": len(scored) - len(scores),
        # The mean alone hides what this distribution usually looks like: a
        # bimodal "nailed it / described a different moment entirely", not a
        # bell curve around the average.
        "histogram": {str(k): histogram.get(k, 0) for k in range(6)},
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def load_preds(preds_path: Path) -> list[dict]:
    with open(preds_path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_preds(preds_path: Path, pairs: list[dict]) -> None:
    """Rewrite the predictions file with per-row scores attached."""
    with open(preds_path, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")


async def score_predictions(
    preds_path: Path,
    results_path: Path,
    judge: bool = True,
    retrieval: bool = True,
):
    pairs = load_preds(preds_path)
    results = {
        "predictions": len(pairs),
        "missing_prediction": sum(1 for p in pairs if not p.get("pred")),
        "judge_model": JUDGE_MODEL if judge else None,
        **cider(pairs),
    }
    if retrieval:
        # The discriminative one. Kept on by default because it is the only
        # metric here that a wrong-moment description cannot pass: see the
        # module docstring above `recall_at_k`.
        results |= recall_at_k(pairs)
        results |= bertscore(pairs)
    if judge:
        results |= await llm_ad_eval(pairs)

    write_preds(preds_path, pairs)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(results, indent=2))
    return results


def format_report(results: dict) -> str:
    lines = [
        "",
        f"  rows scored          {results.get('n', 0)}  "
        f"(of {results.get('predictions', 0)} predictions)",
        f"  CIDEr                {results.get('cider', 0.0):.1f}"
        "     [AutoAD-Zero 17.7 | AutoAD-III 25.0 | Shot-by-shot 26.3 | StrAD-FT 36.3 | human 69.8]",
    ]
    if results.get("recall@1/5") is not None:
        lines += [
            f"  Recall@1/5           {results['recall@1/5']:.1f}"
            "     [chance 20.0 | AutoAD-Zero 26.9 | AutoAD-III 31.2 | StrAD-FT 38.0 | human 80.4]",
            f"  BERTScore F1         {results.get('bertscore', 0.0):.1f}",
        ]
    if results.get("judge_model"):
        lines += [
            f"  LLM-AD-eval          {results.get('llm_ad_eval', 0.0):.2f}"
            "     [AutoAD-II 1.53 | AutoAD-III 2.05 | human 3.06]",
            f"  judge                {results['judge_model']}",
            "  score histogram      "
            + "  ".join(
                f"{k}:{v}" for k, v in (results.get("histogram") or {}).items()
            ),
            "",
            "  Published figures are over the full 7,316-pair CMD-AD-Eval split;",
            "  a subset measured here carries an interval wide enough to overlap",
            "  several of them at once. 'human' is inter-rater agreement between",
            "  two professional describers (AutoAD-III Table 3, tIoU 0.9) — the",
            "  real ceiling, not 100. LLM-AD-eval is judge-relative and CIDEr is",
            "  corpus-relative, so both are run-to-run comparisons only.",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# beyond CIDEr
# --------------------------------------------------------------------------- #
#
# CIDEr's weakness on this task is measurable: on our own corpus, copying the
# *neighbouring* reference AD — describing the wrong moment entirely, with text
# it should never see — scores 30.4 against the pipeline's 38.2. A metric worth
# switching to has to separate those two. The three below attack it from
# different directions:
#
#   BERTScore   paraphrase-tolerant semantic similarity to the reference, so
#               "retrieves the boots" and "picks up the boots" stop being misses.
#   Recall@k/N  AutoAD-II's discriminative metric: can the prediction pick its
#               OWN reference out of its temporal neighbours? This is the direct
#               answer to neighbour-copying, which by construction scores 0.
#   CLIPScore   reference-free (Hessel et al. 2021) — similarity between the
#               predicted line and the actual frames. With one reference per
#               window and a 69.8 human ceiling, the reference is a weak
#               target; the pixels are the ground truth.


_SCORER = None


def _bertscorer():
    """The shared BERTScore model. roberta-large, on this machine's GPU."""
    global _SCORER
    if _SCORER is not None:
        return _SCORER
    from bert_score import BERTScorer

    device = None
    try:
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    except ImportError:
        pass
    logger.info("loading BERTScore (roberta-large) on %s", device or "default")
    _SCORER = BERTScorer(lang="en", rescale_with_baseline=True, device=device)
    return _SCORER


def bertscore(pairs: list[dict]) -> dict:
    """Mean BERTScore F1 (baseline-rescaled), ×100. Writes onto each pair."""
    scored = [p for p in pairs if p.get("pred") and p.get("gt")]
    if not scored:
        return {"bertscore": 0.0}
    # bert_score types score() loosely (its return widens to include str); the
    # concrete return for a plain call is the (P, R, F1) tensor triple.
    f1 = cast(
        "Any",
        _bertscorer().score([p["pred"] for p in scored], [p["gt"] for p in scored]),
    )[2]
    for pair, value in zip(scored, f1.tolist(), strict=True):
        pair["bertscore"] = round(value * 100, 2)
    return {"bertscore": round(float(f1.mean()) * 100, 2)}


def recall_at_k(pairs: list[dict], k: int = 1, n: int = 5) -> dict:
    """AutoAD-II's Recall@k/N: does the prediction retrieve its own reference?

    For each prediction, rank the ``n`` temporally neighbouring reference ADs
    from the same clip by similarity and check whether the true one lands in the
    top ``k``. Unlike every similarity metric, this is *discriminative*: a line
    that describes the neighbouring moment scores zero rather than most of the
    credit, which is exactly the failure CIDEr cannot see.
    """
    # Needs to know which clip a line belongs to and where in it — a record
    # without that cannot be given neighbours, so it sits out rather than
    # bringing the whole scoring pass down.
    scored = [
        p
        for p in pairs
        if p.get("pred") and p.get("gt") and p.get("video_id") and "scaled_start" in p
    ]
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for pair in scored:
        by_clip[pair["video_id"]].append(pair)

    cands: list[str] = []
    refs: list[str] = []
    spans: list[tuple[int, int, int]] = []  # start, count, index of the true ref
    windowed: list[dict] = []  # the pair each span belongs to
    for clip in by_clip.values():
        clip.sort(key=lambda p: p["scaled_start"])
        if len(clip) < n:
            # Fewer neighbours than the window would make the choice easier and
            # the number incomparable, so these are skipped rather than padded.
            continue
        for i, pair in enumerate(clip):
            lo = max(0, min(i - n // 2, len(clip) - n))
            window = clip[lo : lo + n]
            spans.append((len(cands), len(window), i - lo))
            windowed.append(pair)
            cands.extend([pair["pred"]] * len(window))
            refs.extend(w["gt"] for w in window)

    if not spans:
        return {f"recall@{k}/{n}": None, "recall_n": 0}

    f1 = cast("Any", _bertscorer().score(cands, refs))[2]
    flat = f1.tolist()
    hits = 0
    for (start, count, true_i), pair in zip(spans, windowed, strict=True):
        window = flat[start : start + count]
        ranked = sorted(range(count), key=lambda j: window[j], reverse=True)
        # Recorded per pair: the metric is a mean over these, so a confidence
        # interval is one resample away, and the report can show which lines
        # were retrievable at all.
        pair["recall_hit"] = true_i in ranked[:k]
        hits += pair["recall_hit"]
    return {
        f"recall@{k}/{n}": round(100 * hits / len(spans), 1),
        "recall_n": len(spans),
    }


def clipscore(pairs: list[dict], work_dir) -> dict:
    """Reference-free CLIPScore: the predicted line against the frames it describes.

    ``2.5 · max(cos(text, image), 0)`` per Hessel et al., against the mean
    embedding of the frames inside the reference window. Needs the run's scratch
    directory, since that is where the sampled frames live.
    """
    import json as _json

    from bench.anchored import frames_for_window
    from bench.run import clip_blobs
    from qa.retriever import ClipFrameRetriever

    scored = [p for p in pairs if p.get("pred")]
    retriever = ClipFrameRetriever()
    shots_cache: dict[str, list[dict]] = {}
    values: list[float] = []

    for pair in scored:
        clip = pair["video_id"]
        if clip not in shots_cache:
            blobs = clip_blobs(clip, work_dir)
            try:
                doc = _json.loads(blobs.path("state/shots.json").read_text())
            except (FileNotFoundError, ValueError):
                shots_cache[clip] = []
            else:
                shots_cache[clip] = sorted(
                    (f for shot in doc["shots"] for f in shot["frames"]),
                    key=lambda f: f["time"],
                )
        frames = frames_for_window(
            shots_cache[clip], pair["scaled_start"], pair["scaled_end"]
        )
        paths = [
            str(clip_blobs(clip, work_dir).path(f["key"]))
            for f in frames
            if clip_blobs(clip, work_dir).path(f["key"]).exists()
        ]
        if not paths:
            continue
        image = retriever.encode_images(paths).mean(dim=0, keepdim=True)
        image = image / image.norm(dim=-1, keepdim=True)
        text = retriever.encode_text([pair["pred"]])
        value = 2.5 * max(float((image @ text.T).item()), 0.0)
        pair["clipscore"] = round(value * 100, 2)
        values.append(value)

    return {
        "clipscore": round(100 * sum(values) / len(values), 2) if values else 0.0,
        "clipscore_n": len(values),
    }
