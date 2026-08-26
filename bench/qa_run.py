"""``bench qa-run`` — the multi-agent Q&A system against a video-QA benchmark.

Three modes, because an accuracy number on a video benchmark means nothing on
its own. Many VideoQA questions are answerable from language priors alone, so
the controls are the point:

``blind``      the question and its options, no video at all. Whatever this
               scores is the share the agent gets for free from world knowledge
               and answer-option phrasing, and the floor any video system must
               clear to have demonstrated it watched anything.
``subtitles``  the question plus the transcript the pipeline extracted. Isolates
               how much is carried by dialogue rather than pixels.
``agent``      the real thing: ``qa.answer_question`` over a full Timeline, with
               CLIP retrieval, frame inspection and the planner loop.

Each video is prepared once (segmentation, per-frame vision, VAD, transcript)
and reused across its questions, so the marginal cost of a question is the agent
loop alone. Preparation is cached in the same scratch layout as the AD harness.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from bench.config import FRAME_PAD_SEC, INTERVAL_SEC  # noqa: F401
from bench.fetch import clip_path
from bench.qa_dataset import QaItem, group_by_video
from bench.run import _describe_frames, _prepare_clip, clip_blobs
from ingest import IngestError, download_youtube
from timeline import build_timeline

logger = logging.getLogger(__name__)

MODES = ("agent", "blind", "subtitles")

# A letter on its own, or the last letter the model committed to. Checked in
# order; the option-text fallback below catches everything these miss.
_PATTERNS = (
    re.compile(r"(?:^|\n)\s*\(?([A-E])[\.\)]?\s*$", re.MULTILINE),
    re.compile(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-E])\b", re.IGNORECASE),
    re.compile(r"\*\*\s*\(?([A-E])[\.\)]?\s*\*\*"),
)


def parse_choice(answer: str | None, options: list[str]) -> str | None:
    """The letter the agent settled on, or None if it never committed to one."""
    if not answer:
        return None
    text = answer.strip()
    for pattern in _PATTERNS:
        found = pattern.findall(text)
        if found:
            return found[-1].upper()

    # No letter anywhere: fall back to whichever option's *text* the answer
    # overlaps most. A prose answer that names the right thing without labelling
    # it is correct, and scoring it wrong would measure instruction-following.
    lowered = text.lower()
    best, best_score = None, 0.0
    for option in options:
        letter, _, body = option.partition(".")
        body = body.strip().lower().rstrip(".")
        if not body:
            continue
        if body in lowered:
            score = 1.0 + len(body) / 100
        else:
            words = set(re.findall(r"[a-z0-9']+", body))
            hit = words & set(re.findall(r"[a-z0-9']+", lowered))
            score = len(hit) / len(words) if words else 0.0
        if score > best_score:
            best, best_score = letter.strip().upper()[:1], score
    # Below this the "match" is generic words like "the" and means nothing.
    return best if best_score >= 0.5 else None


async def _prepare_video(
    video_id: str,
    url: str,
    clips_dir: Path,
    work_dir: Path,
    interval_sec: float,
    client,
):
    """Download and fully analyse one video, returning a Timeline for the agent."""
    video_path = clip_path(clips_dir, video_id)
    if not video_path.exists():
        await asyncio.to_thread(download_youtube, url, video_path)

    blobs = clip_blobs(video_id, work_dir)
    shots, meta, regions, transcript = await asyncio.to_thread(
        _prepare_clip, video_path, blobs, interval_sec
    )
    shots = await _describe_frames(shots, blobs, client)
    # The real timeline, not the AD harness's ground-truth-anchored one: the Q&A
    # agents navigate by shot and retrieve over frames, so they need the
    # pipeline's own segmentation.
    # build_timeline constructs Frame(**frame), so the per-frame `visual` dicts
    # written by the vision stage are carried onto the model as they are.
    timeline = build_timeline(
        video_id, meta["duration_sec"], shots, regions, transcript
    )
    return timeline, blobs


async def _answer_blind(item: QaItem, client) -> str | None:
    """No video: the question and options, nothing else."""
    from google.genai import types

    import gemini_limits
    from qa.config import TEXT_MODEL

    response = await gemini_limits.with_retries(
        lambda: client.aio.models.generate_content(
            model=TEXT_MODEL,
            contents=[item.prompt],
            config=types.GenerateContentConfig(temperature=0),
        )
    )
    return response.text


async def _answer_from_subtitles(item: QaItem, timeline, client) -> str | None:
    """Dialogue only: what a transcript-reading system could know."""
    from google.genai import types

    import gemini_limits
    from qa.config import TEXT_MODEL

    lines = [
        f"[{seg.start:.0f}s] {seg.audio.transcript}"
        for seg in timeline.segments
        if seg.audio and seg.audio.transcript
    ]
    transcript = "\n".join(lines) or "(no speech was detected in this video)"
    response = await gemini_limits.with_retries(
        lambda: client.aio.models.generate_content(
            model=TEXT_MODEL,
            contents=[f"Transcript of a video:\n{transcript}\n\n{item.prompt}"],
            config=types.GenerateContentConfig(temperature=0),
        )
    )
    return response.text


async def run_benchmark(
    items: list[QaItem],
    clips_dir: Path,
    work_dir: Path,
    preds_path: Path,
    mode: str = "agent",
    interval_sec: float = INTERVAL_SEC,
    force: bool = False,
) -> dict:
    """Answer every item, appending one record per question to ``preds_path``."""
    from google import genai

    import qa

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    preds_path = Path(preds_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    if force and preds_path.exists():
        preds_path.unlink()
    done = set()
    if preds_path.exists():
        with open(preds_path, encoding="utf-8") as handle:
            done = {json.loads(line)["question_id"] for line in handle if line.strip()}

    client = genai.Client()
    by_video = group_by_video([i for i in items if i.question_id not in done])
    answered, failed = 0, 0

    with open(preds_path, "a", encoding="utf-8") as out:
        for n, (video_id, questions) in enumerate(by_video.items(), 1):
            logger.info(
                "qa %d/%d: %s (%d question(s), mode=%s)",
                n,
                len(by_video),
                video_id,
                len(questions),
                mode,
            )
            timeline, blobs = None, None
            if mode != "blind":
                try:
                    timeline, blobs = await _prepare_video(
                        video_id,
                        questions[0].url,
                        clips_dir,
                        work_dir,
                        interval_sec,
                        client,
                    )
                except (IngestError, OSError, RuntimeError, ValueError) as exc:
                    # A dead YouTube link or a broken clip costs its questions,
                    # not the run; they are recorded as unanswered so the
                    # denominator stays honest.
                    failed += len(questions)
                    logger.warning("qa: %s unusable (%r)", video_id, exc)
                    for item in questions:
                        out.write(
                            json.dumps(
                                _record(item, mode, None, None, 0.0, unavailable=True)
                            )
                            + "\n"
                        )
                    out.flush()
                    continue

            for item in questions:
                t0 = time.monotonic()
                try:
                    if mode == "blind":
                        raw = await _answer_blind(item, client)
                        cycles = 0
                    elif mode == "subtitles":
                        raw = await _answer_from_subtitles(item, timeline, client)
                        cycles = 0
                    else:
                        assert timeline is not None  # set for every non-blind mode
                        result = await qa.answer_question(
                            timeline, item.prompt, client, blobs
                        )
                        raw, cycles = result.answer, result.cycles
                except Exception:
                    failed += 1
                    logger.exception("qa: question %s failed", item.question_id)
                    raw, cycles = None, 0
                out.write(
                    json.dumps(_record(item, mode, raw, cycles, time.monotonic() - t0))
                    + "\n"
                )
                out.flush()
                answered += 1

    summary = {
        "mode": mode,
        "answered": answered,
        "failed": failed,
        "videos": len(by_video),
        "preds_path": str(preds_path),
    }
    logger.info("qa: %d answered, %d failed -> %s", answered, failed, preds_path)
    return summary


def _record(
    item: QaItem,
    mode: str,
    raw: str | None,
    cycles,
    elapsed: float,
    unavailable: bool = False,
) -> dict:
    predicted = parse_choice(raw, item.options)
    return {
        # A question whose video could not be fetched at all is not a wrong
        # answer — it is a missing measurement. Scoring it wrong would penalise
        # the video modes against the blind control, which needs no video, and
        # make the two incomparable. The report excludes these from accuracy and
        # counts them separately.
        "video_unavailable": unavailable,
        "dataset": item.dataset,
        "mode": mode,
        "question_id": item.question_id,
        "video_id": item.video_id,
        "task_type": item.task_type,
        "domain": item.domain,
        "duration": item.duration,
        "question": item.question,
        "options": item.options,
        "answer": item.answer,
        "predicted": predicted,
        "correct": predicted == item.answer,
        "raw": (raw or "")[:2000],
        "cycles": cycles,
        "seconds": round(elapsed, 1),
    }
