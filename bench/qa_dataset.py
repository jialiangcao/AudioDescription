"""Loading a multiple-choice video-QA benchmark.

**Video-MME** (`lmms-lab/Video-MME`) is the one wired up. It fits this project
without adapting anything: its videos are YouTube URLs, so ``src/ingest.py``
fetches them, and its *short* split averages about two minutes — the same shape
of clip the AD pipeline already segments and analyses.

CinePile would have been the closer match on content (its clips come from the
same MovieClips channel as Condensed Movies, and it is built partly from audio
descriptions), but the dataset is **gated** on Hugging Face: it needs an access
request and a token. ``load_cinepile`` is here for when that access exists.

The task is 4-way multiple choice, so **chance is 25%** — the number every
result here has to be read against.
"""

import logging
from dataclasses import dataclass
from functools import cache
from typing import Any

logger = logging.getLogger(__name__)

VIDEOMME_REPO = "lmms-lab/Video-MME"
VIDEOMME_FILE = "videomme/test-00000-of-00001.parquet"

CINEPILE_REPO = "tomg-group-umd/cinepile"
CINEPILE_FILE = "v2/test-00000-of-00001.parquet"


@dataclass(frozen=True)
class QaItem:
    """One multiple-choice question about one video."""

    dataset: str
    question_id: str
    video_id: str
    url: str
    question: str
    options: list[str]
    answer: str  # the correct letter, "A".."E"
    task_type: str
    domain: str
    duration: str

    @property
    def prompt(self) -> str:
        """The question as the agent sees it: stem, options, and how to reply.

        The Q&A system answers in prose — it was built for a person typing into
        a box, not for a benchmark harness — so the instruction to end on a bare
        letter is what makes its output scoreable at all. ``parse_choice`` still
        falls back to matching the option text, because a multi-agent loop does
        not always honour a formatting request.
        """
        options = "\n".join(self.options)
        return (
            f"{self.question}\n\n{options}\n\n"
            "Answer with the single letter of the correct option. "
            "End your response with that letter on its own."
        )


def _parquet(repo: str, filename: str) -> Any:
    """The dataset as a DataFrame. Typed Any: pandas' stubs widen a
    boolean-masked frame to ndarray, which it is not."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset")
    return pd.read_parquet(path)


@cache
def _videomme() -> Any:
    return _parquet(VIDEOMME_REPO, VIDEOMME_FILE)


def load_videomme(duration: str = "short", limit_videos: int | None = None):
    """Video-MME items, optionally restricted to the first ``limit_videos`` videos.

    Sliced by video rather than by row: the expensive part is analysing a video,
    and every question about it then costs only the agent loop.
    """
    frame = _videomme()
    frame = frame[frame["duration"] == duration] if duration else frame
    if limit_videos:
        keep = list(dict.fromkeys(frame["videoID"]))[:limit_videos]
        frame = frame[frame["videoID"].isin(keep)]

    items = [
        QaItem(
            dataset="video-mme",
            question_id=str(row["question_id"]),
            video_id=str(row["videoID"]),
            url=str(row["url"]),
            question=str(row["question"]),
            options=[str(option) for option in row["options"]],
            answer=str(row["answer"]).strip().upper()[:1],
            task_type=str(row["task_type"]),
            domain=str(row["domain"]),
            duration=str(row["duration"]),
        )
        for _, row in frame.iterrows()
    ]
    logger.info(
        "video-mme: %d question(s) over %d video(s) (%s)",
        len(items),
        len({i.video_id for i in items}),
        duration or "all durations",
    )
    return items


def load_cinepile(limit_videos: int | None = None):
    """CinePile items. Requires Hugging Face access to the gated dataset."""
    from huggingface_hub.errors import GatedRepoError

    try:
        frame = _parquet(CINEPILE_REPO, CINEPILE_FILE)
    except GatedRepoError as exc:
        raise RuntimeError(
            "CinePile is gated: request access at "
            f"https://huggingface.co/datasets/{CINEPILE_REPO} and log in with "
            "`huggingface-cli login`."
        ) from exc

    if limit_videos:
        keep = list(dict.fromkeys(frame["yt_clip_link"]))[:limit_videos]
        frame = frame[frame["yt_clip_link"].isin(keep)]

    from ingest import video_id as youtube_id

    items = []
    for i, (_, row) in enumerate(frame.iterrows()):
        vid = youtube_id(str(row["yt_clip_link"]))
        if vid is None:
            continue
        # CinePile stores the answer's index; the letters are ours to assign.
        letters = "ABCDE"
        options = [f"{letters[j]}. {choice}" for j, choice in enumerate(row["choices"])]
        items.append(
            QaItem(
                dataset="cinepile",
                question_id=f"{vid}-{i}",
                video_id=vid,
                url=str(row["yt_clip_link"]),
                question=str(row["question"]),
                options=options,
                answer=letters[int(str(row["answer_key_position"]))],
                task_type=str(row.get("question_category", "")),
                domain=str(row.get("movie_name", "")),
                duration="short",
            )
        )
    logger.info("cinepile: %d question(s)", len(items))
    return items


def group_by_video(items: list[QaItem]) -> dict[str, list[QaItem]]:
    grouped: dict[str, list[QaItem]] = {}
    for item in items:
        grouped.setdefault(item.video_id, []).append(item)
    return grouped
