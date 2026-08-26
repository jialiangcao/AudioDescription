"""Reading the CMD-AD CSV.

One row is one human-written audio description. The columns that matter:

``text``                    the reference AD line — what we are trying to match.
``cmd_filename``            ``<year>/<youtube_id>``; the id addresses the clip.
``scaled_start/scaled_end`` the AD's window **on the YouTube clip's own
                            timeline**. These are what the harness uses.
``audiovault_start/end``    the same AD on the *full movie's* AudioVault audio
                            (~90 min). The paper maps one onto the other with a
                            per-movie ``W·t + B`` fit (RANSAC over
                            mel-spectrogram correlations; ``W`` ≠ 1 because NTSC
                            and PAL releases run at different speeds). We
                            already have the mapped values, so these are carried
                            for provenance and never used in scoring.
``duration``                ``scaled_end - scaled_start`` — the time the human
                            describer had to speak in, and therefore the word
                            budget our line is held to.
``imdbid``/``movie_title``  identity, for reporting and for a future CRITIC.
``cmd_clip_idx``            which of the movie's ~10 clips this is. Note it is
                            *not* a key: ``cmd_filename`` already identifies the
                            clip uniquely across movies.
``split``                   ``train`` / ``test``.
"""

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdRow:
    text: str
    cmd_filename: str
    video_id: str
    scaled_start: float
    scaled_end: float
    duration: float
    imdbid: str
    movie_title: str
    cmd_clip_idx: int
    split: str

    @classmethod
    def from_csv(cls, row: dict) -> "AdRow":
        cmd_filename = row["cmd_filename"]
        start = float(row["scaled_start"])
        end = float(row["scaled_end"])
        return cls(
            text=row["text"].strip(),
            cmd_filename=cmd_filename,
            # Everything after the "/" is the YouTube id: "2011/_SQr8I3lcW8".
            video_id=cmd_filename.rsplit("/", 1)[-1],
            scaled_start=start,
            scaled_end=end,
            # Prefer the column, but fall back to the span — a handful of rows
            # in the published CSV carry one without the other.
            duration=float(row["duration"]) if row.get("duration") else end - start,
            imdbid=row.get("imdbid", ""),
            movie_title=row.get("movie_title", ""),
            cmd_clip_idx=int(row["cmd_clip_idx"]) if row.get("cmd_clip_idx") else -1,
            split=row.get("split", ""),
        )


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def load_rows(csv_path: Path | str, split: str | None = None) -> list[AdRow]:
    """Every AD row in the CSV, optionally restricted to one split."""
    rows: list[AdRow] = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                row = AdRow.from_csv(raw)
            except (KeyError, ValueError):
                skipped += 1
                continue
            if split and row.split != split:
                continue
            # A zero- or negative-length window has no gap to narrate into and
            # would give the model a word budget of zero.
            if row.duration <= 0:
                skipped += 1
                continue
            rows.append(row)
    logger.info(
        "dataset: %d row(s) from %s%s%s",
        len(rows),
        csv_path,
        f" (split={split})" if split else "",
        f", {skipped} skipped" if skipped else "",
    )
    return rows


def group_by_clip(rows: list[AdRow]) -> dict[str, list[AdRow]]:
    """Rows grouped by YouTube id, each group in temporal order.

    A clip is the unit of work: it is one download, one pipeline run, and one
    continuous narration context, so every AD inside it is handled together.
    Ordering matters — ``fill_narration_gaps`` feeds each line the preceding
    ones as continuity context, which is only meaningful in time order.
    """
    grouped: dict[str, list[AdRow]] = defaultdict(list)
    for row in rows:
        grouped[row.video_id].append(row)
    return {
        video_id: sorted(clip_rows, key=lambda r: r.scaled_start)
        for video_id, clip_rows in grouped.items()
    }


def spread_across_movies(clips: dict[str, list[AdRow]]) -> list[str]:
    """Clip ids ordered so that taking the first N covers as many movies as possible.

    The CSV is ordered by movie, so slicing it directly — which is what the
    harness did at first — makes ``--limit 15`` mean "the first two movies".
    That is a terrible eval sample: CIDEr's IDF ends up estimated over two
    films' vocabulary, and per-clip scores vary enough within one movie that
    two of them average out nothing.

    Round-robin by ``imdbid`` instead: one clip from each movie, then a second
    from each, and so on. A prefix of this list is always the widest sample of
    movies available at that size.
    """
    by_movie: dict[str, list[str]] = defaultdict(list)
    for video_id, rows in clips.items():
        by_movie[rows[0].imdbid].append(video_id)

    ordered: list[str] = []
    for depth in range(max((len(q) for q in by_movie.values()), default=0)):
        ordered.extend(
            queue[depth] for queue in by_movie.values() if depth < len(queue)
        )
    logger.info(
        "dataset: %d clip(s) across %d movie(s), ordered round-robin",
        len(ordered),
        len(by_movie),
    )
    return ordered
