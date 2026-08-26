"""``python -m bench`` — the benchmark CLI.

uv run python -m bench fetch --csv data/cmd_ad.csv --limit 3
uv run python -m bench run   --csv data/cmd_ad.csv
uv run python -m bench score
"""

import argparse
import asyncio
import json
import logging
import sys

from dotenv import load_dotenv

from bench.config import (
    CLIPS_DIR,
    DATA_ROOT,
    FRAME_PAD_SEC,
    INTERVAL_SEC,
    PREDS_PATH,
    RESULTS_PATH,
    WORK_DIR,
)

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    parser.add_argument(
        "--log-level", default=None, help="override ADESC_LOG_LEVEL for this run"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download the CMD clips named by the CSV")
    fetch.add_argument("--csv", required=True)
    fetch.add_argument("--clips-dir", default=CLIPS_DIR)
    fetch.add_argument("--split", default=None, help="e.g. train, test")
    fetch.add_argument("--limit", type=int, default=None, help="distinct clips")
    fetch.add_argument("--max-height", type=int, default=720)

    run = sub.add_parser("run", help="generate one AD line per ground-truth row")
    run.add_argument("--csv", required=True)
    run.add_argument("--clips-dir", default=CLIPS_DIR)
    run.add_argument("--work-dir", default=WORK_DIR)
    run.add_argument("--out", default=PREDS_PATH, help="predictions JSONL")
    run.add_argument("--split", default=None)
    run.add_argument("--limit", type=int, default=None, help="clips to run")
    run.add_argument(
        "--interval-sec",
        type=float,
        default=INTERVAL_SEC,
        help="frame sampling interval; the cost dial (one vision call per frame)",
    )
    run.add_argument("--frame-pad-sec", type=float, default=FRAME_PAD_SEC)
    run.add_argument(
        "--words-per-sec",
        type=float,
        default=None,
        help="narration word budget per second of gap; default timeline.NARRATION_WORDS_PER_SEC",
    )
    run.add_argument(
        "--force", action="store_true", help="discard existing predictions and redo"
    )

    report = sub.add_parser("report", help="every scored pair + summary stats, as HTML")
    report.add_argument("--preds", default=PREDS_PATH)
    report.add_argument("--results", default=RESULTS_PATH)
    report.add_argument("--out", default=DATA_ROOT / "report.html")

    qa_run = sub.add_parser(
        "qa-run", help="run the Q&A agent against a video-QA benchmark"
    )
    qa_run.add_argument(
        "--dataset", default="video-mme", choices=["video-mme", "cinepile"]
    )
    qa_run.add_argument(
        "--duration", default="short", help="video-mme: short|medium|long"
    )
    qa_run.add_argument("--videos", type=int, default=10, help="videos to cover")
    qa_run.add_argument(
        "--mode",
        default="agent",
        choices=["agent", "blind", "subtitles"],
        help="agent = the real system; blind/subtitles are the controls",
    )
    qa_run.add_argument("--clips-dir", default=DATA_ROOT / "qa_clips")
    qa_run.add_argument("--work-dir", default=DATA_ROOT / "qa_work")
    qa_run.add_argument(
        "--out", default=None, help="default: data/qa_<mode>_preds.jsonl"
    )
    qa_run.add_argument("--interval-sec", type=float, default=INTERVAL_SEC)
    qa_run.add_argument("--force", action="store_true")

    qa_report = sub.add_parser(
        "qa-report", help="accuracy, controls and every question"
    )
    qa_report.add_argument(
        "--preds", nargs="+", required=True, help="one file per mode"
    )
    qa_report.add_argument("--out", default=DATA_ROOT / "qa_report.html")

    score = sub.add_parser("score", help="CIDEr + LLM-AD-eval over the predictions")
    score.add_argument("--preds", default=PREDS_PATH)
    score.add_argument("--out", default=RESULTS_PATH)
    score.add_argument(
        "--no-judge", action="store_true", help="skip the LLM judge; no Gemini calls"
    )
    score.add_argument(
        "--no-retrieval",
        action="store_true",
        help="skip Recall@1/5 and BERTScore, which load roberta-large locally",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)

    from log_config import configure_logging

    if args.log_level:
        import os

        os.environ["ADESC_LOG_LEVEL"] = args.log_level
    configure_logging()

    # The harness runs against no Redis by design, so the shared Gemini token
    # bucket falls open on every single call and says so — with a traceback.
    # That is the right behaviour on a worker, where it means an outage, and
    # pure noise here, where it means "as configured". Say it once ourselves.
    logger.info(
        "bench: no Redis in this process — Gemini calls are not rate-limited "
        "across machines; concurrency is whatever this run sets"
    )
    logging.getLogger("gemini_limits").setLevel(logging.ERROR)

    if args.command == "fetch":
        from bench.fetch import fetch_clips

        summary = fetch_clips(
            args.csv,
            args.clips_dir,
            split=args.split,
            limit=args.limit,
            max_height=args.max_height,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "run":
        from bench.run import run_benchmark

        summary = asyncio.run(
            run_benchmark(
                args.csv,
                args.clips_dir,
                args.work_dir,
                args.out,
                split=args.split,
                limit=args.limit,
                interval_sec=args.interval_sec,
                pad_sec=args.frame_pad_sec,
                words_per_sec=args.words_per_sec,
                force=args.force,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qa-run":
        from bench.qa_dataset import load_cinepile, load_videomme
        from bench.qa_run import run_benchmark as qa_run_benchmark

        items = (
            load_videomme(args.duration, limit_videos=args.videos)
            if args.dataset == "video-mme"
            else load_cinepile(limit_videos=args.videos)
        )
        out = args.out or (DATA_ROOT / f"qa_{args.mode}_preds.jsonl")
        summary = asyncio.run(
            qa_run_benchmark(
                items,
                args.clips_dir,
                args.work_dir,
                out,
                mode=args.mode,
                interval_sec=args.interval_sec,
                force=args.force,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qa-report":
        from bench.qa_report import build_report as build_qa_report
        from bench.qa_report import common_questions, load_runs, summarize_run

        path = build_qa_report(args.preds, args.out)
        runs = load_runs(args.preds)
        # The same restriction the report applies: every mode scored over the
        # questions all of them measured. Without it this printout and the
        # report disagree, and the mode that lost videos looks worse for it.
        shared = common_questions(runs) if len(runs) > 1 else None
        for mode, records in runs.items():
            s_ = summarize_run(records, only=shared)
            print(
                f"  {mode:12} {s_['accuracy']:5.1f}%  "
                f"CI [{s_['ci'][0]}, {s_['ci'][1]}]  "
                f"n={s_['questions']}  chance={s_['chance']}%"
            )
        print(f"wrote {path}")
        return 0

    if args.command == "report":
        from bench.report import build_report

        path = build_report(args.preds, args.results, args.out)
        print(f"wrote {path}")
        return 0

    from bench.metrics import format_report, score_predictions

    results = asyncio.run(
        score_predictions(
            args.preds,
            args.out,
            judge=not args.no_judge,
            retrieval=not args.no_retrieval,
        )
    )
    print(format_report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
