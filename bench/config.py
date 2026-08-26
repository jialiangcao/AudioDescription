"""Knobs for the benchmark harness."""

import os
from pathlib import Path

# Everything the harness reads and writes lives under here by default, well
# away from the job scratch the real workers use.
DATA_ROOT = Path(os.environ.get("BENCH_DATA_ROOT", "data"))
CLIPS_DIR = DATA_ROOT / "clips"
WORK_DIR = DATA_ROOT / "work"
PREDS_PATH = DATA_ROOT / "preds.jsonl"
RESULTS_PATH = DATA_ROOT / "results.json"

# Frame sampling interval, in seconds. This is the harness's main cost dial:
# stage 2 is one Gemini call per sampled frame, so a 2-minute clip at 1.0s is
# ~120 vision calls. Raise it to survey more clips for the same spend.
INTERVAL_SEC = float(os.environ.get("BENCH_INTERVAL_SEC", "1.0"))

# How far either side of a ground-truth AD window to look for sampled frames.
# A 2-second window against 1-second sampling can otherwise catch nothing, and
# a describer is reacting to what surrounds the gap as much as what is in it.
FRAME_PAD_SEC = float(os.environ.get("BENCH_FRAME_PAD_SEC", "2.0"))

# The LLM-AD-eval judge. Kept separate from vision_analysis.MODEL so the judge
# can be pointed at a different model than the one that wrote the line — a model
# scoring its own output has a well-documented self-preference bias.
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "gemini-3.5-flash")

# Concurrent judge calls. Pure network I/O, and every call still passes through
# the shared gemini_limits token bucket.
JUDGE_CONCURRENCY = int(os.environ.get("BENCH_JUDGE_CONCURRENCY", "8"))
