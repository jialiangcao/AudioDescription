# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`adesc` generates audio-description (AD) narration for video: it detects shots, describes each
shot visually, transcribes dialogue, finds speech-free gaps long enough to narrate, and asks
Gemini to write a narration line sized to fit each gap. It also features interactive Q&A.

It runs as a web app: a FastAPI backend (`src/`) drives the pipeline per uploaded video and
streams progress over a websocket; a Next.js frontend (`frontend/`) provides drag-and-drop
upload, a live view of the generated audio-description segments, and the Q&A box. There is no
CLI entrypoint — the old `src/main.py` was removed in favor of the web server.

## Commands

Backend deps use `uv` (Python 3.12, see `pyproject.toml` / `uv.lock`); the frontend uses npm.

```bash
uv sync                                                            # backend deps
uv run uvicorn server:app --app-dir src --reload --reload-dir src  # backend @ :8000
cd frontend && npm install && npm run dev                          # frontend @ :3000
```

`--app-dir src` puts `src/` on `sys.path`, preserving the flat-import convention (below). Use a
**single** worker process only — the `JobStore` is in-memory, so `--workers > 1` would split
jobs across processes. Requires `ffmpeg`/`ffprobe` on PATH (audio extraction) and a
`GEMINI_API_KEY` in `.env` (loaded via `python-dotenv`; the server boots without one and only
fails a job when it actually calls Gemini).

Tests: `uv run pytest`. Lint/format/types: `uv run ruff check .`, `uv run ruff format .`,
`uv run pyright src`.

## Architecture

`src/server.py` (FastAPI) accepts a video upload, creates a job with a temp dir under
`<tempdir>/adesc-jobs/<job_id>/`, and enqueues it onto an `asyncio.Queue` drained by a small
fixed pool of worker tasks. `src/jobs.py` holds the in-memory `JobStore` (job status, timeline
snapshot, append-only event log for websocket replay, per-job subscriber fan-out, TTL sweep, and
crash recovery from a mirrored `status.json`). `src/pipeline.py::run_pipeline()` orchestrates the
six stages below against job-scoped paths and reports progress via an async `on_event` callback
(stage markers, per-shot vision, the assembled timeline, per-segment narration); CPU-bound stages
run via `asyncio.to_thread`, Gemini stages run natively async. The routes are: `POST /api/jobs`
(upload), `WS /api/jobs/{id}/events` (live stream + replay), `GET /api/jobs/{id}` (status +
timeline snapshot), `GET /api/jobs/{id}/frames/{filename}` (keyframe jpgs, path-traversal
checked), `POST /api/jobs/{id}/ask` (Q&A).

The six stages, each in its own module under `src/`:

1. **`segmentation.py`** — `segment_video()` uses PySceneDetect (`ContentDetector`) to find shot
   (camera cut) boundaries, then grabs one midpoint keyframe per shot via OpenCV. Produces a
   list of shot dicts: `{id, start, end, keyframe}`.
2. **`vision_analysis.py`** — `analyze_shots()` is `async`: it sends each keyframe to Gemini
   (`client.aio`, structured JSON-schema output) to get `description`, `entities`, `setting`,
   `on_screen_text`, with up to `DEFAULT_CONCURRENCY` calls in flight via an `asyncio.Semaphore`.
   An optional `on_shot` callback fires per shot as each completes — completion order is not shot
   order, so consumers must key off `shot["id"]`.
3. **`audio_extract.py`** — `extract_audio()` shells out to `ffmpeg` to pull a 16kHz mono WAV
   (the format Silero VAD and Whisper expect). Raises `NoAudioStreamError` if the source video
   has no audio track; `pipeline.py` catches this and skips stages 4–5, leaving `ad_eligible`
   computed purely from shot duration.
4. **`voice_activity.py`** — `detect_speech_regions()` runs Silero VAD over the WAV to get
   `(start, end)` speech spans.
5. **`transcription.py`** — `transcribe()` runs faster-whisper over the whole audio track, then
   drops any Whisper segment that doesn't overlap a Silero speech region. This VAD-filtering step
   exists specifically to suppress Whisper hallucinating text over music/silence.
6. **`timeline.py`** — `build_timeline()` merges shots + speech regions + transcript into a
   pydantic `Timeline` (list of `Segment`s). Per segment it computes `silence_ratio` (fraction of
   the shot with no detected speech), `narratable_gap_sec` (silence_ratio × shot duration), and
   `ad_eligible` (gap ≥ `MIN_NARRATABLE_GAP_SEC`, currently 2.0s). Then
   `vision_analysis.fill_narration_gaps()` (also `async`, same semaphore + optional `on_segment`
   callback) calls Gemini again for each `ad_eligible` segment, capping narration length via
   `NARRATION_WORDS_PER_SEC` (2.5 wps) applied to `narratable_gap_sec`, and feeding neighboring
   dialogue as context so narration doesn't repeat what's already said.

Timelines are serialized as JSON via `Timeline.model_dump_json()`; `save_timeline()` /
`load_timeline()` in `timeline.py` remain for that. `qa.py` lazily initializes its Gemini client
and CLIP retriever on first use (not at import) and serializes CLIP access behind a lock, so the
server can import it without an API key and concurrent `/ask` calls don't collide.

### Module import style

`src/` is not a package (no `__init__.py`); modules import each other with flat, absolute names
(e.g. `from timeline import NARRATION_WORDS_PER_SEC`). This works because the server is launched
with `uvicorn --app-dir src` (and pytest uses `pythonpath = ["src"]`), both of which put `src/`
on `sys.path`. Keep new modules flat in `src/` and use the same unqualified import style rather
than introducing a package layout, unless deliberately migrating away from this.

### I/O layout

- Per-job temp dirs under `<tempdir>/adesc-jobs/<job_id>/` hold the uploaded `source.<ext>`, the
  extracted `audio.wav`, `frames/shot_XXXX.jpg` keyframes, and a mirrored `status.json` for
  crash recovery. They are swept after `JOB_TTL_SEC` (kept until then so Q&A can read the
  frames). The old fixed `src/in`/`src/out` layout is gone.

Both Gemini calls (shot description and gap narration) use the same `MODEL` constant in
`vision_analysis.py`.

### Dev tools
This project uses ruff lint and pyright type checking, ensure there are no errors with either of these in the code you write/edit.

# Testing Rules

- Run unit tests for touched files as you go; full unit+integration+E2E before commit
- If a test fails, assume the code is wrong, not the test. Only
  edit a test if the requirement intentionally changed — say so
  explicitly. Never edit a test just to silence a failure.
- Bug fix → write a failing regression test first, then fix the code.
- New logic (conditionals, calculations, parsing, edge cases) →
  add tests in the same change. Skip trivial glue code.
- Brittle test (breaks on harmless refactors, checks internals not
  outcomes) → rewrite to check behavior, and flag it as a refactor.
- Delete a test only if its feature is gone or it's fully redundant
  — say which and why.
