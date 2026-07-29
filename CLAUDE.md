# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`adesc` generates audio-description (AD) narration for video: it detects shots, samples frames
across each shot and describes **every** frame visually, transcribes dialogue, finds speech-free
gaps long enough to narrate, asks Gemini to write a narration line — sized to fit each gap and
drawn from the shot's whole frame sequence — synthesizes that line to speech with Kokoro, and
mixes the narration back into the video's soundtrack so the result can be watched in one pass.
It also features interactive Q&A.

It runs as a web app: a FastAPI backend (`src/`) drives the pipeline per uploaded video and
streams progress over a websocket; a Next.js frontend (`frontend/`) provides drag-and-drop
upload, a frame-by-frame view (every extracted frame with its timestamp and its own visual
analysis beside it, grouped by shot, with the shot's narration underneath), a player for the
described video, and the Q&A box.
There is no CLI entrypoint — the old `src/main.py` was removed in favor of the web server.

## Commands

Backend deps use `uv` (Python 3.12, see `pyproject.toml` / `uv.lock`); the frontend uses npm.

```bash
uv sync                                                            # backend deps
uv run uvicorn server:app --app-dir src --reload --reload-dir src  # backend @ :8000
cd frontend && npm install && npm run dev                          # frontend @ :3000
```

`--app-dir src` puts `src/` on `sys.path`, preserving the flat-import convention (below). Use a
**single** worker process only — the `JobStore` is in-memory, so `--workers > 1` would split
jobs across processes. Note that stage 2 issues one Gemini call **per sampled frame**, so a job's
API cost scales with total frame count, not shot count — `segmentation.extract_keyframes`'s
`interval_sec` (2.0s) is the knob. Requires `ffmpeg`/`ffprobe` on PATH (audio extraction), `espeak-ng` on
PATH (Kokoro's G2P fallback for narration TTS), and a `GEMINI_API_KEY` in `.env` (loaded via
`python-dotenv`; the server boots without one and only fails a job when it actually calls Gemini).

Tests: `uv run pytest`. Lint/format/types: `uv run ruff check .`, `uv run ruff format .`,
`uv run pyright src`.

## Architecture

`src/server.py` (FastAPI) accepts a video upload, creates a job with a temp dir under
`<tempdir>/adesc-jobs/<job_id>/`, and enqueues it onto an `asyncio.Queue` drained by a small
fixed pool of worker tasks. `src/jobs.py` holds the in-memory `JobStore` (job status, timeline
snapshot, append-only event log for websocket replay, per-job subscriber fan-out, TTL sweep, and
crash recovery from a mirrored `status.json`). `src/pipeline.py::run_pipeline()` orchestrates the
eight stages below against job-scoped paths and reports progress via an async `on_event` callback
(stage markers, the shot/frame skeleton, per-frame vision, the assembled timeline, per-segment
narration text, per-segment narration audio, the combined AD track, the final described video);
CPU-bound stages run via `asyncio.to_thread`, Gemini stages run natively async.
The routes are: `POST /api/jobs` (upload), `WS /api/jobs/{id}/events` (live stream + replay),
`GET /api/jobs/{id}` (status + timeline snapshot), `GET /api/jobs/{id}/frames/{filename}`
(sampled frame jpgs, path-traversal checked), `GET /api/jobs/{id}/narration/{filename}` (synthesized
narration wavs, path-traversal checked), `GET /api/jobs/{id}/described` (the muxed video —
a fixed artifact per job, so no filename parameter), `POST /api/jobs/{id}/ask` (Q&A).

The eight stages, each in its own module under `src/`:

1. **`segmentation.py`** — `segment_video()` uses PySceneDetect (`ContentDetector`) to find shot
   (camera cut) boundaries, then samples a frame every `interval_sec` (2.0s) within each shot via
   OpenCV. Produces a list of shot dicts: `{id, start, end, frames}`, where each frame is
   `{index, time, path}` — `index` its position within the shot, `time` its absolute timestamp in
   the video. No frame is privileged; the old midpoint `keyframe` concept is gone.
2. **`vision_analysis.py`** — `analyze_shots()` is `async` and works **per frame**: it sends every
   sampled frame to Gemini (`client.aio`, structured JSON-schema output) to get `description`,
   `entities`, `actions`, `setting`, `on_screen_text`, with up to `DEFAULT_CONCURRENCY` calls in
   flight across the whole video via an `asyncio.Semaphore`. Each frame is analyzed in isolation
   (`NO_CONTEXT`, no neighbouring frames), so its analysis is a faithful record of that frame
   alone — which is exactly what the frontend displays. There is no shot-level rollup. An optional
   `on_frame(shot, frame)` callback fires per frame as each completes — completion order is
   neither shot nor frame order, so consumers must key off `shot["id"]` and `frame["index"]`.
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
   pydantic `Timeline` (list of `Segment`s, each holding its `frames: list[Frame]` with their
   `FrameAnalysis`). Per segment it computes `silence_ratio` (fraction of the shot with no
   detected speech), `narratable_gap_sec` (the longest contiguous speech-free span), and
   `ad_eligible` (gap ≥ `MIN_NARRATABLE_GAP_SEC`, currently 2.0s). Then
   `vision_analysis.fill_narration_gaps()` (also `async`, with an optional `on_segment` callback)
   calls Gemini again for each `ad_eligible` segment. Narration is the **culmination of the whole
   shot**: every sampled frame is attached as an image, in temporal order, alongside its
   description/actions/on-screen text, so the line describes the arc across the shot rather than
   one instant. Length is capped via `NARRATION_WORDS_PER_SEC` (2.5 wps) applied to
   `narratable_gap_sec`, and neighboring dialogue is fed as context so narration doesn't repeat
   what's already said. This stage runs **sequentially** (unlike stage 2): each line receives the
   preceding `NARRATION_CONTEXT_SCENES` shots' frame descriptions, dialogue, and narration for
   continuity.
7. **`tts.py`** — `synthesize_narration()` is `async`: for each segment with `ad_narration` set,
   it runs the line through Kokoro-82M (`kokoro.KPipeline`, voice `VOICE`, 24kHz) via
   `asyncio.to_thread` and writes a WAV under the job's `narration/`. Since `narratable_gap_sec`
   is only an estimate, a clip longer than its gap is re-synthesized once at a higher `speed`
   (capped at `MAX_SPEED`); if it's still too long it's kept and flagged. Results are written back
   onto the segment as `ad_narration_audio` (server path), `ad_narration_duration_sec`, and
   `ad_narration_overflow`, and streamed via the optional `on_segment` callback. The Kokoro
   pipeline is a single shared model, so clips are synthesized one at a time (in id order), unlike
   the concurrent Gemini stages.
8. **`ad_track.py`** then **`mux.py`** — `build_ad_track()` lays every narration clip into one
   video-length WAV at its `narration_start_sec` (the AD-only track, recorded on the timeline as
   `ad_track_audio`). `mux_described_video()` then shells out to `ffmpeg` to write
   `described.mp4`: the source picture stream-copied (re-encoded to H.264 only if the copy is
   rejected) with an audio track that is the original soundtrack ducked under the narration via
   `sidechaincompress` keyed off the AD track, then mixed with it. `apad` on the AD track keeps a
   short track from truncating the soundtrack, and `amix=duration=first` trims back to the source
   length. A source with no audio stream just gets the AD track as its soundtrack. The result is
   recorded as `Timeline.described_video` and is what the frontend plays; if no segment produced
   narration, both steps are skipped and no described video is written.

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
  extracted `audio.wav`, `frames/shot_XXXX_YY.jpg` sampled frames (`XXXX` = shot id, `YY` = frame
  index within the shot), `narration/shot_XXXX.wav` synthesized narration clips,
  `narration/ad_track.wav` (the combined AD-only track), the muxed `described.mp4`, and a mirrored
  `status.json` for crash recovery. They are swept after `JOB_TTL_SEC` (kept until then so Q&A can
  read the frames and the frontend can serve the frame-by-frame view, play the narration clips, and
  play the described video). The old fixed `src/in`/`src/out` layout is gone.

Both Gemini calls (frame description and gap narration) use the same `MODEL` constant in
`vision_analysis.py`.

### Logging

`src/log_config.py::configure_logging()` installs the root logger's level + formatter; the server
calls it once at import (after `load_dotenv()`). Every module logs through its own
`logging.getLogger(__name__)` and never touches handlers/levels itself, so tests and embedders keep
control. Level comes from the `ADESC_LOG_LEVEL` env var (default `INFO`; use `DEBUG` for the
verbose per-shot / per-frame / per-segment traces, `WARNING` for problems only; note that `DEBUG` now emits one line per
sampled frame, not one per shot). Convention: `INFO`
for stage boundaries and job-lifecycle transitions, `DEBUG` for per-item detail, `WARNING` for
recoverable oddities (no audio stream, no cuts detected, narration overflow), `ERROR` /
`logger.exception` for failures. Noisy third-party loggers (`httpx`, `faster_whisper`, …) are
pinned to `WARNING` unless the level is `DEBUG`.

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
