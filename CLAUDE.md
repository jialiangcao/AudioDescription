# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`adesc` generates audio-description (AD) narration for video: it detects shots, samples frames
across each shot and describes **every** frame visually, transcribes dialogue, finds speech-free
gaps long enough to narrate, asks Gemini to write a narration line — sized to fit each gap and
drawn from the shot's whole frame sequence — synthesizes that line to speech with Kokoro, and
mixes the narration back into the video's soundtrack so the result can be watched in one pass.
It also features interactive Q&A.

It runs as a deployable service: a stateless FastAPI API (`src/`), Celery workers that run the
pipeline stages, Postgres (Supabase) for job state, Redis for the broker and live progress, and
S3-compatible object storage (Cloudflare R2) for all media. A Next.js frontend (`frontend/`)
provides sign-in, drag-and-drop upload, a frame-by-frame view (every extracted frame with its
timestamp and its own visual analysis beside it, grouped by shot, with the shot's narration
underneath), a player for the described video, and the Q&A box.
There is no CLI entrypoint — the old `src/main.py` was removed in favor of the web server.

## Commands

Backend deps use `uv` (Python 3.12, see `pyproject.toml` / `uv.lock`); the frontend uses npm.

```bash
./run.sh          # docker services + migrations + API + both workers + beat + frontend
./run.sh --stop   # also stop the docker services
```

`run.sh` brings up `docker compose` (Postgres, Redis, MinIO — standing in for Supabase, Redis and
R2), applies `supabase/migrations/`, then starts each process separately, as they are separate
apps in production. To run a piece by hand:

```bash
uv sync --extra media                                              # backend deps (see below)
uv run uvicorn server:app --app-dir src --reload --reload-dir src  # API @ :8000
uv run celery --app celery_app --workdir src worker --queues media,qa --concurrency 1
uv run celery --app celery_app --workdir src worker --queues gemini --concurrency 8 --pool threads
cd frontend && npm install && npm run dev                          # frontend @ :3000
```

`--app-dir src` / `--workdir src` puts `src/` on `sys.path`, preserving the flat-import convention
(below). The API is stateless, so `--workers > 1` and multiple replicas are fine.

Dependencies are split: the base set is what the API and the gemini-queue worker need, and the
`media` extra adds torch, OpenCV, faster-whisper, Kokoro and open_clip (~4GB). `segmentation.py`
and `transcription.py` import their heavy dependencies **lazily** to keep `tasks.py` importable
without the extra — CI asserts this, since it is the premise of the two-image split.

Stage 2 issues one Gemini call **per sampled frame**, so a job's API cost scales with total frame
count, not shot count — `segmentation.extract_keyframes`'s `interval_sec` is the knob. Media
workers need `ffmpeg`/`ffprobe` (audio extraction, mux) and `espeak-ng` (Kokoro's G2P fallback) on
PATH; the API and gemini workers need neither. Configuration comes from `.env` (see
`.env.example`): `GEMINI_API_KEY`, `DATABASE_URL`, `REDIS_URL`, the `S3_*`/`R2_BUCKET` settings,
and either Supabase auth settings or `ADESC_DEV_USER_ID` for local work.

Tests: `uv run pytest`. Tests that need Postgres (repo, tasks, server) skip unless
`TEST_DATABASE_URL` or `DATABASE_URL` points at one — `docker compose up -d postgres` provides it,
and CI runs it as a service container. Lint/format/types: `uv run ruff check .`,
`uv run ruff format .`, `uv run pyright src`.

## Architecture

Four processes, deployed as four Fly apps (`fly/*.toml`) built from two images (`docker/`):

- **API** (`src/server.py`, slim image) — stateless. Verifies a Supabase JWT (`src/auth.py`),
  reads and writes job state through `src/repo.py`, hands the browser presigned R2 URLs, and
  streams progress over a websocket. It holds no job state and never touches media bytes, so it
  scales to as many replicas as you like.
- **media worker** (`docker/Dockerfile.media`) — the CPU/native stages plus the `qa` queue (whose
  CLIP retriever needs torch). Concurrency 1 per machine: the Kokoro pipeline is a single shared,
  non-reentrant model.
- **gemini worker** (slim image) — per-frame vision and narration. Pure network I/O, so it runs at
  high thread concurrency and is the pool to scale out first.
- **beat** — the scheduler behind `tasks.reap_stale_jobs`. Exactly one, always.

State and transport:

- `src/repo.py` + `supabase/migrations/` — `jobs`, `job_events`, `timelines`, `qa_runs`, all under
  RLS. `job_events` carries a **per-job monotonic `seq`** (allocated under an advisory lock,
  because the vision stage publishes from several workers at once), which is what lets a
  reconnecting websocket replay `seq > n` instead of the whole log. The timeline lives in its own
  table so it is not rewritten on every event. Jobs carry a `heartbeat_at` worker lease; a job
  whose worker died is found by the reaper rather than sitting in `processing` forever.
- `src/events.py` — Redis pub/sub (`job:{id}:events`). Durability is Postgres's job; a dropped
  publish costs latency, never an event, because clients resume from their last `seq`.
- `src/blobs.py` — `JobBlobs` fronts a local scratch dir with an optional S3-compatible bucket.
  Artifacts are addressed by **job-relative key** (`frames/shot_0000_00.jpg`,
  `narration/ad_track.wav`, `described.mp4`); `fetch()` downloads only what isn't already in
  scratch, and `presign_get/put` hand the browser a direct URL. `store=None` gives a purely local
  instance, which is what `run_pipeline` and the tests use.
- `src/gemini_limits.py` — a Redis token bucket shared by every worker, plus a typed retry policy
  (429/5xx and transport failures back off with jitter and honour `Retry-After`; 4xx fails at
  once). `qa/utils.with_retries` delegates here, so the pipeline and the Q&A agents share one
  global rate limit.

The pipeline as a Celery canvas (`src/tasks.py`, queues in `src/celery_app.py`):

```
segment (media)
  └─ self.replace(chord(
         group(vision × one per shot (gemini), audio (media)),
         chain(build_timeline, narrate, tts, mux)
     ))
```

The fan-out width is only known once segmentation has found the shots, hence `self.replace`.
Vision fans out per **shot** — the largest unit one task owns end to end, so no two tasks write
the same blob. Two wins over the single-process version: per-frame vision spreads across machines,
and the network-bound vision stage overlaps the CPU-bound audio stages. `src/job_state.py` holds
the JSON handoff between stages that now run on different machines. Failure reporting is explicit
in `tasks._run_stage`, **not** a canvas errback — a chord's error callback does not reliably reach
its header, and vision and audio are in the header.

`src/pipeline.py::run_pipeline()` remains as the reference implementation of the stage sequence and
the local/test path; it calls the same stage functions in one process.

Routes: `POST /api/jobs` (reserve + presigned PUT), `POST /api/jobs/{id}/start` (verify the upload,
enqueue), `GET /api/jobs` (list), `GET /api/jobs/{id}` (status + timeline with presigned URLs),
`POST /api/jobs/{id}/media-urls` (presign a batch of keys — progress events carry keys, and
`<img>`/`<audio>`/`<video>` cannot send an auth header), `POST /api/jobs/{id}/ws-ticket`,
`WS /api/jobs/{id}/events?ticket=…&since=…`, `POST /api/jobs/{id}/ask` (queues a run),
`GET /api/qa/{run_id}`, `/healthz`, `/readyz`.

The eight stages, each in its own module under `src/`:

1. **`segmentation.py`** — `segment_video()` uses PySceneDetect (`ContentDetector`) to find shot
   (camera cut) boundaries, then samples a frame every `interval_sec` (2.0s) within each shot via
   OpenCV, writing them through `JobBlobs`. Produces a list of shot dicts:
   `{id, start, end, frames}`, where each frame is `{index, time, key}` — `index` its position
   within the shot, `time` its absolute timestamp in the video, `key` its job-relative blob key.
   No frame is privileged; the old midpoint `keyframe` concept is gone. `probe_video()` reads
   fps/duration once here, so no later stage needs the source video.
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
6. **`timeline.py`** — `build_timeline(job_id, duration_sec, shots, …)` merges shots + speech
   regions + transcript into a
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
   `asyncio.to_thread` and writes a WAV under the job's `narration/` keyspace. Since
   `narratable_gap_sec`
   is only an estimate, a clip longer than its gap is re-synthesized once at a higher `speed`
   (capped at `MAX_SPEED`); if it's still too long it's kept and flagged. Results are written back
   onto the segment as `ad_narration_key`, `ad_narration_duration_sec`, and
   `ad_narration_overflow`, and streamed via the optional `on_segment` callback. The Kokoro
   pipeline is a single shared model, so clips are synthesized one at a time (in id order), unlike
   the concurrent Gemini stages.
8. **`ad_track.py`** then **`mux.py`** — `build_ad_track()` lays every narration clip into one
   video-length WAV at its `narration_start_sec` (the AD-only track, recorded on the timeline as
   `ad_track_key`). `mux_described_video()` then shells out to `ffmpeg` to write
   `described.mp4`: the source picture stream-copied (re-encoded to H.264 only if the copy is
   rejected) with an audio track that is the original soundtrack ducked under the narration via
   `sidechaincompress` keyed off the AD track, then mixed with it. `apad` on the AD track keeps a
   short track from truncating the soundtrack, and `amix=duration=first` trims back to the source
   length. A source with no audio stream just gets the AD track as its soundtrack. The result is
   recorded as `Timeline.described_key` and is what the frontend plays; if no segment produced
   narration, both steps are skipped and no described video is written.

Timelines are serialized as JSON via `Timeline.model_dump_json()`; `save_timeline()` /
`load_timeline()` in `timeline.py` remain for that.

### Q&A (`src/qa/` package)

`POST /api/jobs/{id}/ask` **queues** a run (a run takes minutes, far too long to hold an HTTP
request open) and returns a `run_id`; `tasks.answer_question` executes it on the `qa` queue and
streams the trace as `qa_trace` events on the job's channel, so the panel fills in live. Poll
`GET /api/qa/{run_id}` for the final result.

The system is a port of Symphony's video-QA agents onto Gemini (the port's decisions are recorded in
`qaPLAN.md`, removed from the tree — `git show 1e1d576:qaPLAN.md`), with the control flow
expressed as a LangGraph `StateGraph`.
`qa/orchestrator.py::answer_question()` is the async entrypoint: a text-only **CoreAgent** planner
loops (≤ `MAX_CYCLES` = 17 cycles), each cycle
dispatching one worker agent — **LocalizeAgent** (grounds the question in time: CLIP
`retrieve_tool` or the exhaustive `localize_tool`, which scores every 30s window with a Gemini
vision call), **PerceptionAgent** (a ReAct loop over `frame_inspect_tool` /
`interval_summary_tool` / `frame_associate_tool`), or **SubtitleAgent** (one pass over the
timeline's dialogue transcript) — and appending the result to `history`, a reducer-appended graph
channel that is re-serialized into every planner prompt. A `finish` decision passes through
**ReflectionAgent** exactly once per question (fail-open critic); the result is
`{status, answer, cycles, history}` and the frontend renders the history as a collapsible
reasoning-trace panel.

The graph's payoff is not the loop but that every transition is a named node, so `run(on_step=…)`
can report each step as it happens. Record shapes and their **key order** matter — history is
serialized straight into the planner's next prompt — as do the `if_reflected` latch and the
verbatim not-credible sentence; `tests/test_qa_orchestrator.py` is the specification and passed
the port unedited. The graph is compiled per run so swapping an agent on the instance is picked up.

Tools resolve frames via `qa/frame_index.py` (built from each `Frame.time`, addressing frames by
blob key) and retrieve with `qa/retriever.py` (open_clip behind a lock + an embedding cache keyed
by **bucket-absolute** object key, since job-relative keys repeat across jobs). Models, thinking
levels and top-k knobs live in `qa/config.py`, prompts in `qa/prompts.py`.

### Module import style

`src/` is not a package (no `__init__.py`); modules import each other with flat, absolute names
(e.g. `from timeline import NARRATION_WORDS_PER_SEC`). This works because the server is launched
with `uvicorn --app-dir src` (and pytest uses `pythonpath = ["src"]`), both of which put `src/`
on `sys.path`. Keep new modules flat in `src/` and use the same unqualified import style rather
than introducing a package layout, unless deliberately migrating away from this.

**Exception:** `src/qa/` is a package — the multi-agent Q&A system is too large for one module.
It deliberately keeps the name `qa`, so `import qa` / `qa.answer_question` work exactly as
before; inside the package, imports are package-absolute (`from qa.config import ...`).

### I/O layout

Artifacts are addressed by **job-relative blob key**, never by path. Under `jobs/<job_id>/` in the
bucket, and mirrored into each worker's scratch dir (`ADESC_SCRATCH_ROOT`, a Fly volume in
production):

```
source.<ext>                  the upload (browser PUTs it directly)
audio.wav                     extracted 16kHz mono
frames/shot_XXXX_YY.jpg       sampled frames (XXXX = shot id, YY = frame index in shot)
narration/shot_XXXX.wav       synthesized narration clips
narration/ad_track.wav        the combined AD-only track
described.mp4                 the muxed result
state/shots.json              the shot skeleton + probed fps/duration
state/vision/shot_XXXX.json   one shot's frame analyses (one blob per shot: no write conflicts)
state/audio.json              speech regions + transcript
```

`state/*` is the handoff between stages that run on different machines — `run_pipeline` could keep
these in local variables, Celery tasks cannot. Expiry is an R2 lifecycle rule on `jobs/*` rather
than a sweep task. Scratch is a cache: everything durable is uploaded, so losing a machine costs a
retry, not a job.

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

### Deployment

See `docs/DEPLOY.md` for the full guide (accounts, CLIs, provisioning, secrets).
Two images, four Fly apps, one Vercel project.

- `docker/Dockerfile.slim` — API (`--target api`) and gemini worker (`--target worker`). Carries
  **no** torch/OpenCV/faster-whisper/Kokoro/open_clip; CI asserts that `tasks.py` imports without
  them, since that is the whole premise of the split.
- `docker/Dockerfile.media` — the media+qa worker and beat (`--target beat`). Installs
  `ffmpeg`/`espeak-ng` (previously undeclared system dependencies) and **bakes the model weights
  in** via `docker/warm_models.py`, so a cold start no longer pays ~1.2GB of downloads inside the
  first job.
- `fly/*.toml` — one per app. Only the media worker mounts a volume (scratch); everything else is
  stateless.
- `.github/workflows/ci.yml` — lint, typecheck and pytest against real Postgres/Redis service
  containers, frontend build, and both image builds.
- `.github/workflows/deploy.yml` — on `main`: migrations, then workers, then the API (a worker that
  understands new events can serve an old API, not the reverse), then a `/readyz` smoke test.
  Staging runs automatically; production is gated on a GitHub Environment approval.
- The frontend deploys to Vercel. `NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_SUPABASE_URL` and
  `NEXT_PUBLIC_SUPABASE_ANON_KEY` are **inlined at build time**, so they must be set per Vercel
  environment, not at runtime.

Migrations run before the deploy and both versions are live during a rolling one, so every
migration must be backward compatible with the running code.

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
