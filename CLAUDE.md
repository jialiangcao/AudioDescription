# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`adesc` generates audio-description (AD) narration for video: it detects shots, describes each
shot visually, transcribes dialogue, finds speech-free gaps long enough to narrate, and asks
Claude to write a narration line sized to fit each gap. It also features interactive Q&A.

## Commands

Dependencies are managed with `uv` (Python 3.12, see `pyproject.toml` / `uv.lock`).

```bash
uv sync                       # install dependencies
uv run python src/main.py     # run the full pipeline on in/test.mp4
```

Requires `ffmpeg`/`ffprobe` on PATH (used for audio extraction) and an `ANTHROPIC_API_KEY` in
`.env` (loaded via `python-dotenv`).

There is no test suite and no linter/formatter configured in this repo.

### Re-running the pipeline

`process_video()` in `src/main.py` short-circuits if `out/timeline.json` already exists — it
loads and prints the cached timeline instead of reprocessing. Delete `out/timeline.json` (or
pass a different `timeline_path`) to force a full re-run.

## Architecture

The pipeline (`src/main.py::process_video`) runs six stages, each in its own module under
`src/`:

1. **`segmentation.py`** — `segment_video()` uses PySceneDetect (`ContentDetector`) to find shot
   (camera cut) boundaries, then grabs one midpoint keyframe per shot via OpenCV. Produces a
   list of shot dicts: `{id, start, end, keyframe}`.
2. **`vision_analysis.py`** — `analyze_shots()` sends each keyframe to Claude (vision, structured
   JSON-schema output) to get `description`, `entities`, `setting`, `on_screen_text`.
3. **`audio_extract.py`** — `extract_audio()` shells out to `ffmpeg` to pull a 16kHz mono WAV
   (the format Silero VAD and Whisper expect). Raises `NoAudioStreamError` if the source video
   has no audio track; `main.py` catches this and skips stages 4–5, leaving `ad_eligible`
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
   `vision_analysis.fill_narration_gaps()` calls Claude again for each `ad_eligible` segment,
   capping narration length via `NARRATION_WORDS_PER_SEC` (2.5 wps) applied to
   `narratable_gap_sec`, and feeding neighboring dialogue as context so narration doesn't repeat
   what's already said.

Timelines are persisted as JSON via `Timeline.model_dump_json()` / `save_timeline()` /
`load_timeline()` in `timeline.py`.

### Module import style

`src/` is not a package (no `__init__.py`); modules import each other with flat, absolute names
(e.g. `from timeline import NARRATION_WORDS_PER_SEC`). This only works because `src/main.py` is
invoked directly (`uv run python src/main.py`), which puts `src/` on `sys.path`. Keep new modules
flat in `src/` and use the same unqualified import style rather than introducing a package
layout, unless deliberately migrating away from this.

### I/O layout

- `in/` — source videos/audio (gitignored).
- `out/` — pipeline artifacts: extracted `audio.wav`, `frames/shot_XXXX.jpg` keyframes, and the
  final `timeline.json` (gitignored).

Both Claude calls (shot description and gap narration) use the same `MODEL` constant in
`vision_analysis.py`.
