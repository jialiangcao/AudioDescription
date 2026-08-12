# adesc

Generates audio-description narration for video, and answers questions about it.

It detects shots, describes **every** sampled frame with Gemini, transcribes the
dialogue, finds the speech-free gaps long enough to narrate, writes a narration
line sized to fit each gap, speaks it with Kokoro, and mixes it back into the
soundtrack — ducked under the narration — so the result can be watched in one
pass.

## Quick start

```bash
brew install uv node ffmpeg espeak-ng libsndfile postgresql@16
brew install --cask docker           # then start Docker Desktop

cp .env.example .env                 # set GEMINI_API_KEY, uncomment ADESC_DEV_USER_ID
./run.sh
```

Then open <http://localhost:3000> and drop in a video.

`run.sh` starts Postgres, Redis and MinIO in Docker, applies the migrations, and
runs the API, both Celery workers, beat and the frontend.

## Docs

- **[docs/DEPLOY.md](docs/DEPLOY.md)** — what to install, how to provision
  Supabase / R2 / Redis / Fly / Vercel, and which secret goes where.
- **[CLAUDE.md](CLAUDE.md)** — architecture: the four processes, the Celery
  canvas, the blob-key I/O layout, and the LangGraph Q&A system.

## Tests

```bash
docker compose up -d postgres        # repo/task/server tests skip without it
uv run pytest
```
