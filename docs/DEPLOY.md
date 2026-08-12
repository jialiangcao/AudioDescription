# Deploying adesc

Everything needed to run adesc locally and to deploy it: accounts, CLIs, the
provisioning steps, and which secret goes where.

**Status note.** The application code, migrations, Dockerfiles, Fly configs and
CI are written and tested. The cloud resources below have **not** been
provisioned yet — this is the guide for doing that the first time. Where a
provider's console wording may have drifted, the value you need is described
rather than the exact clicks.

---

## 1. What you're standing up

| Piece | Runs on | Why it's separate |
|---|---|---|
| API | Fly — `adesc-api` | Stateless HTTP + WebSocket. Scale horizontally. |
| media worker | Fly — `adesc-worker-media` | ffmpeg, PySceneDetect, Silero, Whisper, Kokoro, CLIP. Big, CPU-bound, concurrency 1. |
| gemini worker | Fly — `adesc-worker-gemini` | Per-frame vision + narration. Pure network I/O, cheap, scale this one first. |
| beat | Fly — `adesc-beat` | Cron for the stale-job reaper. Exactly one machine. |
| Postgres | Supabase | Jobs, event log, timelines, Q&A runs, auth. |
| Redis | Upstash (via Fly) | Celery broker + result backend, live progress pub/sub, Gemini rate limiter, WS tickets. |
| Object storage | Cloudflare R2 | Source videos, frames, narration clips, described video. |
| Frontend | Vercel | Next.js 15. |

---

## 2. Accounts

Create these first; everything else depends on them.

1. **Google AI Studio** — a Gemini API key. Note your tier's requests-per-minute
   limit; you will set `GEMINI_RPM` from it.
2. **Supabase** — one project for production, one for staging.
3. **Cloudflare** — R2 enabled (needs a card on file even on the free tier).
4. **Fly.io** — with a payment method.
5. **Vercel** — connected to the GitHub repo.
6. **GitHub** — the repo, for Actions.

---

## 3. Local install

### Required

```bash
# macOS
brew install uv node ffmpeg espeak-ng libsndfile postgresql@16
brew install --cask docker            # Docker Desktop, for the local stack

# Debian/Ubuntu
sudo apt-get install -y ffmpeg espeak-ng libsndfile1 postgresql-client nodejs
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **uv** — Python dependency management. Python 3.12 (`.python-version`); uv
  installs it for you.
- **Node 20+** — the frontend. (Verified on 22.)
- **ffmpeg / ffprobe** — audio extraction and the mux. Only the media stages
  shell out to these, but you need them locally since you run everything.
- **espeak-ng** — Kokoro's grapheme-to-phoneme fallback.
- **libsndfile** — what `soundfile` binds to; the AD track is assembled with it.
- **psql** — `run.sh` applies migrations with it.
- **Docker** — runs Postgres, Redis and MinIO locally.

### Deploy-time CLIs

```bash
brew install flyctl supabase/tap/supabase
# or: curl -L https://fly.io/install.sh | sh
```

### First run

```bash
git clone <repo> && cd adesc
cp .env.example .env          # set GEMINI_API_KEY
./run.sh
```

`run.sh` starts Docker services, applies `supabase/migrations/`, installs
dependencies, and runs the API, both workers, beat and the frontend. Open
<http://localhost:3000>.

Locally, auth is off: `.env.example` leaves `ADESC_DEV_USER_ID` commented, so
**uncomment it** to skip sign-in. Any UUID works — it becomes the owner of every
job. The backend logs a loud warning whenever it is set.

```bash
./run.sh --stop        # stop the Docker services
uv run pytest          # tests (needs the Postgres from docker compose)
```

---

## 4. Provisioning

### 4.1 Supabase

Create two projects: `adesc-prod` and `adesc-staging`.

**Creation form:**

| Field | Value | Why |
|---|---|---|
| Name | `adesc-prod` / `adesc-staging` | |
| Database password | generate, save it | It's embedded in `DATABASE_URL`; you cannot read it back later. |
| Region | **East US (North Virginia)** | Match your Fly `primary_region` (`iad` in `fly/*.toml`). Every request the API makes is a round trip; cross-region adds latency to all of them. |
| Postgres version | default | Nothing in the migrations needs a specific one. |
| Plan | Free is fine to start | Free projects **pause after 7 days idle** — fine for staging, not for production. |

**After creation:**

*Authentication → Sign In / Providers*
- Enable **Email**. Leave "Confirm email" on — the magic-link flow satisfies it.
- Password sign-in can be disabled; the frontend only calls `signInWithOtp`.

*Authentication → URL Configuration*
- **Site URL**: your Vercel production URL, e.g. `https://adesc.vercel.app`.
- **Redirect URLs**: add `http://localhost:3000/**` and, if you want sign-in to
  work on preview deploys, a wildcard for them —
  `https://adesc-*-<your-team>.vercel.app/**`. The frontend passes
  `emailRedirectTo: window.location.origin`, so the link returns to whichever
  deployment you signed in from; each of those origins has to be allowlisted.

*Nothing else needs changing.* The migrations create their own tables, RLS
policies and the `pgcrypto` extension, and `auth.users` already exists.

For each project, collect:

- **Connection string** — the *session pooler* URI (port 5432), not the
  transaction pooler. asyncpg uses prepared statements, which the transaction
  pooler does not support.
- **Project URL** — `https://<ref>.supabase.co`.
- **anon key** — public; the browser uses it.
- **JWT verification material** — one of:
  - the **JWT secret** (legacy, symmetric) → set `SUPABASE_JWT_SECRET`, or
  - nothing at all → set `SUPABASE_URL` and the backend fetches the project's
    JWKS to verify asymmetric tokens.

  `src/auth.py` supports both and prefers the secret when present. New projects
  are asymmetric; prefer the JWKS path.

Apply migrations:

```bash
supabase db push --db-url "$STAGING_DATABASE_URL"
supabase db push --db-url "$PROD_DATABASE_URL"
```

This creates `jobs`, `job_events`, `timelines`, `qa_runs` and their RLS
policies. Migrations run automatically on every deploy after this.

### 4.2 Cloudflare R2

Create two buckets: `adesc` and `adesc-staging`.

Create an **R2 API token** with Object Read & Write, and note the access key ID,
secret, and your account ID. The endpoint is:

```
https://<account-id>.r2.cloudflarestorage.com
```

**CORS is required** — the browser PUTs the source video straight to R2, and
GETs frames and the described video from it. Without this, uploads fail with an
opaque network error. Set on each bucket:

```json
[
  {
    "AllowedOrigins": ["https://your-app.vercel.app", "http://localhost:3000"],
    "AllowedMethods": ["GET", "PUT", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Add a **lifecycle rule** expiring `jobs/` after your retention window (24h is a
reasonable start). This is what reclaims job media — there is no sweep task.
Without it, storage grows forever.

Keep the buckets **private**. Everything is served by presigned URL.

### 4.3 Redis

```bash
fly redis create      # Fly's managed Upstash
```

Pick the same region as your apps. Save the connection string — it is shown
once. Create one for staging too.

Redis carries the Celery broker and result backend, job-progress pub/sub, the
Gemini token bucket and WebSocket tickets. None of it is durable state, so a
modest instance is fine; it must have **eviction disabled**, or Celery can lose
queued tasks.

### 4.4 Fly apps

```bash
fly auth login

# production
fly apps create adesc-api
fly apps create adesc-worker-media
fly apps create adesc-worker-gemini
fly apps create adesc-beat

# staging (the deploy workflow appends -staging)
fly apps create adesc-api-staging
fly apps create adesc-worker-media-staging
fly apps create adesc-worker-gemini-staging
fly apps create adesc-beat-staging
```

The media worker needs a scratch volume per machine, in the region from its
config (`iad` by default):

```bash
fly volumes create adesc_scratch --app adesc-worker-media --region iad --size 50
fly volumes create adesc_scratch --app adesc-worker-media-staging --region iad --size 20
```

Scratch is a cache — the source video, frames and intermediate audio for jobs in
flight. Everything durable is uploaded to R2, so losing a machine costs a retry.
Size it for your largest few concurrent videos.

---

## 5. Secrets

### Fly — all four apps

Every app needs the same set. The workers need them because they do the work;
the API needs them because it presigns URLs and reads state.

```bash
for app in adesc-api adesc-worker-media adesc-worker-gemini adesc-beat; do
  fly secrets set --app "$app" \
    GEMINI_API_KEY="…" \
    DATABASE_URL="postgresql://postgres.<ref>:<pw>@…pooler.supabase.com:5432/postgres" \
    REDIS_URL="redis://default:<pw>@<host>:6379" \
    S3_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com" \
    S3_ACCESS_KEY_ID="…" \
    S3_SECRET_ACCESS_KEY="…" \
    S3_REGION="auto" \
    R2_BUCKET="adesc" \
    SUPABASE_URL="https://<ref>.supabase.co" \
    ALLOWED_ORIGINS="https://your-app.vercel.app" \
    GEMINI_RPM="1000"
done
```

Repeat with staging values for the `-staging` apps.

| Variable | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` | workers | Read by `google-genai`. `GOOGLE_API_KEY` also works and wins if both are set. |
| `DATABASE_URL` | all | Supabase **session** pooler, port 5432. |
| `REDIS_URL` | all | Broker, pub/sub, rate limiter, tickets. |
| `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_REGION`, `R2_BUCKET` | all | Without these, `blobs.py` runs local-only and presigning raises. |
| `SUPABASE_URL` | API | JWKS verification. |
| `SUPABASE_JWT_SECRET` | API | Only for legacy symmetric projects; omit otherwise. |
| `ALLOWED_ORIGINS` | API | Comma-separated CORS list. Must include your Vercel domain. |
| `ALLOWED_ORIGIN_REGEX` | API (staging only) | Vercel preview deploys get generated hostnames that can't be listed. Set e.g. `^https://adesc-[a-z0-9-]+\.vercel\.app$` on staging; leave **unset** in production, where the origin list is known and should stay exact. |
| `GEMINI_RPM` | workers | Global requests/minute across *all* workers. Default 1000 is a guess — set it from your tier. |
| `ADESC_LOG_LEVEL` | all | Default `INFO`; `DEBUG` emits one line per sampled frame. |
| `ADESC_DEV_USER_ID` | — | **Never set this in a deployed environment.** It disables auth entirely. |

### GitHub Actions

Repo → Settings → Secrets and variables → Actions:

| Secret | Value |
|---|---|
| `FLY_API_TOKEN` | `fly tokens create deploy` |
| `STAGING_DATABASE_URL` | staging Supabase connection string |
| `PROD_DATABASE_URL` | production Supabase connection string |

Create two **Environments**, `staging` and `production`. Add a required reviewer
to `production` — the deploy workflow waits on that approval before touching
prod.

### Vercel

**Project settings** (Import Git Repository → configure):

| Setting | Value | Why |
|---|---|---|
| Framework Preset | **Next.js** | Pinned by `frontend/vercel.json`, which takes precedence over the dashboard. If this is left on "Other", the build itself succeeds and then fails with `No Output Directory named "public" found` — Vercel looks for a static site instead of `.next`. |
| **Root Directory** | **`frontend`** | The repo is a monorepo; without this the build fails immediately. Leave "Include files outside the root directory" **off** — the frontend needs nothing from `src/`. |
| Build Command | default (`next build`) | |
| Install Command | default (`npm install`) | |
| Output Directory | default | Next.js manages its own. |
| Node.js Version | **20.x or 22.x** | |

**Environment variables.** All three are `NEXT_PUBLIC_*`, so they are **inlined
at build time** — changing one requires a redeploy, not just a restart. Set them
for each environment separately:

| Variable | Production | Preview + Development |
|---|---|---|
| `NEXT_PUBLIC_API_BASE` | `https://adesc-api.fly.dev` | `https://adesc-api-staging.fly.dev` |
| `NEXT_PUBLIC_SUPABASE_URL` | prod project URL | staging project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | prod anon key | staging anon key |

The anon key is designed to be public — it is shipped to the browser and RLS is
what actually protects the data. The **service role key is not used anywhere in
this app**; never put it in a `NEXT_PUBLIC_` variable.

Once you know your production domain, feed it back into two places: the API's
`ALLOWED_ORIGINS` and Supabase's Site URL.

---

## 6. First deploy

CI/CD handles this after the first time, but the first deploy is worth doing by
hand so failures are visible.

```bash
# Workers first: one that understands the new events can serve an old API,
# not the other way round.
fly deploy --config fly/worker-media.toml   --app adesc-worker-media   --remote-only
fly deploy --config fly/worker-gemini.toml  --app adesc-worker-gemini  --remote-only
fly deploy --config fly/beat.toml           --app adesc-beat           --remote-only
fly deploy --config fly/api.toml            --app adesc-api            --remote-only
```

The media image is ~4GB and bakes in Whisper, Kokoro, CLIP and Silero weights,
so the **first build takes 15–25 minutes**. Later builds reuse the layer unless
dependencies change. `--remote-only` builds on Fly rather than pushing 4GB from
your laptop.

Then:

```bash
curl https://adesc-api.fly.dev/healthz    # {"status":"ok"}
curl https://adesc-api.fly.dev/readyz     # postgres + redis both "ok"
fly logs --app adesc-worker-media         # "celery worker process ready"
```

`/readyz` returning 503 names the failing dependency in its body.

After this, pushes to `main` run CI, deploy staging, and wait for approval
before production.

---

## 7. Verifying end to end

1. Sign in on the Vercel URL (magic link).
2. Upload a short clip with dialogue. The upload goes browser → R2 directly; if
   it fails, the R2 CORS policy is the first thing to check.
3. Frames should appear with timestamps within seconds, then fill in with
   descriptions as vision results arrive.
4. In `fly logs --app adesc-worker-media` and `--app adesc-worker-gemini`, the
   vision and audio stages should **overlap** — that's the chord doing its job.
5. When it finishes, the player swaps to the described cut: original audio
   ducking under each narration line.
6. Hard-refresh mid-job. The stream should resume from where it left off, not
   replay from the beginning.
7. Ask a question. The reasoning trace should fill in live, step by step.

Scale check:

```bash
fly scale count 3 --app adesc-worker-gemini
```

Three machines should share one Gemini rate limit, not three — watch for 429s in
the logs.

---

## 8. Operating notes

**Cost.** Stage 2 makes one Gemini call **per sampled frame**, so cost scales
with frame count, not shot count. `segmentation.extract_keyframes`'s
`interval_sec` is the knob — it is currently `1.0`, i.e. one frame per second.
Raising it to `2.0` halves the bill. `GEMINI_RPM` bounds the rate but not the
total.

**Scaling.** Add `adesc-worker-gemini` machines when a backlog builds; that pool
is cheap and network-bound. `adesc-worker-media` runs concurrency 1 by design
(the Kokoro pipeline is a single non-reentrant model), so scale it by adding
machines, each of which needs its own volume.

**Beat must stay at one machine.** Two would double every scheduled tick; zero
means crashed workers' jobs are never reaped.

**Migrations** run before the deploy, and both versions are live during a
rolling one — so every migration must be backward compatible with the running
code.

**A stuck job** will be marked `interrupted` by the reaper within ~10 minutes of
its worker's lease going stale. If jobs are stuck longer than that, check that
beat is running.

**Rollback:** `fly releases --app adesc-api` then
`fly deploy --image <previous>`. Migrations are not rolled back automatically.

---

## 9. Not yet verified

Honest inventory of what has been tested and what has not:

- ✅ Application logic — 243 tests, against a real Postgres.
- ✅ Slim-image import isolation — CI asserts `tasks.py` imports without torch,
  OpenCV, faster-whisper, Kokoro or open_clip.
- ⚠️ **Docker builds have not been run** — Docker was unavailable during
  development. CI builds both images on the first PR; expect to fix something.
- ⚠️ **Redis paths have not run against a real Redis** — the token bucket's Lua,
  pub/sub fan-out, and WS tickets are covered by unit tests with fakes only.
- ⚠️ **R2/MinIO transfers have not been exercised** — `blobs.py` is tested
  against an in-memory fake.
- ⚠️ **No Fly deployment has been performed.** App names, volume mounts and
  secret wiring are as configured, not as observed.

The fastest way to close most of this: start Docker, run `./run.sh`, and put a
real clip through locally. That exercises Redis, MinIO and the full canvas
before any cloud spend.
