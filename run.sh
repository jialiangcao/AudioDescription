#!/usr/bin/env bash
# Local development: the real services the deployed system uses, plus the API,
# both workers, beat and the frontend.
#
#   ./run.sh            start everything (Ctrl+C stops the app processes)
#   ./run.sh --fast     skip `uv sync` / `npm install`
#   ./run.sh --stop     stop the docker services too
#
# Output from all five processes is prefixed by name on stdout and tee'd to
# logs/<name>.log.
set -euo pipefail

cd "$(dirname "$0")"

SKIP_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --stop)  docker compose down; exit 0 ;;
    --fast)  SKIP_DEPS=1 ;;
    *) echo "usage: $0 [--fast] [--stop]" >&2; exit 2 ;;
  esac
done

# Absolute: the frontend is launched from a subshell in frontend/.
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f .env ]]; then
  echo "no .env found; copying .env.example (set GEMINI_API_KEY in it)" >&2
  cp .env.example .env
fi

# --- ports -----------------------------------------------------------------
# A stale API or frontend from a previous run is the most common way this
# script "starts" but serves yesterday's code, so refuse rather than race.
for port in 8000 3000; do
  if lsof -ti "tcp:$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port is already in use:" >&2
    lsof -i "tcp:$port" -sTCP:LISTEN >&2
    echo "stop it first (e.g. kill \$(lsof -ti tcp:$port -sTCP:LISTEN))" >&2
    exit 1
  fi
done

# --- docker ----------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "==> docker daemon is down; starting Docker Desktop"
    open -a Docker
    for _ in $(seq 1 60); do
      docker info >/dev/null 2>&1 && break
      sleep 2
    done
  fi
  docker info >/dev/null 2>&1 || { echo "docker daemon is not running" >&2; exit 1; }
fi

echo "==> starting postgres, redis and minio"
# Not `--wait`: it treats the one-shot minio-init container exiting 0 as a
# failure. Poll the long-running services' healthchecks instead.
docker compose up -d
for _ in $(seq 1 60); do
  unhealthy=""
  for svc in postgres redis minio; do
    cid=$(docker compose ps -q "$svc")
    state=$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo starting)
    [[ "$state" == "healthy" ]] || unhealthy="$unhealthy $svc"
  done
  [[ -z "$unhealthy" ]] && break
  sleep 2
done
[[ -z "$unhealthy" ]] || { echo "services never became healthy:$unhealthy" >&2; exit 1; }

# shellcheck disable=SC1091
set -a; source .env; set +a

echo "==> applying database migrations"
# Plain Postgres has no auth schema; the migrations foreign-key into it. Same
# shim the tests use (tests/conftest.py), so the real migrations run unmodified.
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f tests/sql/auth_shim.sql
for migration in supabase/migrations/*.sql; do
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f "$migration" >/dev/null 2>&1 \
    || echo "    (already applied: $(basename "$migration"))"
done

# Supabase owns the real auth.users; locally nothing populates the shim, so a
# job insert by a signed-in user fails jobs_owner_id_fkey. Seed the ids you
# sign in as (ADESC_LOCAL_USER_IDS in .env, comma-separated).
if [[ -n "${ADESC_LOCAL_USER_IDS:-}" ]]; then
  IFS=, read -ra _uids <<< "$ADESC_LOCAL_USER_IDS"
  for uid in "${_uids[@]}"; do
    uid=$(echo "$uid" | tr -d '[:space:]')
    [[ -n "$uid" ]] || continue
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q \
      -c "insert into auth.users (id) values ('$uid') on conflict do nothing"
    echo "    seeded local auth user $uid"
  done
fi

if [[ "$SKIP_DEPS" == 0 ]]; then
  echo "==> installing dependencies (--fast skips this)"
  uv sync --extra media
  (cd frontend && npm install --silent)
fi

# `--app-dir` / `--workdir` put src/ on sys.path in the parent process only;
# celery's prefork children re-import without it and lose log_config.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

trap 'kill 0' EXIT  # stop every process on Ctrl+C

# Prefix each process's output with its name and keep a copy in logs/.
run_svc() {
  local name=$1; shift
  "$@" > >(tee -a "$LOG_DIR/$name.log" | awk -v p="[$name] " '{print p $0; fflush()}') 2>&1 &
}

# The API and the workers are separate processes here for the same reason they
# are separate apps in production: so a stuck stage cannot take the API down.
run_svc api    uv run uvicorn server:app --app-dir src --reload --reload-dir src
run_svc media  uv run celery --app celery_app --workdir src worker \
                 --queues media,qa --concurrency 1 --pool prefork --loglevel INFO
run_svc gemini uv run celery --app celery_app --workdir src worker \
                 --queues gemini --concurrency 8 --pool threads --loglevel INFO
run_svc beat   uv run celery --app celery_app --workdir src beat --loglevel INFO
(cd frontend && run_svc web npm run dev)

# Report readiness once, rather than leaving you to guess from the log noise.
(
  for _ in $(seq 1 90); do
    api=$(curl -fsS http://localhost:8000/readyz 2>/dev/null) || api=""
    web=$(curl -so /dev/null -w '%{http_code}' http://localhost:3000 2>/dev/null) || web=""
    if [[ -n "$api" && "$web" == "200" ]]; then
      echo "==> ready: frontend http://localhost:3000 | api http://localhost:8000 $api"
      exit 0
    fi
    sleep 2
  done
  echo "==> not ready after 3min; see $LOG_DIR/" >&2
) &

wait
