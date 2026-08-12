#!/usr/bin/env bash
# Local development: the real services the deployed system uses, plus the API,
# both workers, beat and the frontend.
#
#   ./run.sh          start everything
#   ./run.sh --stop   stop the docker services too
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "--stop" ]]; then
  docker compose down
  exit 0
fi

if [[ ! -f .env ]]; then
  echo "no .env found; copying .env.example (set GEMINI_API_KEY in it)" >&2
  cp .env.example .env
fi

echo "==> starting postgres, redis and minio"
docker compose up -d --wait

echo "==> applying database migrations"
# shellcheck disable=SC1091
set -a; source .env; set +a
for migration in supabase/migrations/*.sql; do
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f "$migration" \
    || echo "    (already applied: $(basename "$migration"))"
done

echo "==> installing dependencies"
uv sync --extra media
(cd frontend && npm install --silent)

trap 'kill 0' EXIT  # stop every process on Ctrl+C

# The API and the workers are separate processes here for the same reason they
# are separate apps in production: so a stuck stage cannot take the API down.
uv run uvicorn server:app --app-dir src --reload --reload-dir src &
uv run celery --app celery_app --workdir src worker \
  --queues media,qa --concurrency 1 --pool prefork --loglevel INFO &
uv run celery --app celery_app --workdir src worker \
  --queues gemini --concurrency 8 --pool threads --loglevel INFO &
uv run celery --app celery_app --workdir src beat --loglevel INFO &
(cd frontend && npm run dev) &

wait
