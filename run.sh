#!/usr/bin/env bash
# dev.sh
set -e

trap 'kill 0' EXIT  # kill both processes when you Ctrl+C

uv sync
(uv run uvicorn server:app --app-dir src --reload --reload-dir src) &

(cd frontend && npm install && npm run dev) &

wait
