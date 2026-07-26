#!/bin/sh
set -eu

if [ "$#" -eq 0 ] || [ "$1" = "serve" ]; then
    python -m deploy.preflight
    set -- \
        /app/.venv/bin/uvicorn api:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --timeout-graceful-shutdown \
        "${EXAM_REC_GRACEFUL_SHUTDOWN_SECONDS:-600}"
fi

exec "$@"
