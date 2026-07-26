#!/bin/sh
set -eu

python -m deploy.preflight

exec /app/.venv/bin/uvicorn api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --proxy-headers \
    --forwarded-allow-ips='*' \
    --timeout-graceful-shutdown "${EXAM_REC_GRACEFUL_SHUTDOWN_SECONDS:-600}"
