# syntax=docker/dockerfile:1.7

FROM python:3.12.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

ARG PADDLE_GROUP=paddle-cpu

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    case "$PADDLE_GROUP" in \
        paddle-cpu|paddle-cu118) ;; \
        *) echo "unsupported PADDLE_GROUP: $PADDLE_GROUP" >&2; exit 2 ;; \
    esac \
    && uv sync \
        --locked \
        --no-default-groups \
        --group "$PADDLE_GROUP" \
        --no-install-project


FROM python:3.12.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system --gid 10001 examrec \
    && adduser \
        --system \
        --uid 10001 \
        --ingroup examrec \
        --home /home/examrec \
        --shell /usr/sbin/nologin \
        examrec \
    && mkdir -p /app /data/jobs /data/models \
    && chown -R examrec:examrec /app /data /home/examrec

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EXAM_REC_JOB_ROOT=/data/jobs \
    PADDLE_PDX_CACHE_HOME=/data/models

WORKDIR /app

COPY --link --from=builder --chown=10001:10001 /app/.venv /app/.venv

RUN python -c "import cv2; print('OpenCV', cv2.__version__)"

COPY --link --chown=10001:10001 src ./src

USER examrec

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=4 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health/ready', timeout=5).close()"]

CMD ["/app/.venv/bin/uvicorn", "exam_rec.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "600"]

LABEL org.opencontainers.image.title="Exam Recognition API"
