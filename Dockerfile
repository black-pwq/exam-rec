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

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EXAM_REC_JOB_ROOT=/data/jobs \
    PADDLE_PDX_CACHE_HOME=/data/models

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

WORKDIR /app

COPY --link --from=builder --chown=examrec:examrec /app/.venv /app/.venv

RUN python -c "import cv2; print('OpenCV', cv2.__version__)"

COPY --link --chown=examrec:examrec \
    app_logging.py \
    api.py \
    pipeline.py \
    question_range.py \
    recognition_jobs.py \
    transform.py \
    ./
COPY --link --chown=examrec:examrec extractor ./extractor
COPY --link --chown=examrec:examrec ocr ./ocr
COPY --link --chown=examrec:examrec utils ./utils
COPY --link --chown=examrec:examrec deploy ./deploy

RUN chmod 0555 /app/deploy/entrypoint.sh

USER examrec

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=4 \
    CMD ["python", "-m", "deploy.healthcheck"]

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
CMD ["serve"]

# Keep revision-only metadata last so a new commit does not invalidate the
# operating-system and Python dependency layers.
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Exam Recognition API" \
      org.opencontainers.image.revision="$VCS_REF"
