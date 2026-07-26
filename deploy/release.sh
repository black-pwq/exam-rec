#!/bin/sh
set -eu

runtime="${1:-}"
case "$runtime" in
    cpu)
        paddle_group="paddle-cpu"
        ;;
    gpu)
        paddle_group="paddle-cu118"
        ;;
    *)
        echo "usage: $0 cpu|gpu [image-tag]" >&2
        exit 2
        ;;
esac

if [ ! -f deploy/.env.production ]; then
    echo "missing deploy/.env.production; copy deploy/env.production.example" >&2
    exit 2
fi
if [ ! -s deploy/secrets/llm_api_key ]; then
    echo "missing or empty deploy/secrets/llm_api_key" >&2
    exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "tracked files must be clean before building a release image" >&2
    exit 2
fi

revision="$(git rev-parse HEAD)"
short_revision="$(git rev-parse --short=12 HEAD)"
image="${2:-exam-rec:${short_revision}-${runtime}}"
export EXAM_REC_IMAGE="$image"
export EXAM_REC_VCS_REF="$revision"

compose() {
    if [ "$runtime" = "gpu" ]; then
        docker compose \
            --env-file deploy/.env.production \
            -f compose.yaml \
            -f compose.gpu.yaml \
            "$@"
    else
        docker compose \
            --env-file deploy/.env.production \
            -f compose.yaml \
            "$@"
    fi
}

compose config --quiet
docker build \
    --pull \
    --build-arg "PADDLE_GROUP=$paddle_group" \
    --build-arg "VCS_REF=$revision" \
    --tag "$image" \
    .

previous_image=""
app_id="$(compose ps --quiet app)"
if [ -n "$app_id" ]; then
    previous_image="$(docker inspect --format '{{.Config.Image}}' "$app_id")"
fi

rollback() {
    exit_code=$?
    trap - EXIT INT TERM
    if [ "$exit_code" -ne 0 ] && [ -n "$previous_image" ]; then
        echo "release failed; restoring $previous_image" >&2
        EXAM_REC_IMAGE="$previous_image"
        export EXAM_REC_IMAGE
        compose up --detach --no-build --remove-orphans || true
    fi
    exit "$exit_code"
}
trap rollback EXIT INT TERM

proxy_id="$(compose ps --quiet proxy)"
if [ -n "$proxy_id" ]; then
    if ! compose exec --no-TTY proxy nginx -s quit; then
        compose stop proxy
    fi
    docker wait "$proxy_id" >/dev/null
fi

if [ -n "$app_id" ]; then
    compose run \
        --rm \
        --no-deps \
        app \
        python -m deploy.drain_check --wait --timeout 3600 --interval 5
    compose stop app
fi

compose run --rm --no-deps app python -m deploy.warmup
compose up --detach --no-build --remove-orphans

attempt=0
while [ "$attempt" -lt 60 ]; do
    app_id="$(compose ps --quiet app)"
    if [ -n "$app_id" ]; then
        health="$(
            docker inspect \
                --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
                "$app_id"
        )"
        if [ "$health" = "healthy" ]; then
            compose exec \
                --no-TTY \
                proxy \
                wget -q -O - http://127.0.0.1/health/ready
            echo
            compose ps
            trap - EXIT INT TERM
            echo "deployed $image"
            exit 0
        fi
        if [ "$health" = "unhealthy" ] || [ "$health" = "exited" ]; then
            echo "application became $health during deployment" >&2
            exit 1
        fi
    fi
    attempt=$((attempt + 1))
    sleep 5
done

echo "application did not become healthy before the timeout" >&2
exit 1
