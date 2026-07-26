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
if ! grep -Eq '^EXAM_REC_LLM_API_KEY=.+$' deploy/.env.production; then
    echo "deploy/.env.production must define a non-empty EXAM_REC_LLM_API_KEY" >&2
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
rollback_enabled=false
app_id="$(compose ps --quiet app)"
if [ -n "$app_id" ]; then
    previous_image="$(docker inspect --format '{{.Config.Image}}' "$app_id")"
fi

rollback() {
    exit_code=$?
    trap - EXIT INT TERM
    if [ "$exit_code" -ne 0 ] \
        && [ "$rollback_enabled" = true ] \
        && [ -n "$previous_image" ]; then
        echo "release failed; restoring $previous_image" >&2
        EXAM_REC_IMAGE="$previous_image"
        export EXAM_REC_IMAGE
        compose up --detach --no-build --remove-orphans || true
    fi
    exit "$exit_code"
}
trap rollback EXIT INT TERM

if [ -n "$app_id" ]; then
    echo "waiting for active jobs; keep new submissions paused" >&2
    compose run \
        --rm \
        --no-deps \
        app \
        python -m deploy.drain_check --wait --timeout 3600 --interval 5
    rollback_enabled=true
    compose stop app
fi

# Remove the proxy container left by releases made before Nginx was removed.
legacy_proxy_ids="$(
    docker ps \
        --all \
        --quiet \
        --filter label=com.docker.compose.project=exam-rec \
        --filter label=com.docker.compose.service=proxy
)"
for legacy_proxy_id in $legacy_proxy_ids; do
    docker rm --force "$legacy_proxy_id"
done

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
            compose exec --no-TTY app python -m deploy.healthcheck
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
