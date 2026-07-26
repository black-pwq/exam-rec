# Exam Recognition

Exam Recognition exposes a FastAPI service that accepts PDF question books,
recognizes their pages, and persists incremental and final structured results.

## Development

The default development environment uses PaddlePaddle 3.2.0 for CPU:

```bash
uv sync
uv run pytest
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

On a Linux x86_64 machine with a CUDA 11.8-compatible NVIDIA runtime:

```bash
uv sync --no-default-groups --group paddle-cu118 --group dev --locked
uv run --no-sync python -c \
  "import paddle; print(paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
uv run --no-sync pytest
```

## Production

Production deployment uses Docker Compose, one directly published Uvicorn
process, and persistent volumes for jobs and PaddleOCR models. CPU and CUDA
11.8 images are built from the same lockfile.

```bash
cp deploy/env.production.example deploy/.env.production
chmod 600 deploy/.env.production

# Edit the environment file, then deploy one runtime:
./deploy/release.sh cpu
# or
./deploy/release.sh gpu
```

The runtime container invokes `/app/.venv/bin/uvicorn` directly; `uv` is only
used while building the image. See
[docs/deployment.md](docs/deployment.md) for prerequisites, configuration,
upgrades, rollback, persistence, and operational checks.

## API

The Web API contract and examples are documented in
[docs/exam_rec_web_api.md](docs/exam_rec_web_api.md).

## References

- [PaddleOCR](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html)
- [PaddlePaddle installation](https://www.paddleocr.ai/latest/version3.x/paddlepaddle_installation.html)
