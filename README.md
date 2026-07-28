# Exam Recognition

Exam Recognition exposes a FastAPI service that accepts PDF question books,
recognizes their pages, and persists incremental and final structured results.

## Development

The default development environment uses PaddlePaddle 3.2.0 for CPU:

```bash
cp .env.example .env
uv sync
uv run pytest
uv run uvicorn --app-dir src exam_rec.main:app --host 127.0.0.1 --port 8000
```

On a Linux x86_64 machine with a CUDA 11.8-compatible NVIDIA runtime:

```bash
uv sync --no-default-groups --group paddle-cu118 --group dev --locked
uv run --no-sync python -c \
  "import paddle; print(paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
uv run --no-sync pytest
```

## OCR on Colab

Colab can be used only as the PaddleOCR GPU worker. Install the CLI and
authenticate once:

```bash
uv tool install google-colab-cli \
  --with jupyter-kernel-client==0.15.0

# One-time ADC login if it has not already been configured:
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

For several PDFs, start one reusable session. This installs dependencies,
downloads the models, and keeps the loaded PaddleOCR predictor in the remote
kernel:

```bash
./scripts/colab_ocr_session.sh start exam-ocr

./scripts/colab_ocr.sh --session exam-ocr ./first.pdf ./first.ocr.jsonl
./scripts/colab_ocr.sh --session exam-ocr ./second.pdf ./second.ocr.jsonl

./scripts/colab_ocr_session.sh stop exam-ocr
```

The reusable session continues consuming Colab quota until it is stopped. For
one PDF, the original one-off command still creates, initializes, and always
stops a temporary session:

```bash
./scripts/colab_ocr.sh ./input.pdf ./input.ocr.jsonl
```

For a large batch, use the checkpointed manifest runner. It splits each PDF
into page chunks, skips already-valid results, resumes completed chunks after
a Colab runtime expires, and optionally stops the session when the batch ends:

```bash
.venv/bin/python scripts/colab_ocr_batch.py \
  scripts/question_bank_ocr_manifest.json \
  --input-root '/mnt/d/Temp/正在使用的题库' \
  --output-root ./output \
  --session exam-ocr \
  --session-start-retries 60 \
  --stop-session
```

The result uses the same versioned persistence format as `PersistingOcr` and
can be read locally without running PaddleOCR:

```python
from exam_rec.ocr.cached_ocr import CachedOcr

pages = CachedOcr().predict("./input.ocr.jsonl")
```

The remote `/content` filesystem is temporary. Both modes treat each local
download as the durable copy and refuse to overwrite an existing output file.
Pass the output path as the final argument; standard output contains progress
logs and is not the OCR JSONL stream. Uploads use verified small chunks and
retry transient connection resets automatically.

## Production

Production deployment uses Docker Compose, one directly published Uvicorn
process, and persistent volumes for jobs and PaddleOCR models. CPU and CUDA
11.8 images are built from the same lockfile.

```bash
cp .env.example .env
chmod 600 .env

# Edit .env, then deploy the CPU runtime:
docker compose up --detach --build

# NVIDIA GPU runtime:
docker compose \
  -f compose.yaml \
  -f compose.gpu.yaml \
  up --detach --build
```

Run Compose commands from the project root so the same top-level `.env` file
provides interpolation values and container environment variables. The runtime
container invokes `/app/.venv/bin/uvicorn` directly; `uv` is only used while
building the image. See [docs/deployment.md](docs/deployment.md) for
prerequisites, configuration, persistence, and operational checks.

## API

The Web API contract and examples are documented in
[docs/exam_rec_web_api.md](docs/exam_rec_web_api.md).

## References

- [PaddleOCR](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html)
- [PaddlePaddle installation](https://www.paddleocr.ai/latest/version3.x/paddlepaddle_installation.html)
