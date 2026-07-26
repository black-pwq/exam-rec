# TODO
LlmRegexExtractor sample range adapter
RawTextExtractor -> transform

## Paddle runtime

The default development environment uses PaddlePaddle 3.2.0 for CPU:

```bash
uv sync
```

On a Linux server with a CUDA 11.8-compatible NVIDIA runtime, select the
mutually exclusive GPU dependency group:

```bash
uv sync --no-default-groups --group paddle-cu118 --locked
uv run --no-sync python -c \
  "import paddle; print(paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
uv run --no-sync uvicorn api:app --host 0.0.0.0 --port 8000
```

To run tests against the GPU environment, include the development group:

```bash
uv sync --no-default-groups --group paddle-cu118 --group dev
uv run --no-sync pytest
```

PaddleOCR selects the available GPU automatically when the CUDA-enabled
PaddlePaddle wheel can see one, and otherwise uses CPU.

## REFERENCE
- PaddleOCR
https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html
- PaddlePaddle
https://www.paddleocr.ai/latest/version3.x/paddlepaddle_installation.html
