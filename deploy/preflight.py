from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from recognition_jobs import LlmSettings


@dataclass(frozen=True)
class RuntimeInfo:
    runtime: str
    paddle_version: str
    cuda_compiled: bool
    visible_gpu_count: int


def ensure_writable_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile(
            mode="w",
            prefix=".exam-rec-write-test-",
            dir=path,
            encoding="utf-8",
        ) as temporary:
            temporary.write("ok")
            temporary.flush()
    except OSError as error:
        raise RuntimeError(f"directory is not writable: {path}: {error}") from error


def validate_paddle_runtime(
    runtime: str,
    paddle_module: Any | None = None,
) -> RuntimeInfo:
    normalized = runtime.strip().lower()
    if normalized not in {"cpu", "gpu"}:
        raise RuntimeError("EXAM_REC_RUNTIME must be either 'cpu' or 'gpu'")

    paddle = paddle_module or importlib.import_module("paddle")
    cuda_compiled = bool(paddle.is_compiled_with_cuda())
    visible_gpu_count = (
        int(paddle.device.cuda.device_count()) if cuda_compiled else 0
    )

    if normalized == "gpu" and (not cuda_compiled or visible_gpu_count < 1):
        raise RuntimeError(
            "GPU runtime requires a CUDA-enabled Paddle build and at least "
            "one visible GPU"
        )
    if normalized == "cpu" and cuda_compiled:
        raise RuntimeError(
            "CPU runtime requires the CPU Paddle build; refusing implicit "
            "GPU-to-CPU fallback"
        )

    return RuntimeInfo(
        runtime=normalized,
        paddle_version=str(paddle.__version__),
        cuda_compiled=cuda_compiled,
        visible_gpu_count=visible_gpu_count,
    )


def run_preflight() -> RuntimeInfo:
    LlmSettings.from_env()
    job_root = Path(os.getenv("EXAM_REC_JOB_ROOT", "var/jobs")).expanduser()
    model_root = Path(
        os.getenv("PADDLE_PDX_CACHE_HOME", "~/.paddlex")
    ).expanduser()
    ensure_writable_directory(job_root)
    ensure_writable_directory(model_root)
    return validate_paddle_runtime(os.getenv("EXAM_REC_RUNTIME", "cpu"))


def main() -> None:
    info = run_preflight()
    print(
        json.dumps(
            {
                "status": "ok",
                "job_root": os.getenv("EXAM_REC_JOB_ROOT", "var/jobs"),
                "model_cache": os.getenv(
                    "PADDLE_PDX_CACHE_HOME",
                    "~/.paddlex",
                ),
                **asdict(info),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
