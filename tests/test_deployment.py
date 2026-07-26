from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy.drain_check import find_active_jobs
from deploy.preflight import ensure_writable_directory, validate_paddle_runtime


class FakePaddle:
    def __init__(self, *, cuda: bool, gpu_count: int) -> None:
        self.__version__ = "test"
        self._cuda = cuda
        self.device = SimpleNamespace(
            cuda=SimpleNamespace(device_count=lambda: gpu_count)
        )

    def is_compiled_with_cuda(self) -> bool:
        return self._cuda


def test_preflight_accepts_matching_cpu_and_gpu_runtimes() -> None:
    cpu = validate_paddle_runtime(
        "cpu",
        FakePaddle(cuda=False, gpu_count=0),
    )
    gpu = validate_paddle_runtime(
        "gpu",
        FakePaddle(cuda=True, gpu_count=2),
    )

    assert cpu.runtime == "cpu"
    assert not cpu.cuda_compiled
    assert gpu.runtime == "gpu"
    assert gpu.visible_gpu_count == 2


@pytest.mark.parametrize(
    "runtime,paddle",
    [
        ("gpu", FakePaddle(cuda=False, gpu_count=0)),
        ("gpu", FakePaddle(cuda=True, gpu_count=0)),
        ("cpu", FakePaddle(cuda=True, gpu_count=1)),
    ],
)
def test_preflight_rejects_runtime_mismatches(
    runtime: str,
    paddle: FakePaddle,
) -> None:
    with pytest.raises(RuntimeError):
        validate_paddle_runtime(runtime, paddle)


def test_preflight_creates_and_checks_writable_directories(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "cache"

    ensure_writable_directory(destination)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def write_status(root: Path, job_id: str, status: str) -> None:
    job_dir = root / job_id
    job_dir.mkdir()
    (job_dir / "status.json").write_text(
        json.dumps({"job_id": job_id, "status": status}),
        encoding="utf-8",
    )


def test_drain_check_only_reports_active_jobs(tmp_path: Path) -> None:
    write_status(tmp_path, "a" * 32, "completed")
    write_status(tmp_path, "b" * 32, "running")
    write_status(tmp_path, "c" * 32, "queued")

    assert find_active_jobs(tmp_path) == [
        {"job_id": "b" * 32, "status": "running"},
        {"job_id": "c" * 32, "status": "queued"},
    ]
