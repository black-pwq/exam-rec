from __future__ import annotations

import os
from pathlib import Path

import pytest
from exam_rec.environment import load_project_environment


def test_project_environment_loads_defaults_without_overriding_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "EXAM_REC_LOG_LEVEL=DEBUG\nEXAM_REC_PORT=9000\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("EXAM_REC_LOG_LEVEL", raising=False)
    monkeypatch.setenv("EXAM_REC_PORT", "8080")

    assert load_project_environment(dotenv)

    assert os.environ["EXAM_REC_LOG_LEVEL"] == "DEBUG"
    assert os.environ["EXAM_REC_PORT"] == "8080"
